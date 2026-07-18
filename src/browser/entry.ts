/**
 * Browser core bundle entry. Re-exports the EXACT codec + crypto that pass the
 * Node gates, so the cross-arch tester (Chunk B) runs the same determinism-
 * critical code — not a re-implementation that could drift into a stub. esbuild
 * bundles this to an IIFE that assigns `window.ChaffCore`.
 *
 * The only browser-specific piece (the GPT-2 logit source via transformers.js)
 * lives in the HTML, because it needs the CDN module import. Everything below is
 * environment-agnostic and identical to the code under test in Node.
 */
import pinsJson from "../../pins.json" with { type: "json" };
import type { LogitSource } from "../inference/types.js";
import { planStep, encode, decode, DivergenceError, type CoderConfig, type StepTrace } from "../codec/coder.js";
import { fixedPointSoftmax, entropyBits } from "../fixedpoint.js";
import { hashInts } from "../util/hash.js";
import { BitReader, BitWriter } from "../util/bits.js";
import { seal, open, type CryptoParams } from "../crypto/aead.js";
import { localize } from "../harness/localizer.js";
import { SplitMix64 } from "../util/prng.js";

const pins = pinsJson as unknown as {
  codec: { logitScale: number; bucketWidth: number; maxBitsPerStep: number; softmaxPrecBits: number };
  crypto: { kdfIterations: number; saltBytes: number; nonceBytes: number; keyBits: number };
  payload: { magic: string; lengthFieldBits: number; maxSecretBytes: number };
};

function hashOf(a: ArrayLike<number>): string {
  return hashInts(a);
}

function coderConfig(bucketWidthOverride?: number): CoderConfig {
  return {
    logitScale: pins.codec.logitScale,
    bucketWidth: bucketWidthOverride ?? pins.codec.bucketWidth,
    maxBitsPerStep: pins.codec.maxBitsPerStep,
    softmaxPrecBits: pins.codec.softmaxPrecBits,
  };
}

function cryptoParams(): CryptoParams {
  return {
    kdfIterations: pins.crypto.kdfIterations,
    saltBytes: pins.crypto.saltBytes,
    nonceBytes: pins.crypto.nonceBytes,
    keyBits: pins.crypto.keyBits,
  };
}

function framePayload(blob: Uint8Array): { bytes: Uint8Array; bitLength: number } {
  const w = new BitWriter();
  w.writeBits(blob.length, pins.payload.lengthFieldBits);
  for (const b of blob) w.writeBits(b, 8);
  return w.finish();
}

function unframePayload(bits: Uint8Array, bitLength: number): Uint8Array {
  const r = new BitReader(bits, bitLength);
  const blobLen = r.readBits(pins.payload.lengthFieldBits);
  const blob = new Uint8Array(blobLen);
  for (let i = 0; i < blobLen; i++) blob[i] = r.readBits(8);
  return blob;
}

/**
 * Async logit source for the browser: GPT-2's forward is async, so the source's
 * logits() may return a Promise. These loops mirror codec/coder.ts's encode /
 * decode EXACTLY (same bit handling, same frame) and reuse the identical
 * planStep decision core; the only difference is `await source.logits(...)`.
 * `scripts/parity.ts` proves in Node that, driven by the same reference LM,
 * these produce byte-identical results to the sync gate path — so this is not a
 * divergent re-implementation.
 */
export interface AsyncLogitSource {
  vocabSize: number;
  id: string;
  weightHash: string;
  logits(prefix: number[]): Float64Array | Promise<Float64Array>;
}

async function asyncEncode(
  source: AsyncLogitSource,
  payload: Uint8Array,
  payloadBitLength: number,
  cfg: CoderConfig,
): Promise<{ cover: number[]; trace: StepTrace[] }> {
  const reader = new BitReader(payload, payloadBitLength);
  const prefix: number[] = [];
  const cover: number[] = [];
  const trace: StepTrace[] = [];
  const maxSteps = payloadBitLength * 8 + 64;
  let step = 0;
  while (reader.remaining > 0) {
    if (step >= maxSteps) throw new Error("encode exceeded step cap — model capacity too low for this bucketWidth.");
    const plan = planStep(await source.logits(prefix), cfg);
    const j = plan.k > 0 ? reader.readBits(plan.k) : 0;
    const tokenId = plan.ranking[j]!;
    cover.push(tokenId);
    trace.push(traceOf(step, prefix.length, tokenId, plan, j));
    prefix.push(tokenId);
    step++;
  }
  return { cover, trace };
}

async function asyncDecode(
  source: AsyncLogitSource,
  cover: number[],
  cfg: CoderConfig,
): Promise<{ bytes: Uint8Array; bitLength: number; trace: StepTrace[] }> {
  const prefix: number[] = [];
  const w = new BitWriter();
  const trace: StepTrace[] = [];
  for (let step = 0; step < cover.length; step++) {
    const plan = planStep(await source.logits(prefix), cfg);
    const observed = cover[step]!;
    const j = plan.ranking.indexOf(observed);
    if (j < 0) throw new DivergenceError(step, "tokenizer", `cover token ${observed} not in vocabulary at step ${step}`);
    if (plan.k > 0 && j >= 1 << plan.k)
      throw new DivergenceError(step, "bucket", `observed token rank ${j} outside usable set (k=${plan.k}) at step ${step}`);
    if (plan.k > 0) w.writeBits(j, plan.k);
    trace.push(traceOf(step, prefix.length, observed, plan, j));
    prefix.push(observed);
  }
  const fin = w.finish();
  return { bytes: fin.bytes, bitLength: fin.bitLength, trace };
}

function traceOf(index: number, prefixLen: number, tokenId: number, plan: ReturnType<typeof planStep>, selectedRank: number): StepTrace {
  const topN = Math.min(plan.ranking.length, 64);
  return {
    index,
    prefixLen,
    tokenId,
    logitsHash: hashOf(plan.quant),
    softmaxHash: hashOf(plan.probs),
    rankingHash: hashOf(plan.ranking.slice(0, topN)),
    k: plan.k,
    selectedRank,
    entropyBits: entropyBits(plan.probs, plan.denom),
  };
}

export interface EncodeOut {
  cover: number[];
  trace: StepTrace[];
  blobBytes: number;
  payloadBits: number;
}

/** secret + password → cover token ids (driving `source`, possibly async). */
export async function encodeSecret(
  source: AsyncLogitSource,
  password: string,
  secret: Uint8Array,
  bucketWidth?: number,
): Promise<EncodeOut> {
  const aad = new TextEncoder().encode(pins.payload.magic);
  const blob = await seal(password, secret, aad, cryptoParams());
  const framed = framePayload(blob);
  const r = await asyncEncode(source, framed.bytes, framed.bitLength, coderConfig(bucketWidth));
  return { cover: r.cover, trace: r.trace, blobBytes: blob.length, payloadBits: framed.bitLength };
}

export interface DecodeOut {
  secret: Uint8Array;
  trace: StepTrace[];
}

/** cover token ids + password → secret (re-running `source`). Throws on tag/decode failure. */
export async function decodeSecret(
  source: AsyncLogitSource,
  password: string,
  cover: number[],
  bucketWidth?: number,
): Promise<DecodeOut> {
  const dec = await asyncDecode(source, cover, coderConfig(bucketWidth));
  const blob = unframePayload(dec.bytes, dec.bitLength);
  const aad = new TextEncoder().encode(pins.payload.magic);
  const secret = await open(password, blob, aad, cryptoParams());
  return { secret, trace: dec.trace };
}

/**
 * Deterministic mock logit source — lets the tester prove the codec + crypto
 * round-trip OFFLINE (no GPT-2 download) with the same code path. Peaked,
 * well-separated logits so capacity is healthy. This is the same class of check
 * that passes G1 in Node; it is NOT the cross-arch test (that needs real GPT-2).
 */
export class MockLogitSource implements LogitSource {
  readonly vocabSize: number;
  readonly id = "chaff-mock@1";
  readonly weightHash = "mock";
  constructor(vocab = 256) {
    this.vocabSize = vocab;
  }
  logits(prefix: number[]): Float64Array {
    let seed = 0x9e3779b9n;
    for (let i = 0; i < prefix.length; i++) seed = (seed * 1000003n + BigInt(prefix[i]! + 1)) & ((1n << 64n) - 1n);
    const rng = new SplitMix64(seed);
    const out = new Float64Array(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) out[i] = rng.nextGaussian() * 4.0;
    return out;
  }
}

export const ChaffCore = {
  encodeSecret,
  decodeSecret,
  asyncEncode,
  asyncDecode,
  planStep,
  encode,
  decode,
  localize,
  fixedPointSoftmax,
  entropyBits,
  seal,
  open,
  BitReader,
  BitWriter,
  DivergenceError,
  MockLogitSource,
  coderConfig,
  cryptoParams,
  pins,
};

// Attach to window for the classic-script inline bundle.
(globalThis as unknown as Record<string, unknown>).ChaffCore = ChaffCore;
