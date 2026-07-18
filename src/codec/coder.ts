/**
 * Rank-bucket steganographic coder — determinism mandate #3.
 *
 * At each token step:
 *   1. Read raw logits from the model for the current prefix.
 *   2. Quantize: q[i] = round(logit[i] * logitScale)  (integer domain).
 *   3. argsort(q) descending, tie-break by token id ascending  → ranking.
 *   4. Capacity k = the largest number of bits such that the top 2^k ranked
 *      candidates are ALL separated by adjacent quantized gaps >= bucketWidth.
 *      Any per-arch logit noise below bucketWidth/(2*logitScale) therefore
 *      cannot reorder the coding set, so the decoder reconstructs the same
 *      ranking and the same k. Bucket width is the robustness/capacity dial.
 *   5. Encode: read k payload bits → index j; emit ranking[j].
 *      Decode: find the observed token's rank j; output k bits.
 *
 * The scheme round-trips bit-exact whenever the (quantized) distribution is
 * identical on both sides — which is the entire cross-arch thesis. The
 * fixed-point softmax is computed for capacity reporting only; token selection
 * runs on the quantized logits directly, so the backend's float softmax is
 * never on the coding path.
 */
import type { LogitSource } from "../inference/types.js";
import { BitReader, BitWriter } from "../util/bits.js";
import { fixedPointSoftmax, entropyBits } from "../fixedpoint.js";
import { hashInts } from "../util/hash.js";

export interface CoderConfig {
  logitScale: number;
  bucketWidth: number;
  maxBitsPerStep: number;
  softmaxPrecBits: number;
  /** Optional hard ceiling on emitted tokens to guard against a zero-capacity model. */
  maxSteps?: number;
}

export type Stage = "tokenizer" | "logit" | "softmax" | "bucket" | "none";

export interface StepTrace {
  index: number;
  prefixLen: number;
  tokenId: number;
  logitsHash: string;
  softmaxHash: string;
  rankingHash: string;
  k: number;
  selectedRank: number;
  entropyBits: number;
}

export interface StepPlan {
  quant: Int32Array;
  ranking: number[]; // token ids, most-likely first
  k: number;
  probs: number[];
  denom: number;
}

export class DivergenceError extends Error {
  constructor(
    readonly index: number,
    readonly stage: Stage,
    message: string,
  ) {
    super(message);
    this.name = "DivergenceError";
  }
}

function quantize(logits: Float64Array, logitScale: number): Int32Array {
  const q = new Int32Array(logits.length);
  for (let i = 0; i < logits.length; i++) q[i] = Math.round(logits[i]! * logitScale) | 0;
  return q;
}

/** Snap a quantized logit onto the bucket grid: integer bucket id. */
function bucketId(q: number, width: number): number {
  // Symmetric rounding toward nearest bucket; deterministic for negatives.
  return Math.round(q / width);
}

/** Deterministic plan for one step: ranking + capacity + fixed-point probs.
 *
 * The coding decision runs on BUCKETED logits (snap-to-grid of width
 * bucketWidth), which is the literal "fixed-point-bucketed gaps" of mandate #3:
 * any per-arch logit noise that does not push a value across a bucket boundary
 * leaves the bucket id — and therefore the whole decision — unchanged. Only the
 * top 2^k candidates, each sitting in a strictly separated bucket, are used to
 * carry bits, so a used candidate cannot be reordered without a full bucket
 * crossing.
 */
export function planStep(logits: Float64Array, cfg: CoderConfig): StepPlan {
  const quant = quantize(logits, cfg.logitScale);
  const V = quant.length;
  const W = cfg.bucketWidth;

  const bucket = new Int32Array(V);
  for (let i = 0; i < V; i++) bucket[i] = bucketId(quant[i]!, W);

  // argsort by bucket descending, tie-break by token id ascending.
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => (bucket[b]! - bucket[a]!) || (a - b));

  // Capacity: largest k such that the top 2^k candidates occupy strictly
  // descending distinct buckets (adjacent bucket ids differ by >= 1).
  const maxK = Math.min(cfg.maxBitsPerStep, Math.floor(Math.log2(V)));
  let k = 0;
  for (let cand = 1; cand <= maxK; cand++) {
    const m = 1 << cand;
    let ok = true;
    for (let j = 0; j < m - 1; j++) {
      if (bucket[idx[j]!]! - bucket[idx[j + 1]!]! < 1) {
        ok = false;
        break;
      }
    }
    if (ok) k = cand;
    else break;
  }

  const sm = fixedPointSoftmax(Array.from(quant), {
    logitScale: cfg.logitScale,
    softmaxPrecBits: cfg.softmaxPrecBits,
  });

  return { quant, ranking: idx, k, probs: sm.probs, denom: sm.denom };
}

function makeTrace(index: number, prefixLen: number, tokenId: number, plan: StepPlan, selectedRank: number): StepTrace {
  const topN = Math.min(plan.ranking.length, 64);
  return {
    index,
    prefixLen,
    tokenId,
    logitsHash: hashInts(plan.quant),
    softmaxHash: hashInts(plan.probs),
    rankingHash: hashInts(plan.ranking.slice(0, topN)),
    k: plan.k,
    selectedRank,
    entropyBits: entropyBits(plan.probs, plan.denom),
  };
}

export interface EncodeResult {
  cover: number[];
  trace: StepTrace[];
  bitsCarried: number;
  steps: number;
}

/**
 * Encode a payload bitstream into cover tokens. Emits exactly enough tokens to
 * carry all `payloadBitLength` bits (the final step is zero-padded).
 */
export function encode(
  source: LogitSource,
  payload: Uint8Array,
  payloadBitLength: number,
  cfg: CoderConfig,
  seedPrefix: number[] = [],
): EncodeResult {
  const reader = new BitReader(payload, payloadBitLength);
  const prefix = seedPrefix.slice();
  const cover: number[] = [];
  const trace: StepTrace[] = [];
  const maxSteps = cfg.maxSteps ?? payloadBitLength * 8 + 64;

  let step = 0;
  while (reader.remaining > 0) {
    if (step >= maxSteps) {
      throw new Error(
        `encode exceeded ${maxSteps} steps with ${reader.remaining} bits left — model capacity too low for this bucketWidth.`,
      );
    }
    const logits = source.logits(prefix);
    const plan = planStep(logits, cfg);
    const k = plan.k;
    const j = k > 0 ? reader.readBits(k) : 0;
    const tokenId = plan.ranking[j]!;
    cover.push(tokenId);
    trace.push(makeTrace(step, prefix.length, tokenId, plan, j));
    prefix.push(tokenId);
    step++;
  }

  return { cover, trace, bitsCarried: payloadBitLength, steps: step };
}

export interface DecodeResult {
  bytes: Uint8Array;
  bitLength: number;
  trace: StepTrace[];
}

/**
 * Decode cover tokens back into a bitstream. `wantBits` (if given) stops once
 * that many bits have been recovered; otherwise every cover token is consumed.
 */
export function decode(
  source: LogitSource,
  cover: number[],
  cfg: CoderConfig,
  wantBits: number | undefined,
  seedPrefix: number[] = [],
): DecodeResult {
  const prefix = seedPrefix.slice();
  const writer = new BitWriter();
  const trace: StepTrace[] = [];

  for (let step = 0; step < cover.length; step++) {
    if (wantBits !== undefined && writer.bitLength >= wantBits) break;
    const logits = source.logits(prefix);
    const plan = planStep(logits, cfg);
    const observed = cover[step]!;
    const j = plan.ranking.indexOf(observed);
    if (j < 0) {
      throw new DivergenceError(step, "tokenizer", `cover token ${observed} not in vocabulary at step ${step}`);
    }
    const usable = 1 << plan.k;
    if (plan.k > 0 && j >= usable) {
      throw new DivergenceError(
        step,
        "bucket",
        `observed token rank ${j} outside usable set (k=${plan.k}, top ${usable}) at step ${step}`,
      );
    }
    if (plan.k > 0) writer.writeBits(j, plan.k);
    trace.push(makeTrace(step, prefix.length, observed, plan, j));
    prefix.push(observed);
  }

  const fin = writer.finish();
  return { bytes: fin.bytes, bitLength: fin.bitLength, trace };
}
