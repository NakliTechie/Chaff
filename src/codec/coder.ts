/**
 * Rank-bucket steganographic coder — determinism mandate #3.
 *
 * At each token step:
 *   1. Read raw logits from the model for the current prefix.
 *   2. Quantize: q[i] = round(logit[i] * logitScale)  (integer domain).
 *   3. Order by bucket descending, tie-break by token id ascending  → ranking.
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
 * fixed-point softmax is computed for capacity reporting / diagnostic traces
 * only; token selection runs on the quantized logits directly, so the
 * backend's float softmax is never on the coding path.
 *
 * Performance: the coding path never does a full O(V log V) argsort. Encode
 * selects only the top 2^maxBitsPerStep candidates (≤ 64) in O(V · M);
 * decode finds the observed token's rank by an O(V) outrank count. Diagnostic
 * work (fixed-point softmax + makeTrace) is opt-in via `cfg.trace`.
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
  /**
   * When true, compute per-step diagnostic traces (hashes, entropy) and the
   * fixed-point softmax those traces depend on. Default false — the app/coding
   * path skips all of that work. Gates that need the divergence localizer
   * (G5, fault injection) must pass `trace: true`.
   */
  trace?: boolean;
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
  /** Token ids ordered most-likely first. Only the top 2^maxK entries are
   *  materialised — enough for coding (ranking[j] for j < 2^k) and for the
   *  diagnostic rankingHash (top ≤ 64). */
  ranking: number[];
  k: number;
  /** Empty when `cfg.trace` is not set (softmax is diagnostic-only). */
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

function fillBuckets(quant: Int32Array, width: number): Int32Array {
  const V = quant.length;
  const bucket = new Int32Array(V);
  for (let i = 0; i < V; i++) bucket[i] = bucketId(quant[i]!, width);
  return bucket;
}

/**
 * Top-M token ids by (bucket desc, token-id asc) — the prefix of a full
 * argsort under the same key. O(V · M) binary-insertion selection; M ≤ 64.
 */
function selectTopM(bucket: Int32Array, M: number): number[] {
  const V = bucket.length;
  if (M <= 0 || V === 0) return [];
  const m = M < V ? M : V;
  const top: number[] = new Array(m);
  const topB: number[] = new Array(m);
  let size = 0;

  for (let id = 0; id < V; id++) {
    const b = bucket[id]!;
    if (size < m) {
      // Binary search insertion into the sorted top[0..size).
      let lo = 0;
      let hi = size;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const mb = topB[mid]!;
        const midId = top[mid]!;
        // mid is strictly better than id → id goes after mid.
        if (mb > b || (mb === b && midId < id)) lo = mid + 1;
        else hi = mid;
      }
      for (let i = size; i > lo; i--) {
        top[i] = top[i - 1]!;
        topB[i] = topB[i - 1]!;
      }
      top[lo] = id;
      topB[lo] = b;
      size++;
    } else {
      // Must strictly outrank the current worst (last) to enter.
      const lastB = topB[m - 1]!;
      const lastId = top[m - 1]!;
      if (!(b > lastB || (b === lastB && id < lastId))) continue;
      let lo = 0;
      let hi = m - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const mb = topB[mid]!;
        const midId = top[mid]!;
        if (mb > b || (mb === b && midId < id)) lo = mid + 1;
        else hi = mid;
      }
      for (let i = m - 1; i > lo; i--) {
        top[i] = top[i - 1]!;
        topB[i] = topB[i - 1]!;
      }
      top[lo] = id;
      topB[lo] = b;
    }
  }
  return top;
}

/**
 * Capacity k: largest cand ≤ maxK such that the top 2^cand candidates occupy
 * strictly descending distinct buckets (adjacent bucket ids differ by ≥ 1).
 * Identical rule to the historical full-argsort path.
 */
function capacityK(bucket: Int32Array, ranking: number[], maxK: number): number {
  let k = 0;
  for (let cand = 1; cand <= maxK; cand++) {
    const m = 1 << cand;
    if (m > ranking.length) break;
    let ok = true;
    for (let j = 0; j < m - 1; j++) {
      if (bucket[ranking[j]!]! - bucket[ranking[j + 1]!]! < 1) {
        ok = false;
        break;
      }
    }
    if (ok) k = cand;
    else break;
  }
  return k;
}

/**
 * Rank of `tokenId` under (bucket desc, id asc) — how many tokens outrank it.
 * O(V); bit-identical to `fullArgsort.indexOf(tokenId)`.
 */
export function rankOfToken(bucket: Int32Array, tokenId: number): number {
  const bObs = bucket[tokenId]!;
  let rank = 0;
  for (let id = 0; id < tokenId; id++) {
    // Smaller id: outranks if bucket ≥ observed.
    if (bucket[id]! >= bObs) rank++;
  }
  for (let id = tokenId + 1; id < bucket.length; id++) {
    // Larger id: outranks only with a strictly higher bucket.
    if (bucket[id]! > bObs) rank++;
  }
  return rank;
}

/** Rank of a token from quantized logits + bucket width (rebuilds the bucket grid). */
export function rankOfQuantized(quant: Int32Array, bucketWidth: number, tokenId: number): number {
  return rankOfToken(fillBuckets(quant, bucketWidth), tokenId);
}

/** Deterministic plan for one step: ranking + capacity (+ optional fixed-point probs).
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
  const bucket = fillBuckets(quant, cfg.bucketWidth);

  const maxK = Math.min(cfg.maxBitsPerStep, Math.floor(Math.log2(V)));
  // Need the top 2^maxK candidates for capacity; also covers rankingHash (≤ 64).
  const topM = 1 << maxK;
  const ranking = selectTopM(bucket, topM);
  const k = capacityK(bucket, ranking, maxK);

  if (cfg.trace) {
    // Diagnostic only — never on the app coding path.
    const quantArr = new Array<number>(V);
    for (let i = 0; i < V; i++) quantArr[i] = quant[i]!;
    const sm = fixedPointSoftmax(quantArr, {
      logitScale: cfg.logitScale,
      softmaxPrecBits: cfg.softmaxPrecBits,
    });
    return { quant, ranking, k, probs: sm.probs, denom: sm.denom };
  }

  return { quant, ranking, k, probs: [], denom: 0 };
}

/**
 * Shared per-step prep used by encode/decode: quantize, buckets, top-M ranking,
 * capacity. Avoids building a StepPlan object on the hot path when traces are off.
 */
function planCore(
  logits: Float64Array,
  cfg: CoderConfig,
): { quant: Int32Array; bucket: Int32Array; ranking: number[]; k: number } {
  const quant = quantize(logits, cfg.logitScale);
  const bucket = fillBuckets(quant, cfg.bucketWidth);
  const V = quant.length;
  const maxK = Math.min(cfg.maxBitsPerStep, Math.floor(Math.log2(V)));
  const ranking = selectTopM(bucket, 1 << maxK);
  const k = capacityK(bucket, ranking, maxK);
  return { quant, bucket, ranking, k };
}

function makeTrace(
  index: number,
  prefixLen: number,
  tokenId: number,
  plan: StepPlan,
  selectedRank: number,
): StepTrace {
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

function planFromCore(
  core: { quant: Int32Array; ranking: number[]; k: number },
  cfg: CoderConfig,
): StepPlan {
  if (!cfg.trace) {
    return { quant: core.quant, ranking: core.ranking, k: core.k, probs: [], denom: 0 };
  }
  const V = core.quant.length;
  const quantArr = new Array<number>(V);
  for (let i = 0; i < V; i++) quantArr[i] = core.quant[i]!;
  const sm = fixedPointSoftmax(quantArr, {
    logitScale: cfg.logitScale,
    softmaxPrecBits: cfg.softmaxPrecBits,
  });
  return { quant: core.quant, ranking: core.ranking, k: core.k, probs: sm.probs, denom: sm.denom };
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
  const wantTrace = cfg.trace === true;

  let step = 0;
  while (reader.remaining > 0) {
    if (step >= maxSteps) {
      throw new Error(
        `encode exceeded ${maxSteps} steps with ${reader.remaining} bits left — model capacity too low for this bucketWidth.`,
      );
    }
    const logits = source.logits(prefix);
    const core = planCore(logits, cfg);
    const k = core.k;
    const j = k > 0 ? reader.readBits(k) : 0;
    const tokenId = core.ranking[j]!;
    cover.push(tokenId);
    if (wantTrace) {
      trace.push(makeTrace(step, prefix.length, tokenId, planFromCore(core, cfg), j));
    }
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
  const wantTrace = cfg.trace === true;

  for (let step = 0; step < cover.length; step++) {
    if (wantBits !== undefined && writer.bitLength >= wantBits) break;
    const logits = source.logits(prefix);
    const core = planCore(logits, cfg);
    const observed = cover[step]!;
    const V = core.bucket.length;
    if (observed < 0 || observed >= V) {
      throw new DivergenceError(step, "tokenizer", `cover token ${observed} not in vocabulary at step ${step}`);
    }
    const j = rankOfToken(core.bucket, observed);
    const usable = 1 << core.k;
    if (core.k > 0 && j >= usable) {
      throw new DivergenceError(
        step,
        "bucket",
        `observed token rank ${j} outside usable set (k=${core.k}, top ${usable}) at step ${step}`,
      );
    }
    if (core.k > 0) writer.writeBits(j, core.k);
    if (wantTrace) {
      trace.push(makeTrace(step, prefix.length, observed, planFromCore(core, cfg), j));
    }
    prefix.push(observed);
  }

  const fin = writer.finish();
  return { bytes: fin.bytes, bitLength: fin.bitLength, trace };
}
