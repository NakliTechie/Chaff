/**
 * Entropy coder — Huffman over top-K candidates (nethical6-style).
 *
 * Replaces the rank-bucket coder's fixed bits/token with a variable-length code
 * that tracks the model's own distribution, so cover length ≈ payloadBits /
 * H(distribution) instead of payloadBits / (1–2 bits). Selection follows model
 * probability (Huffman code length ≈ -log2 p) and the payload is AES ciphertext
 * (uniform bits), so the emitted tokens read as natural model sampling.
 *
 * DETERMINISM: reversibility requires both peers to build the *identical*
 * candidate set and Huffman tree at every step. That holds only when both run
 * the same model + runtime (same-arch). This coder deliberately trades the
 * rank-bucket coder's cross-arch robustness for capacity + natural cover.
 *
 * Termination is safe: every step's tree has ≥ 2 leaves, so each emitted token
 * consumes ≥ 1 payload bit; the encoder loops until all payload bits are
 * consumed (the final token may read a few zero-pad bits past the end, which the
 * decoder reproduces and `unframePayload` ignores).
 */
import type { LogitSource } from "../inference/types.js";
import { BitReader, BitWriter } from "../util/bits.js";
import { fixedPointSoftmax } from "../fixedpoint.js";
import type { CoderConfig, StepTrace, EncodeResult, DecodeResult } from "./coder.js";
import { DivergenceError } from "./coder.js";

const DEFAULT_TOP_K = 128;

interface HuffNode {
  weight: bigint;
  minId: number; // smallest token id in this subtree — unique per node → total order
  token?: number; // set on leaves
  left?: HuffNode;
  right?: HuffNode;
}

export interface StepModel {
  /** Root of the Huffman tree (internal node; ≥ 2 leaves). */
  root: HuffNode;
  /** token id → root→leaf code bits (0=left, 1=right). */
  codeOf: Map<number, number[]>;
}

export type { HuffNode };

/** Top-K token ids by (quantized logit desc, token id asc). K ≥ 2. */
function topKByLogit(logits: Float64Array, logitScale: number, topK: number): { ids: number[]; quant: number[] } {
  const V = logits.length;
  const K = Math.max(2, Math.min(topK, V));
  // Partial selection: keep the K best by (q desc, id asc). K is small (≤ few hundred).
  const bestIds: number[] = [];
  const bestQ: number[] = [];
  for (let id = 0; id < V; id++) {
    const q = Math.round(logits[id]! * logitScale) | 0;
    if (bestIds.length < K) {
      // insertion into the sorted-desc list
      let lo = 0,
        hi = bestIds.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (bestQ[mid]! > q || (bestQ[mid]! === q && bestIds[mid]! < id)) lo = mid + 1;
        else hi = mid;
      }
      bestIds.splice(lo, 0, id);
      bestQ.splice(lo, 0, q);
    } else if (q > bestQ[K - 1]! || (q === bestQ[K - 1]! && id < bestIds[K - 1]!)) {
      let lo = 0,
        hi = K - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (bestQ[mid]! > q || (bestQ[mid]! === q && bestIds[mid]! < id)) lo = mid + 1;
        else hi = mid;
      }
      bestIds.splice(lo, 0, id);
      bestQ.splice(lo, 0, q);
      bestIds.pop();
      bestQ.pop();
    }
  }
  return { ids: bestIds, quant: bestQ };
}

/** Deterministic Huffman tree + code table over the top-K candidates. */
export function buildStepModel(logits: Float64Array, cfg: CoderConfig): StepModel {
  const { ids, quant } = topKByLogit(logits, cfg.logitScale, cfg.topK ?? DEFAULT_TOP_K);
  // The termination guarantee (each step consumes >= 1 payload bit) requires a
  // tree with >= 2 leaves; a degenerate vocab of 1 would otherwise spin to the
  // step cap. Enforce it here rather than relying on the invariant holding.
  if (ids.length < 2) throw new Error("entropy coder needs a vocabulary of at least 2 tokens");
  // Integer weights ∝ probability, floored to ≥ 1 so every candidate is reachable.
  const sm = fixedPointSoftmax(quant, { logitScale: cfg.logitScale, softmaxPrecBits: cfg.softmaxPrecBits });
  let nodes: HuffNode[] = ids.map((id, i) => ({ weight: BigInt(sm.probs[i]! + 1), minId: id, token: id }));

  // Merge the two smallest by (weight asc, minId asc) until one root remains.
  while (nodes.length > 1) {
    nodes.sort((a, b) => (a.weight === b.weight ? a.minId - b.minId : a.weight < b.weight ? -1 : 1));
    const l = nodes.shift()!;
    const r = nodes.shift()!;
    nodes.push({ weight: l.weight + r.weight, minId: Math.min(l.minId, r.minId), left: l, right: r });
  }
  const root = nodes[0]!;

  const codeOf = new Map<number, number[]>();
  const walk = (node: HuffNode, path: number[]): void => {
    if (node.token !== undefined) {
      codeOf.set(node.token, path);
      return;
    }
    walk(node.left!, [...path, 0]);
    walk(node.right!, [...path, 1]);
  };
  walk(root, []);
  return { root, codeOf };
}

function traceOf(index: number, prefixLen: number, tokenId: number, codeLen: number): StepTrace {
  // Lightweight trace: enough for the divergence localizer to name the first
  // diverging token/step. No full-vocab hashing on this path.
  return {
    index,
    prefixLen,
    tokenId,
    logitsHash: "",
    softmaxHash: "",
    rankingHash: "",
    k: codeLen,
    selectedRank: 0,
    entropyBits: 0,
  };
}

/**
 * Encode a payload bitstream into cover tokens. Emits tokens until every payload
 * bit is consumed; the final token may read zero-pad bits past the end.
 */
export function encodeEntropy(
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
  const wantTrace = cfg.trace === true;
  const maxSteps = cfg.maxSteps ?? payloadBitLength + 64;

  let step = 0;
  while (reader.remaining > 0) {
    if (step >= maxSteps) {
      throw new Error(`encode exceeded ${maxSteps} steps — model capacity too low.`);
    }
    const model = buildStepModel(source.logits(prefix), cfg);
    // Walk the tree with payload bits (zero-padded past end) to a leaf.
    let node = model.root;
    let codeLen = 0;
    while (node.token === undefined) {
      node = reader.readBit() === 0 ? node.left! : node.right!;
      codeLen++;
    }
    const tokenId = node.token;
    cover.push(tokenId);
    if (wantTrace) trace.push(traceOf(step, prefix.length, tokenId, codeLen));
    prefix.push(tokenId);
    step++;
  }

  return { cover, trace, bitsCarried: payloadBitLength, steps: step };
}

/**
 * Decode cover tokens back into a bitstream by concatenating each token's
 * Huffman code. Trailing zero-pad bits from the last token are recovered too;
 * the framing layer extracts the payload by its length field and ignores them.
 */
export function decodeEntropy(
  source: LogitSource,
  cover: number[],
  cfg: CoderConfig,
  _wantBits: number | undefined,
  seedPrefix: number[] = [],
): DecodeResult {
  const prefix = seedPrefix.slice();
  const writer = new BitWriter();
  const trace: StepTrace[] = [];
  const wantTrace = cfg.trace === true;

  for (let step = 0; step < cover.length; step++) {
    const model = buildStepModel(source.logits(prefix), cfg);
    const observed = cover[step]!;
    const code = model.codeOf.get(observed);
    if (code === undefined) {
      throw new DivergenceError(step, "tokenizer", `cover token ${observed} not in top-K candidate set at step ${step}`);
    }
    for (const bit of code) writer.writeBit(bit);
    if (wantTrace) trace.push(traceOf(step, prefix.length, observed, code.length));
    prefix.push(observed);
  }

  const fin = writer.finish();
  return { bytes: fin.bytes, bitLength: fin.bitLength, trace };
}
