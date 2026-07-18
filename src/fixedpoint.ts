/**
 * Fixed-point softmax — the codec's OWN probability computation.
 *
 * Determinism mandate #2: the codec reads raw pre-softmax logits and computes
 * its own fixed-point softmax with integer math, a fixed denominator, and
 * deterministic rounding. The backend's float softmax is NEVER on the coding
 * path. Everything below is BigInt / integer arithmetic, so it is bit-identical
 * on every architecture and every JS engine.
 *
 * The rank-bucket coder (codec/coder.ts) selects tokens from the *quantized
 * logits* directly, so it does not even need probabilities — but capacity
 * reporting (G4, bits/token) does, and the mandate requires the codec to own a
 * fixed-point softmax rather than borrow the backend's. This module is that
 * softmax and the single source of truth for the probability view.
 */

/** log2(e) as a fixed-point value with 30 fractional bits: round(log2(e) * 2^30). */
const LOG2E_Q30 = 1549082004n; // log2(e) = 1.4426950408889634 → * 2^30
const Q30 = 30n;

/** 2^x minimax coefficients on x∈[0,1], scaled by 2^30 (Horner order high→low). */
const P0 = 1073741824n; // 1.0000000  * 2^30
const P1 = 744261230n; //  0.6931472  * 2^30
const P2 = 257852109n; //  0.2401598  * 2^30
const P3 = 59944301n; //   0.0558282  * 2^30
const P4 = 9652713n; //    0.0089893  * 2^30

/** Evaluate 2^g for g∈[0,1) given gQ30 = round(g * 2^30). Returns round(2^g * 2^30). */
function exp2FracQ30(gQ30: bigint): bigint {
  // Horner in Q30 fixed point: acc = ((((P4*g + P3)*g + P2)*g + P1)*g + P0)
  let acc = P4;
  acc = (acc * gQ30) >> Q30;
  acc += P3;
  acc = (acc * gQ30) >> Q30;
  acc += P2;
  acc = (acc * gQ30) >> Q30;
  acc += P1;
  acc = (acc * gQ30) >> Q30;
  acc += P0;
  return acc; // ≈ 2^g in Q30, within [2^30, 2^31]
}

export interface SoftmaxConfig {
  /** round(logit * logitScale) is the integer logit domain. */
  logitScale: number;
  /** Denominator exponent: probabilities sum to 2^softmaxPrecBits. */
  softmaxPrecBits: number;
}

export interface SoftmaxResult {
  /** Integer probabilities, sum exactly 2^softmaxPrecBits. */
  probs: number[];
  /** Sum denominator (2^softmaxPrecBits). */
  denom: number;
}

/**
 * Fixed-point softmax over quantized integer logits.
 * @param quantLogits round(rawLogit * logitScale), integers.
 */
export function fixedPointSoftmax(quantLogits: number[], cfg: SoftmaxConfig): SoftmaxResult {
  const n = quantLogits.length;
  const precBits = BigInt(cfg.softmaxPrecBits);
  const denomBig = 1n << precBits;
  if (n === 0) return { probs: [], denom: Number(denomBig) }; // empty distribution: nothing to normalize
  const scale = BigInt(cfg.logitScale);

  // Canonicalize inputs to int32 ONCE so the max and every element read identical
  // integer values (no max-vs-element coercion drift for out-of-range inputs).
  const q = new Int32Array(n);
  for (let i = 0; i < n; i++) q[i] = Math.round(quantLogits[i]!) | 0;
  let maxQ = q[0]!;
  for (let i = 1; i < n; i++) if (q[i]! > maxQ) maxQ = q[i]!;
  const maxQBig = BigInt(maxQ);

  // e[i] = exp((Q_i - Q_max)/scale) in Q(precBits) fixed point.
  const e: bigint[] = new Array(n);
  let sum = 0n;
  for (let i = 0; i < n; i++) {
    const dq = BigInt(q[i]!) - maxQBig; // <= 0
    // t = -dq/scale * log2(e) >= 0  (exponent for base-2)
    // tQ30 = (-dq) * LOG2E_Q30 / scale
    const negDq = -dq;
    const tQ30 = (negDq * LOG2E_Q30) / scale; // Q30, >= 0
    const iPart = tQ30 >> Q30; // integer part of t (number of halvings)
    const fPart = tQ30 - (iPart << Q30); // fractional part in Q30, [0,1)
    // 2^{-t} = 2^{-i} * 2^{-f} = 2^{-i} / 2^{f}
    const twoPowF = exp2FracQ30(fPart); // 2^f in Q30, [2^30, 2^31]
    // value = (2^precBits) * 2^{-i} / 2^f  → keep in Q(precBits)
    // numerator = denom << 30 ; divide by 2^f ; then shift right by iPart
    let val = (denomBig << Q30) / twoPowF; // = 2^{-f} in Q(precBits)
    if (iPart >= 64n) {
      val = 0n;
    } else {
      // round-shift right by iPart
      if (iPart > 0n) {
        const half = 1n << (iPart - 1n);
        val = (val + half) >> iPart;
      }
    }
    e[i] = val;
    sum += val;
  }

  if (sum === 0n) {
    // Degenerate (should not happen): uniform.
    const each = Number(denomBig / BigInt(n));
    const probs = new Array<number>(n).fill(each);
    probs[0]! += Number(denomBig) - each * n;
    return { probs, denom: Number(denomBig) };
  }

  // Normalize to sum exactly 2^precBits with deterministic remainder handling.
  const probs = new Array<number>(n);
  let allocated = 0n;
  // largest-remainder: compute floor shares, track remainders.
  const rema: { idx: number; rem: bigint }[] = [];
  for (let i = 0; i < n; i++) {
    const scaled = e[i]! * denomBig; // may be large; BigInt handles it
    const q = scaled / sum;
    const r = scaled - q * sum;
    probs[i] = Number(q);
    allocated += q;
    rema.push({ idx: i, rem: r });
  }
  let leftover = Number(denomBig - allocated);
  // Distribute leftover to largest remainders (tie-break: lower index) for determinism.
  rema.sort((a, b) => (a.rem === b.rem ? a.idx - b.idx : a.rem > b.rem ? -1 : 1));
  for (let k = 0; k < rema.length && leftover > 0; k++) {
    probs[rema[k]!.idx]! += 1;
    leftover--;
  }
  return { probs, denom: Number(denomBig) };
}

/** Shannon entropy (bits) of an integer probability distribution. Reporting only. */
export function entropyBits(probs: number[], denom: number): number {
  let h = 0;
  for (const p of probs) {
    if (p <= 0) continue;
    const pr = p / denom;
    h -= pr * Math.log2(pr);
  }
  return h;
}
