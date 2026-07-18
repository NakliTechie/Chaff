/**
 * Reference LM — a small fixed-weight transformer used as the deterministic
 * logit oracle for the M0 headless gates.
 *
 * Why not GPT-2 here? The HuggingFace egress is blocked in the CI sandbox, so
 * the real Xenova/gpt2 weights cannot be fetched headless. The determinism the
 * gates test lives in the CODEC, not the model, so a self-contained
 * deterministic model is the correct oracle for G1/G5 — it removes model-
 * download flakiness from the determinism proof entirely. The real ORT-Web +
 * GPT-2 backend is exercised by the cross-arch tester (Chunk B) in a browser.
 *
 * The forward pass is plain fp64 with a fixed operation order: no threads, no
 * SIMD reduction reordering, no randomness at inference. Given identical input
 * it returns identical logits run-to-run on the machine running the gates.
 */
import { SplitMix64, seedFromString } from "../util/prng.js";
import { hashInts } from "../util/hash.js";
import type { LogitSource } from "./types.js";

export interface RefLmConfig {
  vocab: number;
  contextTokens: number;
  dModel: number;
  nLayer: number;
  nHead: number;
  seed: string;
  /** Output temperature: sharpens logits so per-step capacity is healthy.
   * Part of the model identity — folded into the weight hash. */
  outputScale?: number;
  /** Feed-forward width multiplier (dff = ffMult * dModel). Default 4. */
  ffMult?: number;
}

const DEFAULT_OUTPUT_SCALE = 3.0;
const DEFAULT_FF_MULT = 4;

interface LayerWeights {
  ln1g: Float64Array;
  ln1b: Float64Array;
  wq: Float64Array;
  wk: Float64Array;
  wv: Float64Array;
  wo: Float64Array;
  ln2g: Float64Array;
  ln2b: Float64Array;
  w1: Float64Array;
  b1: Float64Array;
  w2: Float64Array;
  b2: Float64Array;
}

export class ReferenceLM implements LogitSource {
  readonly vocabSize: number;
  readonly id = "chaff-ref-lm@1";
  readonly weightHash: string;

  private readonly cfg: RefLmConfig;
  private readonly emb: Float64Array; // [vocab][d]
  private readonly pos: Float64Array; // [context][d]
  private readonly layers: LayerWeights[];
  private readonly lnFg: Float64Array;
  private readonly lnFb: Float64Array;
  private readonly wout: Float64Array; // [d][vocab]
  private readonly outputScale: number;
  private readonly ffMult: number;

  // Reusable scratch buffers (sized for the max window) — avoids per-token GC.
  private readonly sH: Float64Array;
  private readonly sNormed: Float64Array;
  private readonly sQ: Float64Array;
  private readonly sK: Float64Array;
  private readonly sV: Float64Array;
  private readonly sAttn: Float64Array;
  private readonly sMid: Float64Array;
  private readonly sScores: Float64Array;
  private readonly sLast: Float64Array;

  constructor(cfg: RefLmConfig) {
    this.cfg = cfg;
    this.outputScale = cfg.outputScale ?? DEFAULT_OUTPUT_SCALE;
    this.ffMult = cfg.ffMult ?? DEFAULT_FF_MULT;
    const D = cfg.dModel;
    const C = cfg.contextTokens;
    this.sH = new Float64Array(C * D);
    this.sNormed = new Float64Array(C * D);
    this.sQ = new Float64Array(C * D);
    this.sK = new Float64Array(C * D);
    this.sV = new Float64Array(C * D);
    this.sAttn = new Float64Array(C * D);
    this.sMid = new Float64Array(this.ffMult * D);
    this.sScores = new Float64Array(C);
    this.sLast = new Float64Array(D);
    this.vocabSize = cfg.vocab;
    const rng = new SplitMix64(seedFromString(cfg.seed));
    const d = cfg.dModel;

    const gauss = (n: number, scale: number): Float64Array => {
      const a = new Float64Array(n);
      for (let i = 0; i < n; i++) a[i] = rng.nextGaussian() * scale;
      return a;
    };
    const ones = (n: number): Float64Array => new Float64Array(n).fill(1);
    const zeros = (n: number): Float64Array => new Float64Array(n);

    // Materialize weights in a FIXED order — this order defines the model identity.
    this.emb = gauss(cfg.vocab * d, 1 / Math.sqrt(d));
    this.pos = gauss(cfg.contextTokens * d, 1 / Math.sqrt(d));

    this.layers = [];
    for (let l = 0; l < cfg.nLayer; l++) {
      this.layers.push({
        ln1g: ones(d),
        ln1b: zeros(d),
        wq: gauss(d * d, 1 / Math.sqrt(d)),
        wk: gauss(d * d, 1 / Math.sqrt(d)),
        wv: gauss(d * d, 1 / Math.sqrt(d)),
        wo: gauss(d * d, 1 / Math.sqrt(d)),
        ln2g: ones(d),
        ln2b: zeros(d),
        w1: gauss(d * this.ffMult * d, 1 / Math.sqrt(d)),
        b1: zeros(this.ffMult * d),
        w2: gauss(this.ffMult * d * d, 1 / Math.sqrt(this.ffMult * d)),
        b2: zeros(d),
      });
    }
    this.lnFg = ones(d);
    this.lnFb = zeros(d);
    this.wout = gauss(d * cfg.vocab, 1 / Math.sqrt(d));

    this.weightHash = this.computeWeightHash();
  }

  private computeWeightHash(): string {
    // Fold a sparse fingerprint of every weight block into one hash. We quantize
    // to int32 (× 1e6) so the hash is stable and cheap.
    const q = (a: Float64Array): number[] => {
      const out: number[] = [];
      // sample every block fully but as quantized ints
      for (let i = 0; i < a.length; i++) out.push(Math.round(a[i]! * 1e6) | 0);
      return out;
    };
    const parts: number[] = [];
    parts.push(Math.round(this.outputScale * 1e6) | 0, this.ffMult | 0);
    parts.push(...q(this.emb), ...q(this.pos), ...q(this.wout), ...q(this.lnFg), ...q(this.lnFb));
    for (const L of this.layers) {
      parts.push(
        ...q(L.wq), ...q(L.wk), ...q(L.wv), ...q(L.wo),
        ...q(L.w1), ...q(L.b1), ...q(L.w2), ...q(L.b2),
        ...q(L.ln1g), ...q(L.ln1b), ...q(L.ln2g), ...q(L.ln2b),
      );
    }
    return hashInts(parts);
  }

  private static layerNorm(x: Float64Array, off: number, d: number, g: Float64Array, b: Float64Array): void {
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[off + i]!;
    mean /= d;
    let variance = 0;
    for (let i = 0; i < d; i++) {
      const v = x[off + i]! - mean;
      variance += v * v;
    }
    variance /= d;
    const inv = 1 / Math.sqrt(variance + 1e-5);
    for (let i = 0; i < d; i++) {
      x[off + i] = (x[off + i]! - mean) * inv * g[i]! + b[i]!;
    }
  }

  private static gelu(v: number): number {
    // tanh approximation (deterministic on a single engine).
    const c = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }

  /**
   * Full forward over the windowed prefix; returns logits for the NEXT token.
   *
   * Uses reusable scratch buffers and the standard "only the last position is
   * needed" optimization: for every layer we compute K/V at all positions, but
   * the query/attention/MLP for non-final layers run at all positions (later
   * layers attend to them), while the FINAL layer only computes the last
   * position's path. Determinism is unaffected — the arithmetic and its order
   * are identical to the naive version for the value we return.
   */
  logits(prefix: number[]): Float64Array {
    const d = this.cfg.dModel;
    const ctx = this.cfg.contextTokens;
    const nHead = this.cfg.nHead;
    const nLayer = this.cfg.nLayer;
    const headDim = d / nHead;
    const dff = this.ffMult * d;

    // Window to the last `ctx` tokens. Empty prefix → single BOS-ish zero token.
    const start = prefix.length === 0 ? 0 : Math.max(0, prefix.length - ctx);
    const T = prefix.length === 0 ? 1 : prefix.length - start;

    const h = this.sH;
    for (let t = 0; t < T; t++) {
      const raw = prefix.length === 0 ? 0 : prefix[start + t]!;
      const tok = ((raw % this.vocabSize) + this.vocabSize) % this.vocabSize;
      for (let i = 0; i < d; i++) h[t * d + i] = this.emb[tok * d + i]! + this.pos[t * d + i]!;
    }

    const scale = 1 / Math.sqrt(headDim);
    const normed = this.sNormed;
    const Q = this.sQ;
    const K = this.sK;
    const V = this.sV;
    const attnOut = this.sAttn;
    const mid = this.sMid;
    const scores = this.sScores;
    const lastT = T - 1;

    for (let li = 0; li < nLayer; li++) {
      const L = this.layers[li]!;
      const isFinal = li === nLayer - 1;
      const qStart = isFinal ? lastT : 0; // which positions need the full post-KV path

      // --- Attention block: layerNorm (all positions), then K/V (all), Q (subset). ---
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < d; i++) normed[t * d + i] = h[t * d + i]!;
        ReferenceLM.layerNorm(normed, t * d, d, L.ln1g, L.ln1b);
      }
      for (let t = 0; t < T; t++) {
        for (let o = 0; o < d; o++) {
          let sk = 0, sv = 0;
          for (let i = 0; i < d; i++) {
            const x = normed[t * d + i]!;
            sk += x * L.wk[i * d + o]!;
            sv += x * L.wv[i * d + o]!;
          }
          K[t * d + o] = sk;
          V[t * d + o] = sv;
        }
      }
      for (let t = qStart; t < T; t++) {
        for (let o = 0; o < d; o++) {
          let sq = 0;
          for (let i = 0; i < d; i++) sq += normed[t * d + i]! * L.wq[i * d + o]!;
          Q[t * d + o] = sq;
        }
      }

      // Causal self-attention per head, only for the needed query positions.
      for (let hd = 0; hd < nHead; hd++) {
        const base = hd * headDim;
        for (let t = qStart; t < T; t++) {
          let maxS = -Infinity;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            for (let i = 0; i < headDim; i++) s += Q[t * d + base + i]! * K[j * d + base + i]!;
            s *= scale;
            scores[j] = s;
            if (s > maxS) maxS = s;
          }
          let sum = 0;
          for (let j = 0; j <= t; j++) {
            const e = Math.exp(scores[j]! - maxS);
            scores[j] = e;
            sum += e;
          }
          for (let i = 0; i < headDim; i++) {
            let acc = 0;
            for (let j = 0; j <= t; j++) acc += (scores[j]! / sum) * V[j * d + base + i]!;
            attnOut[t * d + base + i] = acc;
          }
        }
      }

      // Output projection + residual (needed positions).
      for (let t = qStart; t < T; t++) {
        for (let o = 0; o < d; o++) {
          let s = 0;
          for (let i = 0; i < d; i++) s += attnOut[t * d + i]! * L.wo[i * d + o]!;
          h[t * d + o] = h[t * d + o]! + s;
        }
      }

      // --- MLP block (needed positions). Reuse `normed` for the LN2 output. ---
      for (let t = qStart; t < T; t++) {
        for (let i = 0; i < d; i++) normed[t * d + i] = h[t * d + i]!;
        ReferenceLM.layerNorm(normed, t * d, d, L.ln2g, L.ln2b);
        for (let o = 0; o < dff; o++) {
          let s = L.b1[o]!;
          for (let i = 0; i < d; i++) s += normed[t * d + i]! * L.w1[i * dff + o]!;
          mid[o] = ReferenceLM.gelu(s);
        }
        for (let o = 0; o < d; o++) {
          let s = L.b2[o]!;
          for (let i = 0; i < dff; i++) s += mid[i]! * L.w2[i * d + o]!;
          h[t * d + o] = h[t * d + o]! + s;
        }
      }
    }

    // Final norm on the LAST position only.
    const last = this.sLast;
    for (let i = 0; i < d; i++) last[i] = h[lastT * d + i]!;
    ReferenceLM.layerNorm(last, 0, d, this.lnFg, this.lnFb);

    // Head → logits (fresh array; callers may hold it across the next call).
    const logits = new Float64Array(this.vocabSize);
    for (let v = 0; v < this.vocabSize; v++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += last[i]! * this.wout[i * this.vocabSize + v]!;
      logits[v] = s * this.outputScale;
    }
    return logits;
  }
}

export function makeReferenceLM(cfg: RefLmConfig): ReferenceLM {
  return new ReferenceLM(cfg);
}
