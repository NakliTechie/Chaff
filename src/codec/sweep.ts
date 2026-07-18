/**
 * Bucket-width / precision sweep (G4 REPORT).
 *
 * Determinism mandate #3: bucket width is the robustness/capacity dial. This
 * tool sweeps it and records the "knee" — where widening the bucket stops
 * buying robustness. For each width it measures:
 *   - capacity: effective bits/token on a representative payload
 *   - robustness: round-trip survival when the DECODER sees a noise-perturbed
 *     logit source (a local stand-in for the real cross-arch divergence of G2)
 *
 * The knee is auto-selected as the smallest bucket width whose survival is 100%
 * at the target noise level; its capacity is the reported operating point.
 */
import { loadPins } from "../pins.js";
import { referenceLMFromPins } from "../inference/factory.js";
import { SplitMix64 } from "../util/prng.js";
import { encode, planStep, type CoderConfig } from "./coder.js";
import type { LogitSource } from "../inference/types.js";

export interface SweepPoint {
  bucketWidth: number;
  bitsPerToken: number;
  coverTokens: number;
  /** Per-token decision agreement under simulated cross-arch noise. */
  perTokenByEpsilon: { epsilon: number; perToken: number }[];
}

export interface SweepReport {
  points: SweepPoint[];
  kneeBucketWidth: number;
  kneeBitsPerToken: number;
  targetEpsilon: number;
  perTokenTarget: number;
  logitScale: number;
}

/** Deterministic pseudo-random payload of `bits` bits. */
function randomPayload(bits: number, seed = 0x5eed1234): { bytes: Uint8Array; bitLength: number } {
  const nbytes = Math.ceil(bits / 8);
  const bytes = new Uint8Array(nbytes);
  let s = seed >>> 0;
  for (let i = 0; i < nbytes; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    bytes[i] = (s >>> 16) & 0xff;
  }
  return { bytes, bitLength: bits };
}

function baseCfg(logitScale: number, maxBits: number, precBits: number, bucketWidth: number): CoderConfig {
  return { logitScale, bucketWidth, maxBitsPerStep: maxBits, softmaxPrecBits: precBits, maxSteps: 1_000_000 };
}

/** Additive uniform noise in [-epsilon, epsilon], seeded per step. Simulates
 * the small cross-arch logit divergence G2 measures for real. */
function addNoise(clean: Float64Array, epsilon: number, seed: bigint): Float64Array {
  const rng = new SplitMix64(seed & ((1n << 64n) - 1n));
  const out = new Float64Array(clean.length);
  for (let i = 0; i < clean.length; i++) out[i] = clean[i]! + (rng.nextFloat() * 2 - 1) * epsilon;
  return out;
}

export function runSweep(
  source: LogitSource,
  opts: {
    logitScale: number;
    maxBitsPerStep: number;
    softmaxPrecBits: number;
    widths: number[];
    epsilons: number[];
    targetEpsilon: number;
    perTokenTarget: number;
    payloadBits: number;
    trials: number;
  },
): SweepReport {
  const points: SweepPoint[] = [];
  for (const W of opts.widths) {
    const cfg = baseCfg(opts.logitScale, opts.maxBitsPerStep, opts.softmaxPrecBits, W);
    const payload = randomPayload(opts.payloadBits);
    let enc;
    try {
      enc = encode(source, payload.bytes, payload.bitLength, cfg);
    } catch {
      points.push({
        bucketWidth: W,
        bitsPerToken: 0,
        coverTokens: Infinity,
        perTokenByEpsilon: opts.epsilons.map((e) => ({ epsilon: e, perToken: 0 })),
      });
      continue;
    }
    const bitsPerToken = payload.bitLength / enc.cover.length;

    // Per-token robustness: walk the true cover, compute the clean decision once
    // per step, then perturb in place for each (epsilon, trial) and check whether
    // the decision (k AND the encoded token's rank) is unchanged.
    const agree = new Map<number, { hit: number; total: number }>();
    for (const e of opts.epsilons) agree.set(e, { hit: 0, total: 0 });

    const prefix: number[] = [];
    for (let step = 0; step < enc.cover.length; step++) {
      const clean = source.logits(prefix);
      const cleanPlan = planStep(clean, cfg);
      const observed = enc.cover[step]!;
      const jClean = cleanPlan.ranking.indexOf(observed);
      for (const epsilon of opts.epsilons) {
        const acc = agree.get(epsilon)!;
        for (let t = 0; t < opts.trials; t++) {
          const seed = BigInt(0xc0ffee + step * 2654435761 + t * 40503);
          const noisy = addNoise(clean, epsilon, seed);
          const noisyPlan = planStep(noisy, cfg);
          const jNoisy = noisyPlan.ranking.indexOf(observed);
          const ok = noisyPlan.k === cleanPlan.k && jNoisy === jClean;
          if (ok) acc.hit++;
          acc.total++;
        }
      }
      prefix.push(observed);
    }

    const perTokenByEpsilon = opts.epsilons.map((epsilon) => {
      const a = agree.get(epsilon)!;
      return { epsilon, perToken: a.total > 0 ? a.hit / a.total : 1 };
    });
    points.push({ bucketWidth: W, bitsPerToken, coverTokens: enc.cover.length, perTokenByEpsilon });
  }

  // Knee: smallest width whose per-token agreement at targetEpsilon meets target.
  let kneeBucketWidth = points[points.length - 1]?.bucketWidth ?? 0;
  let kneeBitsPerToken = points[points.length - 1]?.bitsPerToken ?? 0;
  for (const p of points) {
    const s = p.perTokenByEpsilon.find((x) => x.epsilon === opts.targetEpsilon);
    if (s && s.perToken >= opts.perTokenTarget && p.bitsPerToken > 0) {
      kneeBucketWidth = p.bucketWidth;
      kneeBitsPerToken = p.bitsPerToken;
      break;
    }
  }

  return {
    points,
    kneeBucketWidth,
    kneeBitsPerToken,
    targetEpsilon: opts.targetEpsilon,
    perTokenTarget: opts.perTokenTarget,
    logitScale: opts.logitScale,
  };
}

export const SWEEP_DEFAULTS = {
  widths: [128, 192, 256, 384, 512, 768, 1024],
  epsilons: [0.002, 0.008, 0.03, 0.1],
  // Target: a plausible small cross-arch fp32 logit divergence (~0.008 logits)
  // with high per-token decision agreement. The knee is the smallest width that
  // clears it; below it robustness falls off, above it capacity keeps dropping
  // for little robustness gain.
  targetEpsilon: 0.008,
  perTokenTarget: 0.98,
  payloadBits: 256,
  trials: 6,
};

export function sweepFromPins(source: LogitSource, pins: ReturnType<typeof loadPins>): SweepReport {
  return runSweep(source, {
    logitScale: pins.codec.logitScale,
    maxBitsPerStep: pins.codec.maxBitsPerStep,
    softmaxPrecBits: pins.codec.softmaxPrecBits,
    ...SWEEP_DEFAULTS,
  });
}

export function formatSweep(report: SweepReport): string {
  const lines: string[] = [];
  const epsHdr = report.points[0]?.perTokenByEpsilon.map((s) => s.epsilon.toString()).join(" ") ?? "";
  lines.push(`Bucket-width sweep (logitScale=${report.logitScale}, target=${report.perTokenTarget} per-token @ eps=${report.targetEpsilon} logits)`);
  lines.push("");
  lines.push(`  W     bits/tok  coverTok   per-token agreement @ eps=[${epsHdr}]`);
  for (const p of report.points) {
    const surv = p.perTokenByEpsilon.map((s) => s.perToken.toFixed(3)).join("  ");
    const cover = p.coverTokens === Infinity ? "  n/a" : String(p.coverTokens).padStart(8);
    lines.push(`  ${String(p.bucketWidth).padStart(4)}  ${p.bitsPerToken.toFixed(3).padStart(7)}  ${cover}   ${surv}`);
  }
  lines.push("");
  lines.push(
    `KNEE: bucketWidth=${report.kneeBucketWidth}  capacity=${report.kneeBitsPerToken.toFixed(3)} bits/token ` +
      `(>=${report.perTokenTarget} per-token agreement @ eps=${report.targetEpsilon})`,
  );
  return lines.join("\n");
}

// Standalone: `npm run sweep`
if (import.meta.url === `file://${process.argv[1]}`) {
  const pins = loadPins();
  const lm = referenceLMFromPins(pins);
  const report = sweepFromPins(lm, pins);
  console.log("\n" + formatSweep(report) + "\n");
}
