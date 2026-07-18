/**
 * Fault injection for G5. Proves the divergence localizer is real — it must
 * pinpoint the EXACT token index and the correct stage of an injected fault, and
 * report "no divergence" on a clean round-trip. A stub localizer that always
 * blames step 0 (or always passes) fails these.
 *
 * Faults exercised:
 *   - logit     : perturb one raw logit at a chosen step → stage "logit"
 *   - tokenizer : corrupt one carried cover token        → stage "tokenizer"
 *   - bucket    : force a different capacity at one step → stage "bucket"
 *   - softmax   : unit-checked via the localizer classifier directly, because a
 *                 deterministic integer softmax cannot diverge from identical
 *                 logits by construction (that invariance is itself a guarantee).
 */
import type { LogitSource } from "../inference/types.js";
import { encode, planStep, type CoderConfig, type StepTrace } from "../codec/coder.js";
import { hashInts } from "../util/hash.js";
import { localize, type DivergenceReport } from "./localizer.js";

/** Wraps a source and perturbs one token's logit at exactly one step. */
export class LogitFaultSource implements LogitSource {
  readonly vocabSize: number;
  readonly id: string;
  readonly weightHash: string;
  constructor(
    private readonly inner: LogitSource,
    private readonly atPrefixLen: number,
    private readonly tokenIndex: number,
    private readonly delta: number,
  ) {
    this.vocabSize = inner.vocabSize;
    this.id = inner.id + "+logitfault";
    this.weightHash = inner.weightHash;
  }
  logits(prefix: number[]): Float64Array {
    const out = this.inner.logits(prefix);
    if (prefix.length === this.atPrefixLen) {
      const copy = Float64Array.from(out);
      copy[this.tokenIndex] = copy[this.tokenIndex]! + this.delta;
      return copy;
    }
    return out;
  }
}

export interface FaultResult {
  scenario: string;
  expectedIndex: number;
  expectedStage: string;
  report: DivergenceReport;
  ok: boolean;
}

/** Build the encoder-side trace once for a fixed payload/cover. */
function encodeTrace(source: LogitSource, cfg: CoderConfig): { cover: number[]; trace: StepTrace[] } {
  // Deterministic payload long enough to have many coding steps.
  const payload = new Uint8Array(24);
  for (let i = 0; i < payload.length; i++) payload[i] = (i * 37 + 11) & 0xff;
  const r = encode(source, payload, payload.length * 8, cfg);
  return { cover: r.cover, trace: r.trace };
}

function hashOf(a: ArrayLike<number>): string {
  return hashInts(a);
}

/**
 * Re-derive the decoder trace by replaying the cover through a (possibly faulted)
 * source. Unlike decode(), this never throws on divergence — it records the
 * observed token's rank even when it falls outside the usable set — so the
 * localizer always receives a full, aligned trace to compare.
 */
function decodeTrace(source: LogitSource, cover: number[], cfg: CoderConfig): StepTrace[] {
  const prefix: number[] = [];
  const trace: StepTrace[] = [];
  for (let step = 0; step < cover.length; step++) {
    const plan = planStep(source.logits(prefix), cfg);
    const observed = cover[step]!;
    const j = plan.ranking.indexOf(observed);
    trace.push({
      index: step,
      prefixLen: prefix.length,
      tokenId: observed,
      logitsHash: hashOf(plan.quant),
      softmaxHash: hashOf(plan.probs),
      rankingHash: hashOf(plan.ranking.slice(0, Math.min(64, plan.ranking.length))),
      k: plan.k,
      selectedRank: j,
      entropyBits: 0,
    });
    prefix.push(observed);
  }
  return trace;
}

export function runFaultInjection(source: LogitSource, cfg: CoderConfig): { ok: boolean; results: FaultResult[] } {
  const results: FaultResult[] = [];
  const { cover, trace: encTrace } = encodeTrace(source, cfg);

  // Choose a non-trivial, non-zero step to defeat a "always step 0" stub.
  const faultStep = Math.min(7, cover.length - 1);

  // --- 1. Clean round-trip: localizer must report NO divergence. ---
  {
    const decT = decodeTrace(source, cover, cfg);
    const report = localize(encTrace, decT);
    results.push({
      scenario: "clean round-trip (no fault)",
      expectedIndex: -1,
      expectedStage: "none",
      report,
      ok: report.diverged === false,
    });
  }

  // --- 2. Logit fault at faultStep: perturb the top token's logit downward a lot. ---
  {
    // Perturb the currently-selected token so the ranking/bucket also shifts,
    // but the EARLIEST differing stage is the raw logit.
    const topToken = encTrace[faultStep]!.tokenId;
    const faulted = new LogitFaultSource(source, faultStep, topToken, -50 * cfg.bucketWidth / 1024 - 5);
    const decT = decodeTrace(faulted, cover, cfg);
    const report = localize(encTrace, decT);
    results.push({
      scenario: `logit fault @ index ${faultStep} (token ${topToken})`,
      expectedIndex: faultStep,
      expectedStage: "logit",
      report,
      ok: report.diverged && report.index === faultStep && report.stage === "logit",
    });
  }

  // --- 3. Logit fault at a DIFFERENT index: proves the index tracks the fault. ---
  {
    const altStep = Math.min(3, cover.length - 1);
    const topToken = encTrace[altStep]!.tokenId;
    const faulted = new LogitFaultSource(source, altStep, topToken, -50 * cfg.bucketWidth / 1024 - 5);
    const decT = decodeTrace(faulted, cover, cfg);
    const report = localize(encTrace, decT);
    results.push({
      scenario: `logit fault @ index ${altStep} (control for step-0 stub)`,
      expectedIndex: altStep,
      expectedStage: "logit",
      report,
      ok: report.diverged && report.index === altStep && report.stage === "logit",
    });
  }

  // --- 4. Tokenizer/carrier fault: corrupt one carried cover token. ---
  {
    const corruptStep = Math.min(5, cover.length - 1);
    const corrupted = cover.slice();
    // Replace with a token guaranteed to be a different, valid vocab id.
    corrupted[corruptStep] = (corrupted[corruptStep]! + 1) % source.vocabSize;
    const decT = decodeTrace(source, corrupted, cfg);
    const report = localize(encTrace, decT);
    results.push({
      scenario: `carrier/token fault @ index ${corruptStep}`,
      expectedIndex: corruptStep,
      expectedStage: "tokenizer",
      report,
      ok: report.diverged && report.index === corruptStep && report.stage === "tokenizer",
    });
  }

  // --- 5. Bucket fault: force a smaller capacity at one step via a per-step width. ---
  {
    const bStep = Math.min(6, cover.length - 1);
    // Decode with a widened bucket at just this step so k differs but logits match.
    const decT = replayTraceWithWidthOverride(source, cover, cfg, bStep, cfg.bucketWidth * 64);
    const report = localize(encTrace, decT);
    results.push({
      scenario: `bucket fault @ index ${bStep} (capacity forced)`,
      expectedIndex: bStep,
      expectedStage: "bucket",
      report,
      ok: report.diverged && report.index === bStep && report.stage === "bucket",
    });
  }

  // --- 6. Softmax-stage classification (unit check of the localizer). ---
  {
    const a = encTrace[Math.min(4, encTrace.length - 1)]!;
    const b: StepTrace = { ...a, softmaxHash: a.softmaxHash + "x" }; // logits match, softmax differs
    const report = localize([a], [b]);
    results.push({
      scenario: "softmax-stage classifier unit check",
      expectedIndex: 0,
      expectedStage: "softmax",
      report,
      ok: report.diverged && report.index === 0 && report.stage === "softmax",
    });
  }

  const ok = results.every((r) => r.ok);
  return { ok, results };
}

/** Replay producing a decoder trace where ONE step uses an overridden bucket width. */
function replayTraceWithWidthOverride(
  source: LogitSource,
  cover: number[],
  cfg: CoderConfig,
  atStep: number,
  overrideWidth: number,
): StepTrace[] {
  const prefix: number[] = [];
  const trace: StepTrace[] = [];
  for (let step = 0; step < cover.length; step++) {
    const stepCfg = step === atStep ? { ...cfg, bucketWidth: overrideWidth } : cfg;
    const plan = planStep(source.logits(prefix), stepCfg);
    const observed = cover[step]!;
    const j = plan.ranking.indexOf(observed);
    trace.push({
      index: step,
      prefixLen: prefix.length,
      tokenId: observed,
      logitsHash: hashOf(plan.quant),
      softmaxHash: hashOf(plan.probs),
      rankingHash: hashOf(plan.ranking.slice(0, Math.min(64, plan.ranking.length))),
      k: plan.k,
      selectedRank: j,
      entropyBits: 0,
    });
    prefix.push(observed);
  }
  return trace;
}
