/**
 * Divergence localizer. Given two aligned per-step traces (one from the encoder,
 * one from the decoder), find the FIRST step where they diverge and classify the
 * earliest stage that differs, in causal order:
 *
 *   tokenizer → logit → softmax → bucket
 *
 * Classification precedence (a difference at an earlier stage explains all later
 * ones, so we report the earliest):
 *   1. logitsHash differ                     → "logit"   (raw distribution differs)
 *   2. softmaxHash differ (logits matched)   → "softmax" (fixed-point softmax differs)
 *   3. k or rankingHash differ (sm matched)  → "bucket"  (capacity/ranking differs)
 *   4. tokenId/selectedRank differ (rest ok) → "tokenizer" (the carried token itself
 *                                                          was altered — carrier/token
 *                                                          stream corruption)
 *
 * This is the tool G5 proves by fault injection: it must pinpoint the exact
 * token index and the correct stage, not a canned "step 0".
 */
import type { Stage, StepTrace } from "../codec/coder.js";

export interface DivergenceReport {
  diverged: boolean;
  index: number; // -1 when identical
  stage: Stage; // "none" when identical
  detail: string;
}

function classify(a: StepTrace, b: StepTrace): Stage {
  if (a.logitsHash !== b.logitsHash) return "logit";
  if (a.softmaxHash !== b.softmaxHash) return "softmax";
  if (a.k !== b.k || a.rankingHash !== b.rankingHash) return "bucket";
  if (a.tokenId !== b.tokenId || a.selectedRank !== b.selectedRank) return "tokenizer";
  return "none";
}

export function localize(encTrace: StepTrace[], decTrace: StepTrace[]): DivergenceReport {
  const n = Math.min(encTrace.length, decTrace.length);
  for (let i = 0; i < n; i++) {
    const a = encTrace[i]!;
    const b = decTrace[i]!;
    const stage = classify(a, b);
    if (stage !== "none") {
      return {
        diverged: true,
        index: i,
        stage,
        detail: describe(stage, a, b),
      };
    }
  }
  if (encTrace.length !== decTrace.length) {
    return {
      diverged: true,
      index: n,
      stage: "bucket",
      detail: `trace length mismatch: enc=${encTrace.length} dec=${decTrace.length} (capacity/stop desync)`,
    };
  }
  return { diverged: false, index: -1, stage: "none", detail: "traces identical" };
}

function describe(stage: Stage, a: StepTrace, b: StepTrace): string {
  switch (stage) {
    case "logit":
      return `raw logits differ (enc ${a.logitsHash} vs dec ${b.logitsHash}) at token index ${a.index}`;
    case "softmax":
      return `fixed-point softmax differs (enc ${a.softmaxHash} vs dec ${b.softmaxHash}) at token index ${a.index}`;
    case "bucket":
      return `bucketing differs (enc k=${a.k}/${a.rankingHash} vs dec k=${b.k}/${b.rankingHash}) at token index ${a.index}`;
    case "tokenizer":
      return `carried token differs (enc tok=${a.tokenId}#${a.selectedRank} vs dec tok=${b.tokenId}#${b.selectedRank}) at token index ${a.index}`;
    default:
      return "identical";
  }
}
