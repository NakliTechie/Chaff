/**
 * Single place that turns pins into the local logit oracle, so every gate wires
 * the model identically (same seed, same outputScale, same shape → same hash).
 */
import type { Pins } from "../pins.js";
import { makeReferenceLM, ReferenceLM } from "./reference-lm.js";

export function referenceLMFromPins(pins: Pins): ReferenceLM {
  return makeReferenceLM({
    vocab: pins.model.vocab,
    contextTokens: pins.model.contextTokens,
    dModel: pins.model.dModel,
    nLayer: pins.model.nLayer,
    nHead: pins.model.nHead,
    seed: pins.model.seed,
    outputScale: pins.model.outputScale,
    ffMult: pins.model.ffMult,
  });
}
