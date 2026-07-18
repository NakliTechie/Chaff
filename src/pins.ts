/**
 * Pins loader — determinism mandate #4. Both encoder and decoder load this file
 * and refuse to run on any mismatch. In M0 (single process) the "both sides"
 * check is structural: every consumer reads pins through here, and the model's
 * live weight hash is checked against expectations at boot.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface Pins {
  schema: string;
  model: {
    id: string;
    kind: string;
    vocab: number;
    contextTokens: number;
    dModel: number;
    nLayer: number;
    nHead: number;
    seed: string;
    outputScale?: number;
    ffMult?: number;
    weightHash: string;
    rationale?: string;
  };
  ortBackend: Record<string, unknown>;
  tokenizer: Record<string, unknown>;
  codec: {
    logitScale: number;
    logitClampFix: number;
    softmaxPrecBits: number;
    expPolyId: string;
    bucketWidth: number;
    maxBitsPerStep: number;
    tieBreak: string;
    topK?: number;
    coder?: string;
    rationale?: string;
  };
  crypto: {
    kdf: string;
    kdfIterations: number;
    keyBits: number;
    aead: string;
    nonceBytes: number;
    saltBytes: number;
    tagBits: number;
    note?: string;
  };
  payload: {
    magic: string;
    lengthFieldBits: number;
    maxSecretBytes: number;
  };
}

let cached: Pins | null = null;

export function loadPins(): Pins {
  if (cached) return cached;
  const path = resolve(__dirname, "..", "pins.json");
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    throw new Error(`pins.json not found at ${path}; refusing to run without pins.`);
  }
  const pins = JSON.parse(raw) as Pins;
  if (pins.schema !== "chaff/pins@1") {
    throw new Error(`pins schema mismatch: got "${pins.schema}", refusing to run.`);
  }
  // Structural sanity — a partial/edited pins file must fail closed.
  const need: [boolean, string][] = [
    [pins.codec.logitScale > 0, "codec.logitScale"],
    [pins.codec.bucketWidth > 0, "codec.bucketWidth"],
    [pins.codec.softmaxPrecBits >= 4, "codec.softmaxPrecBits"],
    [pins.codec.maxBitsPerStep >= 1, "codec.maxBitsPerStep"],
    [pins.crypto.kdfIterations >= 600000, "crypto.kdfIterations>=600000"],
    [pins.crypto.keyBits === 256, "crypto.keyBits===256"],
    [pins.payload.lengthFieldBits >= 16, "payload.lengthFieldBits"],
  ];
  for (const [ok, name] of need) {
    if (!ok) throw new Error(`pins invalid: ${name} failed validation; refusing to run.`);
  }
  cached = pins;
  return pins;
}

/**
 * Assert a live model matches the pinned identity. weightHash "computed-at-boot"
 * means "trust and freeze whatever the seed produces"; a concrete value must match.
 */
export function assertModelPinned(pins: Pins, liveId: string, liveWeightHash: string): void {
  if (pins.model.id !== liveId) {
    throw new Error(`model id mismatch: pinned "${pins.model.id}" vs live "${liveId}"; refusing.`);
  }
  if (pins.model.weightHash !== "computed-at-boot" && pins.model.weightHash !== liveWeightHash) {
    throw new Error(
      `model weightHash mismatch: pinned "${pins.model.weightHash}" vs live "${liveWeightHash}"; refusing.`,
    );
  }
}
