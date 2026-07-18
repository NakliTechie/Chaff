/**
 * State file — the spine for resume/notify. Every `npm run verify` run writes
 * it: what was tried, what passed, the chosen pins, the swept knee, and, on any
 * halt, the tried-trail (which determinism strategy failed, at which stage, at
 * which token index).
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface GateStatus {
  id: string;
  type: "BLOCKER" | "REPORT";
  green: boolean;
  summary: string;
}

export interface TriedTrail {
  strategy: string;
  stage: string;
  tokenIndex: number;
  detail: string;
}

export interface ChaffState {
  schema: "chaff/state@1";
  timestamp: string;
  verdict: "GREEN" | "RED";
  environment: {
    note: string;
    localOracle: string;
    crossArchBackend: string;
  };
  pins: {
    modelId: string;
    weightHash: string;
    bucketWidth: number;
    topK: number;
    logitScale: number;
    softmaxPrecBits: number;
    kdfIterations: number;
  };
  gates: GateStatus[];
  capacity: {
    measuredBitsPerToken: number;
    kneeBucketWidth: number;
    kneeBitsPerToken: number;
    targetEpsilon: number;
  };
  tried: string[];
  triedTrail: TriedTrail[];
  nextHumanStep: string;
}

export function writeState(state: ChaffState): string {
  const dir = resolve(__dirname, "..", "state");
  mkdirSync(dir, { recursive: true });
  const path = resolve(dir, "m0.json");
  writeFileSync(path, JSON.stringify(state, null, 2) + "\n", "utf8");
  return path;
}
