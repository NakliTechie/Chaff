/**
 * G1 round-trip runner. 100/100 random secrets (1–500 B) must round-trip
 * bit-exact on the same machine: secret → cover tokens → secret. Any mismatch,
 * any thrown decode, or any wrong-length recovery fails the gate.
 *
 * A deterministic RNG seeds the secrets so the run is reproducible and the state
 * file can record exactly what was tested.
 */
import type { LogitSource } from "../inference/types.js";
import type { Pins } from "../pins.js";
import { decodeSecret, encodeSecret } from "../pipeline.js";
import { localize } from "./localizer.js";
import { SplitMix64 } from "../util/prng.js";

export interface RoundTripCase {
  index: number;
  secretLen: number;
  coverTokens: number;
  ok: boolean;
  detail?: string;
}

export interface RoundTripResult {
  ok: boolean;
  passed: number;
  total: number;
  cases: RoundTripCase[];
  totalSecretBytes: number;
  totalCoverTokens: number;
  firstFailure?: RoundTripCase;
}

function makeSecret(rng: SplitMix64, len: number): Uint8Array {
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = Number(rng.nextU64() & 0xffn);
  return out;
}

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export async function runRoundTrip(
  source: LogitSource,
  pins: Pins,
  opts: { trials: number; seed?: bigint } = { trials: 100 },
): Promise<RoundTripResult> {
  const rng = new SplitMix64(opts.seed ?? 0x1234_5678_9abc_def0n);
  const cases: RoundTripCase[] = [];
  let passed = 0;
  let totalSecretBytes = 0;
  let totalCoverTokens = 0;
  let firstFailure: RoundTripCase | undefined;

  for (let i = 0; i < opts.trials; i++) {
    // Secret length uniform in [1, 500], with the boundaries always covered.
    let len: number;
    if (i === 0) len = 1;
    else if (i === 1) len = 500;
    else len = 1 + Number(rng.nextU64() % 500n);

    const secret = makeSecret(rng, len);
    const password = "pw-" + (rng.nextU64() % 1_000_000n).toString();
    totalSecretBytes += len;

    let c: RoundTripCase;
    try {
      const enc = await encodeSecret(source, password, secret, pins);
      totalCoverTokens += enc.cover.length;
      const dec = await decodeSecret(source, password, enc.cover, pins);
      const ok = bytesEqual(secret, dec.secret);
      c = { index: i, secretLen: len, coverTokens: enc.cover.length, ok };
      if (!ok) {
        // Localize the first diverging step for the state file's tried-trail.
        const div = localize(enc.trace, dec.trace);
        c.detail = `mismatch: recovered ${dec.secret.length}B vs ${len}B; ${div.detail}`;
      }
    } catch (e) {
      c = { index: i, secretLen: len, coverTokens: 0, ok: false, detail: `threw: ${(e as Error).message}` };
    }

    if (c.ok) passed++;
    else if (!firstFailure) firstFailure = c;
    cases.push(c);
  }

  return {
    ok: passed === opts.trials,
    passed,
    total: opts.trials,
    cases,
    totalSecretBytes,
    totalCoverTokens,
    firstFailure,
  };
}
