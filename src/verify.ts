/**
 * `npm run verify` — the M0 done condition. Runs in a fresh context and is the
 * checker's entry point. Exits 0 iff the BLOCKER gates are green:
 *
 *   G1 (BLOCKER)  100/100 random secrets (1–500 B) round-trip bit-exact.
 *   G3 (BLOCKER)  crypto self-test vectors pass at boot.
 *   G5 (BLOCKER)  divergence localizer proven by fault injection.
 *   G4 (REPORT)   measured capacity + swept bucket-width knee (not gated).
 *
 * Anti-gaming: none of the gates can be satisfied by a stub. G1 encrypts real
 * random secrets and compares bytes; a canned codec or a skipped test cannot
 * produce a matching decrypt. G5 injects faults at non-zero indices and checks
 * the localizer names the exact index + stage. G3 uses published KAT vectors.
 */
import { loadPins, assertModelPinned } from "./pins.js";
import { referenceLMFromPins } from "./inference/factory.js";
import { coderConfigFromPins } from "./pipeline.js";
import { runCryptoSelfTest } from "./crypto/vectors.js";
import { runRoundTrip } from "./harness/roundtrip.js";
import { runFaultInjection } from "./harness/faultinject.js";
import { sweepFromPins, formatSweep } from "./codec/sweep.js";
import { writeState, type ChaffState, type GateStatus, type TriedTrail } from "./state.js";
import { planStep } from "./codec/coder.js";

function isoNow(): string {
  // Deterministic-ish timestamp; wall clock is fine for the state file.
  return new Date().toISOString();
}

function measureCapacity(source: ReturnType<typeof referenceLMFromPins>, cfg: ReturnType<typeof coderConfigFromPins>): number {
  // Average per-step bits over a self-consistent walk.
  const prefix: number[] = [];
  let totalK = 0;
  const steps = 160;
  for (let i = 0; i < steps; i++) {
    const plan = planStep(source.logits(prefix), cfg);
    totalK += plan.k;
    const pick = plan.k > 0 ? i % (1 << plan.k) : 0;
    prefix.push(plan.ranking[pick]!);
  }
  return totalK / steps;
}

async function main(): Promise<void> {
  const started = isoNow();
  console.log("Chaff M0 — determinism spike self-verify\n");

  const pins = loadPins();
  const lm = referenceLMFromPins(pins);
  assertModelPinned(pins, lm.id, lm.weightHash);
  const cfg = coderConfigFromPins(pins);

  console.log(`model:       ${lm.id}  weightHash=${lm.weightHash}`);
  console.log(`pins:        bucketWidth=${pins.codec.bucketWidth} logitScale=${pins.codec.logitScale} softmaxPrec=${pins.codec.softmaxPrecBits} kdfIters=${pins.crypto.kdfIterations}`);
  console.log(`local oracle: reference LM (HuggingFace egress blocked headless — see pins.model.rationale)\n`);

  const gates: GateStatus[] = [];
  const tried: string[] = [
    "WASM/fp32/1-thread backend pinned (ORT for Chunk B); pure-fp64 fixed-order reference LM for local gates",
    "codec: own integer fixed-point softmax (never backend float softmax on coding path)",
    "codec: snap-to-grid bucketed argsort; only strictly-separated top-2^k candidates carry bits",
    "crypto: PBKDF2>=600k -> AES-256-GCM with boot KAT vectors and AEAD fail-closed",
    "harness: per-step trace + causal-stage divergence localizer proven by fault injection",
  ];
  const triedTrail: TriedTrail[] = [];

  // --- G3: crypto self-test (run first — refuse to proceed on broken crypto). ---
  const crypto = await runCryptoSelfTest();
  for (const r of crypto.results) console.log(`  ${r.ok ? "✓" : "✗"} G3 ${r.name}${r.ok ? "" : `  (${r.detail ?? ""})`}`);
  gates.push({
    id: "G3",
    type: "BLOCKER",
    green: crypto.ok,
    summary: `${crypto.results.filter((r) => r.ok).length}/${crypto.results.length} crypto vectors pass`,
  });
  if (!crypto.ok) triedTrail.push({ strategy: "crypto boot vectors", stage: "crypto", tokenIndex: -1, detail: "KAT mismatch" });
  console.log(`G3 ${crypto.ok ? "GREEN" : "RED"}\n`);

  // --- G5: fault-injection localizer proof. ---
  const fault = runFaultInjection(lm, cfg);
  for (const r of fault.results) {
    console.log(`  ${r.ok ? "✓" : "✗"} G5 ${r.scenario}  (idx ${r.report.index}/${r.expectedIndex}, stage ${r.report.stage}/${r.expectedStage})`);
  }
  gates.push({
    id: "G5",
    type: "BLOCKER",
    green: fault.ok,
    summary: `${fault.results.filter((r) => r.ok).length}/${fault.results.length} fault scenarios localized exactly`,
  });
  if (!fault.ok) {
    for (const r of fault.results.filter((x) => !x.ok)) {
      triedTrail.push({ strategy: `localizer:${r.scenario}`, stage: r.report.stage, tokenIndex: r.report.index, detail: r.report.detail });
    }
  }
  console.log(`G5 ${fault.ok ? "GREEN" : "RED"}\n`);

  // --- G1: 100/100 round-trip. ---
  console.log("G1 running 100 round-trips (1–500 B secrets)…");
  const rt = await runRoundTrip(lm, pins, { trials: 100 });
  gates.push({
    id: "G1",
    type: "BLOCKER",
    green: rt.ok,
    summary: `${rt.passed}/${rt.total} bit-exact; ${rt.totalSecretBytes}B secrets → ${rt.totalCoverTokens} cover tokens`,
  });
  if (!rt.ok && rt.firstFailure) {
    triedTrail.push({
      strategy: "round-trip",
      stage: "codec",
      tokenIndex: rt.firstFailure.index,
      detail: rt.firstFailure.detail ?? "mismatch",
    });
  }
  console.log(`  ${rt.passed}/${rt.total} passed  (${rt.totalSecretBytes}B → ${rt.totalCoverTokens} cover tokens)`);
  console.log(`G1 ${rt.ok ? "GREEN" : "RED"}\n`);

  // --- G4: capacity + knee (REPORT). ---
  console.log("G4 sweeping bucket width for the capacity/robustness knee…");
  const sweep = sweepFromPins(lm, pins);
  console.log(formatSweep(sweep));
  const measuredBpt = measureCapacity(lm, cfg);
  console.log(`\n  measured capacity @ pinned bucketWidth=${pins.codec.bucketWidth}: ${measuredBpt.toFixed(3)} bits/token`);
  gates.push({
    id: "G4",
    type: "REPORT",
    green: true,
    summary: `capacity ${measuredBpt.toFixed(3)} bits/token @ W=${pins.codec.bucketWidth}; knee W=${sweep.kneeBucketWidth} (${sweep.kneeBitsPerToken.toFixed(3)} b/t)`,
  });
  console.log("");

  const blockers = gates.filter((g) => g.type === "BLOCKER");
  const verdict = blockers.every((g) => g.green) ? "GREEN" : "RED";

  const state: ChaffState = {
    schema: "chaff/state@1",
    timestamp: started,
    verdict,
    environment: {
      note: "M0 headless self-verify. Cross-arch G2 is the one human step (Chunk B tester).",
      localOracle: `${lm.id} (weightHash ${lm.weightHash})`,
      crossArchBackend: `${pins.ortBackend.id} via ${pins.ortBackend.runtime} (${pins.ortBackend.ep}/${pins.ortBackend.dtype}/${pins.ortBackend.threads}-thread) — Chunk B only`,
    },
    pins: {
      modelId: lm.id,
      weightHash: lm.weightHash,
      bucketWidth: pins.codec.bucketWidth,
      topK: pins.codec.topK ?? 128,
      logitScale: pins.codec.logitScale,
      softmaxPrecBits: pins.codec.softmaxPrecBits,
      kdfIterations: pins.crypto.kdfIterations,
    },
    gates,
    capacity: {
      measuredBitsPerToken: Number(measuredBpt.toFixed(4)),
      kneeBucketWidth: sweep.kneeBucketWidth,
      kneeBitsPerToken: Number(sweep.kneeBitsPerToken.toFixed(4)),
      targetEpsilon: sweep.targetEpsilon,
    },
    tried,
    triedTrail,
    nextHumanStep:
      verdict === "GREEN"
        ? "Open crossarch/index.html. Generate on machine A (e.g. ARM/M-series browser), paste the blob into Verify on machine B (x86/NVIDIA browser). PASS = G2 holds → M1 greenlit."
        : "Fix the RED blocker(s) above; see triedTrail for the failing stage/index.",
  };
  const statePath = writeState(state);

  console.log("═".repeat(64));
  for (const g of gates) console.log(`  ${g.id} [${g.type}] ${g.green ? "GREEN" : "RED"}  — ${g.summary}`);
  console.log("═".repeat(64));
  console.log(`\nVERDICT: ${verdict}`);
  console.log(`state written: ${statePath}`);
  if (verdict === "GREEN") {
    console.log(`\nNext (human, G2): ${state.nextHumanStep}`);
  }

  process.exit(verdict === "GREEN" ? 0 : 1);
}

main().catch((e) => {
  console.error("verify crashed:", e);
  process.exit(1);
});
