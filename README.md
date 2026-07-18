# Chaff — M0 Determinism Spike

Browser-native linguistic steganography. Encrypt a secret, then use a local
LLM's token distribution to encode the ciphertext as innocent-looking cover text
you paste into any chat app. The recipient — same model, same shared secret —
recovers it. Zero server, zero account, offline after model load.

**M0 is the determinism spike:** headless, no UI. It proves the riskiest
assumption — that an arithmetic-coding stego stream round-trips bit-exact — and
hands the human exactly one step: a cross-architecture round-trip (G2) they run
in a real browser on a second machine.

## Use it (no install, no build)

The usable app is a single self-contained file, [`index.html`](./index.html),
served live via GitHub Pages:

**→ https://naklitechie.github.io/Chaff/**

Open it, click **Load model** (one-time GPT-2 download, cached after), set a
shared password, then **Hide** a secret into innocent cover text or **Reveal**
one you received. Everything runs in your browser — no server, no account,
offline after the model loads. The cross-arch determinism tester is at
[`/crossarch/`](https://naklitechie.github.io/Chaff/crossarch/).

The sections below document the headless determinism spike (the Node gate suite)
that backs the app — it is not needed to use the page.

## Honest threat model — read this first

Chaff produces **plausible cover under casual or manual review.** It reads as
ordinary text to a human or a keyword filter.

- **Do claim:** "reads as ordinary text to a human or a keyword filter."
- **Do *not* claim:** "undetectable" or "defeats detection."

LLM-stego is statistically detectable by an adversary who has the exact model
and is looking for it. Security rests on the adversary lacking (a) the shared
secret and (b) suspicion strong enough to run a matched-model detector. Anyone
who ships the overclaim endangers the user.

## Quick start

```bash
npm install
npm run verify          # the M0 gates: G1 + G3 + G5 green, G4 reported (~2–3 min)
npm run sweep           # G4 bucket-width / capacity knee sweep on its own
npm run build:browser   # build crossarch/index.html (the G2 tester)
npm run parity          # prove the browser core == the Node gate core
npm run check:bundle    # run the built browser bundle's offline codec self-test
```

`npm run verify` exits `0` only when every BLOCKER gate is green and writes
`state/m0.json` (the resume/notify spine).

## The gates

| Gate | Type | Condition | Latest |
|---|---|---|---|
| **G1** | BLOCKER | 100/100 random secrets (1–500 B) round-trip bit-exact, same machine | ✅ 100/100 |
| **G3** | BLOCKER | Crypto self-test vectors pass at boot | ✅ 7/7 |
| **G5** | BLOCKER | Divergence localizer proven by fault injection (exact token index + stage) | ✅ 6/6 |
| **G4** | REPORT | Measured capacity (bits/token) + swept bucket-width knee | ~2.0 bits/tok @ W=512 (knee) |
| **G2** | HUMAN | Cross-arch round-trip (ARM ↔ x86) in a real browser | ← the one human step |

Anti-gaming is built in: G1 encrypts real random secrets and compares bytes (a
canned codec cannot produce a matching decrypt); G5 injects faults at **non-zero**
token indices and checks the localizer names the exact index and stage (a stub
that always blames step 0 fails); G3 uses published KAT vectors.

## Determinism mandates (the crux)

1. **Backend:** ONNX Runtime Web, WASM EP, fp32, single-thread. No WebGPU. (In
   the cross-arch tester; the local gates use a deterministic reference LM — see
   below.)
2. **The codec computes its own fixed-point softmax** — integer math, fixed
   denominator, deterministic rounding (`src/fixedpoint.ts`). The backend's
   float softmax is never on the coding path.
3. **Encode against `argsort(logits)` + snap-to-grid bucketed gaps**
   (`src/codec/coder.ts`). Only the top `2^k` candidates that sit in strictly
   separated buckets carry bits, so per-arch logit noise that doesn't cross a
   bucket boundary can't reorder the coding set. **Bucket width is the
   robustness/capacity dial** — swept, with the knee recorded and pinned.
4. **Everything is pinned** in [`pins.json`](./pins.json) (model + hash,
   tokenizer, opset, ORT-Web version, bucket width, precision, KDF). Both sides
   load-or-refuse (`src/pins.ts`).

## Environmental note (why a reference LM for the local gates)

The M0 CI sandbox blocks HuggingFace egress, so `Xenova/gpt2` cannot be fetched
headless. The determinism the gates test lives in the **codec, crypto, pipeline,
and divergence harness — all model-agnostic** — so the local gates run against a
self-contained **deterministic reference transformer** (`src/inference/
reference-lm.ts`, fixed-weight, pure-fp64, fixed op order). This *removes*
model-download flakiness from the determinism proof.

The **real ORT-Web + GPT-2 backend** is wired into the cross-arch tester
(`crossarch/index.html`), which the human runs in a real browser where HF *is*
reachable — that is exactly what G2 is for. The tester shares the **same** codec
+ crypto bundle that passes the Node gates (`npm run parity` proves it is
byte-identical), so it is not a re-implementation that could drift into a stub.

## G2 — the one human step

1. `npm run build:browser`, then open `crossarch/index.html` in a browser on
   **machine A** (e.g. an ARM / Apple-silicon laptop).
2. Click **Load GPT-2**, then run the **offline codec self-test** and the
   **GPT-2 same-machine self-test** — both must be green.
3. **Generate** a cover blob from a secret + shared password. Copy the blob.
4. On **machine B** (a different arch, e.g. x86 / NVIDIA), open the same file,
   load GPT-2, paste the blob into **Verify**, enter the same password.

| G2 outcome | Next |
|---|---|
| PASS cross-arch | Greenlight **M1** (UI + UX/UI reference gate). |
| PASS same-arch only | Scope narrows to same-runtime pairs — still shippable, different pitch. |
| FAIL same-arch | Tokenizer round-trip or fixed-point codec broken — determinism rework. |

A FAIL localizes to a stage (tokenizer / logit / softmax / bucket) and token
index, so you know *where* the streams diverged.

## Layout

```
pins.json                 frozen determinism parameters (load-or-refuse)
src/
  fixedpoint.ts           integer fixed-point softmax (mandate #2)
  pins.ts                 pins loader + model-identity assertion
  pipeline.ts             secret ⇄ cover (AEAD + frame + codec)
  conversation.ts         message chaining (seq + chain, tamper/drop/reorder detection)
  state.ts                state-file writer
  verify.ts               `npm run verify` — the gate orchestrator
  inference/
    types.ts              LogitSource contract
    reference-lm.ts       deterministic fixed-weight transformer (local oracle)
    factory.ts            build the reference LM from pins
  codec/
    coder.ts              rank-bucket coder + per-step trace (mandate #3)
    sweep.ts              bucket-width / capacity knee sweep (G4)
  crypto/
    aead.ts               PBKDF2≥600k → AES-256-SIV (msgs) / AES-256-GCM (at-rest)
    aes-siv.ts            clean-room AES-SIV (RFC 5297) on WebCrypto primitives
    vectors.ts            boot KAT self-test (G3): PBKDF2 / GCM / CMAC / SIV
  harness/
    roundtrip.ts          100/100 round-trip runner (G1)
    localizer.ts          causal-stage divergence localizer
    faultinject.ts        fault-injection proof of the localizer (G5)
  browser/
    entry.ts              shared browser core (bundled into the tester)
crossarch/
  template.html           cross-arch tester source
  index.html              built, self-contained G2 tester
scripts/                  build-browser, parity, bundle-check
state/m0.json             last run's verdict, pins, knee, tried-trail
```

## Crypto stack

PBKDF2-SHA256 (≥600k) → **AES-256-SIV** (RFC 5297, nonce-misuse resistant) for
messages; **AES-256-GCM** for at-rest conversation state. Fails closed. AES-SIV
is a clean-room implementation on WebCrypto primitives (CMAC + S2V + AES-CTR),
verified against the RFC 4493 and RFC 5297 known-answer vectors at boot (G3).
Argon2id is a future upgrade.

**Conversations (message chaining).** Naming a conversation links its messages
into a single ordered, authenticated chain (`src/conversation.ts`): each message
carries an AEAD-authenticated `[seq, prevChain]` header, so tampering, dropped
messages, reordering, and replays are detected. Chain state is stored per
conversation, encrypted at rest under the password. Leave the conversation blank
for a stateless one-off message.

## Non-goals (M0)

No UI. No carrier robustness (M2). No `window.chaff` API (M3). No Llama (M4). No
WebGPU. No relay, server, account, or telemetry.

## License

AGPL-3.0-or-later at the tool. The steganography technique is reimplemented
clean in TypeScript — not line-ported from any GPL source. The GPL/AGPL boundary
stays at the tool and never touches the Apache-licensed inference primitive.
