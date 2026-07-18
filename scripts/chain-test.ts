/**
 * Conversation-chaining capability test (Node). Drives the browser core's
 * encodeMessage/decodeMessage with the reference LM and checks that tampering,
 * deletion, reordering, and replay are detected — the parity capability.
 */
import { loadPins } from "../src/pins.js";
import { referenceLMFromPins } from "../src/inference/factory.js";
import { ChaffCore } from "../src/browser/entry.js";
import type { ConvState } from "../src/conversation.js";

const pins = loadPins();
const lm = referenceLMFromPins(pins);
const W = pins.codec.bucketWidth;
const CONV = "alice+bob";
const PW = "shared-pass";
const enc = (s: string) => new TextEncoder().encode(s);
const dec = (u: Uint8Array) => new TextDecoder().decode(u);

let ok = true;
const chk = (name: string, cond: boolean, extra = "") => { ok &&= cond; console.log(`${cond ? "PASS" : "FAIL"}  ${name}${extra ? "  — " + extra : ""}`); };

// Sender composes an ordered conversation of 4 messages.
const msgs = ["hi", "meet at 9", "bring the file", "done"];
let sState: ConvState | null = null;
const wire: { cover: number[] }[] = [];
for (const m of msgs) {
  const out = await ChaffCore.encodeMessage(lm, PW, enc(m), CONV, sState, W);
  sState = out.newState;
  wire.push({ cover: out.cover });
}

// 1. Happy path: receiver processes in order → all intact.
{
  let rState: ConvState | null = null;
  let allOk = true;
  const recovered: string[] = [];
  for (let i = 0; i < wire.length; i++) {
    const r = await ChaffCore.decodeMessage(lm, PW, wire[i]!.cover, CONV, rState, W);
    rState = r.newState;
    recovered.push(dec(r.secret));
    if (!(r.status.ok && r.status.seq === i + 1)) allOk = false;
  }
  chk("in-order conversation: all messages intact", allOk && recovered.join("|") === msgs.join("|"), recovered.join(" / "));
}

// 2. Deletion: skip message #2 → receiver flags missing.
{
  let rState: ConvState | null = null;
  const r1 = await ChaffCore.decodeMessage(lm, PW, wire[0]!.cover, CONV, rState, W); rState = r1.newState;
  const r3 = await ChaffCore.decodeMessage(lm, PW, wire[2]!.cover, CONV, rState, W); // skipped #2
  chk("deletion detected", !r3.status.ok && r3.status.warnings.includes("missing-message"), r3.status.warnings.join(","));
}

// 3. Reorder / replay: re-feed message #1 after it was already consumed.
{
  let rState: ConvState | null = null;
  const a = await ChaffCore.decodeMessage(lm, PW, wire[0]!.cover, CONV, rState, W); rState = a.newState;
  const b = await ChaffCore.decodeMessage(lm, PW, wire[1]!.cover, CONV, rState, W); rState = b.newState;
  const replay = await ChaffCore.decodeMessage(lm, PW, wire[0]!.cover, CONV, rState, W);
  chk("replay/reorder detected", !replay.status.ok && replay.status.warnings.includes("reorder-or-replay"), replay.status.warnings.join(","));
}

// 4. Tampering: flip a cover token → AES-SIV auth fails (decode throws).
{
  const tampered = wire[0]!.cover.slice();
  tampered[Math.floor(tampered.length / 2)] = (tampered[Math.floor(tampered.length / 2)]! + 1) % lm.vocabSize;
  let threw = false;
  try { await ChaffCore.decodeMessage(lm, PW, tampered, CONV, null, W); } catch { threw = true; }
  chk("tampering rejected (AEAD)", threw);
}

// 5. Wrong conversation id: AAD binding rejects it.
{
  let threw = false;
  try { await ChaffCore.decodeMessage(lm, PW, wire[0]!.cover, "someone-else", null, W); } catch { threw = true; }
  chk("wrong conversation id rejected", threw);
}

console.log(ok ? "\nCHAINING GREEN" : "\nCHAINING RED");
process.exit(ok ? 0 : 1);
