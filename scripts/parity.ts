/**
 * Bundle parity test. Proves the browser core's async codec loops
 * (asyncEncode/asyncDecode in src/browser/entry.ts) are byte-identical to the
 * sync gate path (codec/coder.ts) when driven by the same reference LM. This is
 * the anti-stub guarantee for Chunk B: the cross-arch tester runs the same
 * decision core that passes G1, not a divergent re-implementation.
 */
import { loadPins } from "../src/pins.js";
import { referenceLMFromPins } from "../src/inference/factory.js";
import { coderConfigFromPins } from "../src/pipeline.js";
import { encodeEntropy as syncEncode, decodeEntropy as syncDecode } from "../src/codec/entropy.js";
import { ChaffCore } from "../src/browser/entry.js";

function arrEq(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function bytesEq(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const pins = loadPins();
const lm = referenceLMFromPins(pins);
const cfg = coderConfigFromPins(pins);

let ok = true;

// Fixed payload → sync encode vs async encode must be identical cover.
const payload = new Uint8Array(40);
for (let i = 0; i < payload.length; i++) payload[i] = (i * 53 + 7) & 0xff;
const bitLen = payload.length * 8;

const sync = syncEncode(lm, payload, bitLen, cfg);
const asyncOut = await ChaffCore.asyncEncode(lm, payload, bitLen, cfg);
const coverMatch = arrEq(sync.cover, asyncOut.cover);
console.log(`${coverMatch ? "PASS" : "FAIL"}  async encode cover == sync encode cover (${sync.cover.length} tokens)`);
ok &&= coverMatch;

// Decode both ways → identical recovered bits.
const syncDec = syncDecode(lm, sync.cover, cfg, undefined);
const asyncDec = await ChaffCore.asyncDecode(lm, sync.cover, cfg);
const bitsMatch = syncDec.bitLength === asyncDec.bitLength && bytesEq(syncDec.bytes, asyncDec.bytes);
console.log(`${bitsMatch ? "PASS" : "FAIL"}  async decode bits == sync decode bits`);
ok &&= bitsMatch;

// Full secret round-trip through the browser core (async glue + crypto).
const secret = new TextEncoder().encode("cross-arch parity secret 🌱");
const enc = await ChaffCore.encodeSecret(lm, "pw123", secret);
const dec = await ChaffCore.decodeSecret(lm, "pw123", enc.cover);
const rtMatch = bytesEq(secret, dec.secret);
console.log(`${rtMatch ? "PASS" : "FAIL"}  browser-core secret round-trip (${secret.length}B → ${enc.cover.length} tokens)`);
ok &&= rtMatch;

console.log(ok ? "\nPARITY GREEN — browser core == gate core" : "\nPARITY RED");
process.exit(ok ? 0 : 1);
