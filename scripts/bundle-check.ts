import { build } from "esbuild";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const r = await build({ entryPoints:[resolve(root,"src/browser/entry.ts")], bundle:true, format:"iife", globalName:"NS", target:["es2020"], platform:"browser", write:false, loader:{".json":"json"} });
const code = r.outputFiles[0]!.text;
// eval the IIFE in this global scope so it sets globalThis.ChaffCore
(0, eval)(code);
const Core = (globalThis as unknown as Record<string, any>).ChaffCore;
if (!Core) throw new Error("bundle did not set globalThis.ChaffCore");
console.log("ChaffCore keys:", Object.keys(Core).join(", "));
const mock = new Core.MockLogitSource(256);
let pass = 0; const N = 12;
for (let i=0;i<N;i++){
  const len = 1 + ((i*41+7)%120);
  const secret = new Uint8Array(len); for(let j=0;j<len;j++) secret[j]=(j*31+i*7)&0xff;
  const enc = await Core.encodeSecret(mock, "pw"+i, secret, Core.pins.codec.bucketWidth);
  const dec = await Core.decodeSecret(mock, "pw"+i, enc.cover, Core.pins.codec.bucketWidth);
  const ok = dec.secret.length===len && [...dec.secret].every((b,k)=>b===secret[k]);
  if (ok) pass++; else console.log("  mismatch trial", i);
}
console.log(`offline codec self-test (bundled): ${pass}/${N} ${pass===N?"PASS":"FAIL"}`);
process.exit(pass===N?0:1);
