/**
 * Builds the self-contained cross-arch tester (Chunk B): bundles the shared
 * browser core (src/browser/entry.ts) into an IIFE with esbuild and inlines it
 * into crossarch/template.html, producing crossarch/index.html — one file the
 * human opens in a real browser to run G2.
 */
import { build } from "esbuild";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");

const result = await build({
  entryPoints: [resolve(root, "src/browser/entry.ts")],
  bundle: true,
  format: "iife",
  globalName: "ChaffCoreNS",
  target: ["es2020"],
  platform: "browser",
  write: false,
  legalComments: "none",
  loader: { ".json": "json" },
});

const bundle = result.outputFiles[0]!.text;
const template = readFileSync(resolve(root, "crossarch/template.html"), "utf8");
const placeholder = "/*__CHAFF_CORE_BUNDLE__*/";
if (!template.includes(placeholder)) throw new Error("template missing core-bundle placeholder");
const html = template.replace(placeholder, bundle);
const outPath = resolve(root, "crossarch/index.html");
writeFileSync(outPath, html, "utf8");
console.log(`built ${outPath} (${(html.length / 1024).toFixed(1)} KB, core bundle ${(bundle.length / 1024).toFixed(1)} KB)`);
