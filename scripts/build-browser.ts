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
const placeholder = "/*__CHAFF_CORE_BUNDLE__*/";

const targets: [string, string][] = [
  ["web/index.template.html", "index.html"], // the usable single-file app (GitHub Pages landing)
  ["crossarch/template.html", "crossarch/index.html"], // the cross-arch determinism tester (G2)
];

for (const [tpl, out] of targets) {
  const template = readFileSync(resolve(root, tpl), "utf8");
  if (!template.includes(placeholder)) throw new Error(`${tpl} missing core-bundle placeholder`);
  const html = template.replace(placeholder, bundle);
  const outPath = resolve(root, out);
  writeFileSync(outPath, html, "utf8");
  console.log(`built ${out} (${(html.length / 1024).toFixed(1)} KB)`);
}
console.log(`core bundle ${(bundle.length / 1024).toFixed(1)} KB`);
