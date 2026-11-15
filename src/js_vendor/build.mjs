// Regenerates the JavaScript that `ReactiveKernels.js_source` copies into an
// emitted module (see README.md). Run from this folder:
//
//     npm ci && node build.mjs
//
// Each bundle becomes one file, one statement defining its global, bundled
// with esbuild. The file's header names the package and version and carries
// the notices of the original implementation it follows.
import * as esbuild from "esbuild";
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));

// A bundle: the file it writes, its npm package (for the version and license
// files), the module source esbuild bundles, the global the file defines, the
// license line of its header, the license files copied to `licenses/`, and a
// check that the global works.
const stdlib = (name) => ({
  file: `${name}.js`,
  package: `@stdlib/math-base-special-${name}`,
  url: "https://github.com/stdlib-js/stdlib",
  entry: `module.exports = require("@stdlib/math-base-special-${name}/lib/main.js");`,
  global: `_rk_stdlib_${name}`,
  license: "the Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0",
  licenseFiles: ["LICENSE", "NOTICE"],
  check: (f) => typeof f === "function" && Number.isFinite(f(0.5)),
});
const BUNDLES = ["erf", "erfc", "erfcx", "gamma", "gammaln"].map(stdlib);

// stdlib-js detects typed arrays, byte order and property support through
// polyfill modules; every JavaScript engine the modules run in has them built
// in, so these read the built-in objects instead.
const shim = (file) => join(here, "shims", file);
const alias = {
  "@stdlib/array-float64": shim("float64array.cjs"),
  "@stdlib/array-uint32": shim("uint32array.cjs"),
  "@stdlib/array-uint16": shim("uint16array.cjs"),
  "@stdlib/array-uint8": shim("uint8array.cjs"),
  "@stdlib/assert-is-little-endian": shim("is-little-endian.cjs"),
  "@stdlib/utils-define-nonenumerable-read-only-property":
    shim("define-nonenumerable-read-only-property.cjs"),
  "@stdlib/utils-define-property": shim("define-property.cjs"),
};

// The original implementations' notices, as the sources quote them in fenced
// ```text blocks of their license comments.
function notices(legal) {
  const found = [];
  for (const m of legal.matchAll(/```text\n([\s\S]*?)```/g)) {
    const text = m[1].split("\n").map((l) => l.replace(/^\s*\*\s?/, "").trimEnd())
      .join("\n").trim();
    const key = text.replace(/\s+/g, " ");
    if (!found.some((t) => t.replace(/\s+/g, " ") === key)) found.push(text);
  }
  return found;
}

const esbuildVersion = JSON.parse(readFileSync(join(here, "node_modules/esbuild/package.json"))).version;
for (const b of BUNDLES) {
  const pkgDir = join(here, "node_modules", b.package);
  const version = JSON.parse(readFileSync(join(pkgDir, "package.json"))).version;
  const result = await esbuild.build({
    stdin: { contents: b.entry, resolveDir: here, loader: "js" },
    bundle: true, format: "iife", globalName: b.global, platform: "neutral", mainFields: ["main"],
    target: "es2020", minify: true, legalComments: "external", alias, write: false,
    outfile: join(here, b.file), logLevel: "warning",
  });
  const code = result.outputFiles.find((f) => f.path.endsWith(".js")).text.trim();
  const legal = result.outputFiles.find((f) => f.path.endsWith(".LEGAL.txt"))?.text ?? "";
  const noticeFile = join(pkgDir, "NOTICE");
  const packageNotice = existsSync(noticeFile) ? " " + readFileSync(noticeFile, "utf8").trim() : "";
  const original = notices(legal);
  const header = [
    `// ${b.package} ${version} (${b.url}), bundled with esbuild ${esbuildVersion}`,
    `// by ReactiveKernels.jl src/js_vendor/build.mjs.${packageNotice}`,
    `// Licensed under ${b.license}`,
    "// (full text in ReactiveKernels.jl src/js_vendor/licenses)." +
      (original.length ? " It follows original code under these notices:" : ""),
    ...original.map((t) => "/*\n" + t.replaceAll("*/", "* /") + "\n*/"),
  ].join("\n");
  // A self-check: the bundle defines the global it names, and it works.
  const f = new Function(`${code}\nreturn ${b.global};`)();
  if (!b.check(f)) throw new Error(`${b.file}: bundle check failed`);
  writeFileSync(join(here, b.file), header + "\n" + code + "\n");
  mkdirSync(join(here, "licenses", b.package), { recursive: true });
  for (const file of b.licenseFiles) copyFileSync(join(pkgDir, file), join(here, "licenses", b.package, file));
  console.log(`${b.file}: ${b.package} ${version}, ${code.length} bytes of code`);
}
