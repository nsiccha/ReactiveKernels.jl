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
// license line of its header, the license files copied to `licenses/`, a
// check that the global works, and optionally `embedLicense`, a license file
// whose text the header carries in full (a license that must travel with every
// copy, as MIT's permission notice must).
const stdlib = (name, license, ...args) => ({
  file: `${name}.js`,
  package: `@stdlib/math-base-special-${name}`,
  url: "https://github.com/stdlib-js/stdlib",
  entry: `module.exports = require("@stdlib/math-base-special-${name}/lib/main.js");`,
  global: `_rk_stdlib_${name}`,
  license,
  licenseFiles: ["LICENSE", "NOTICE"],
  check: (f) => typeof f === "function" && Number.isFinite(f(...args)),
});
const APACHE = "the Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0";
// Boost-derived code is also under the Boost Software License 1.0.
const APACHE_BOOST = APACHE + ", and the Boost Software License 1.0, https://www.boost.org/LICENSE_1_0.txt";
// `rk_ode_rk45_tol`'s Dormand–Prince 5(4) integrator (src/js_ode.jl).
const dopri = {
  file: "dopri.js",
  package: "@reside-ic/dopri",
  url: "https://github.com/mrc-ide/dopri-js",
  entry: 'export { Dopri } from "@reside-ic/dopri";',
  global: "_rk_dopri",
  license: "the MIT License, below",
  licenseFiles: ["LICENCE"],
  embedLicense: "LICENCE",
  check: (m) => {
    const d = new m.Dopri((t, y, dy) => { dy[0] = -y[0]; }, 1, {});
    d.initialise(0, [1]);
    return Math.abs(d.run(1)([1])[0][0] - Math.exp(-1)) < 1e-5;
  },
};
const BUNDLES = [
  stdlib("erf", APACHE, 0.5), stdlib("erfc", APACHE, 0.5), stdlib("erfcx", APACHE, 0.5),
  stdlib("gamma", APACHE, 0.5), stdlib("gammaln", APACHE, 0.5),
  stdlib("betainc", APACHE_BOOST, 0.3, 4, 0.5), stdlib("betaincinv", APACHE_BOOST, 0.05, 4, 0.5),
  dopri,
];

// stdlib-js detects typed arrays, byte order and property support through
// polyfill modules, and logs through Node's `debug`. Every JavaScript engine
// the modules run in has the built-in objects, so these read those instead,
// and the logger is silent.
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
  // Node's `debug` logger, which the inverse functions call: silent.
  debug: shim("debug.cjs"),
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
    ...(b.embedLicense ? ["/*\n" + readFileSync(join(pkgDir, b.embedLicense), "utf8").trim().replaceAll("*/", "* /") + "\n*/"] : []),
  ].join("\n");
  // A self-check: the bundle defines the global it names, and it works.
  const f = new Function(`${code}\nreturn ${b.global};`)();
  if (!b.check(f)) throw new Error(`${b.file}: bundle check failed`);
  writeFileSync(join(here, b.file), header + "\n" + code + "\n");
  mkdirSync(join(here, "licenses", b.package), { recursive: true });
  for (const file of b.licenseFiles) copyFileSync(join(pkgDir, file), join(here, "licenses", b.package, file));
  console.log(`${b.file}: ${b.package} ${version}, ${code.length} bytes of code`);
}
