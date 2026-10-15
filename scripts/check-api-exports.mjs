// Verify that the runtime entry (src/index.mjs) and the hand-written
// declarations (src/index.d.ts) agree on the *value* exports — every name the
// module exports must be declared, and vice versa. Type-only declarations
// (interface/type) are ignored.
//
//   node scripts/check-api-exports.mjs
//
// Runs as the first half of `npm run typecheck` so a name added to one side
// cannot silently go missing from the other.
import { readFile } from "node:fs/promises";

const dts = await readFile(new URL("../src/index.d.ts", import.meta.url), "utf8");
const module = await import(new URL("../src/index.mjs", import.meta.url));

const declared = new Set();
const re = /^export (?:declare )?(?:function|const|let|var|class|enum)\s+([A-Za-z0-9_$]+)/gm;
for (const match of dts.matchAll(re)) declared.add(match[1]);

const runtime = new Set(Object.keys(module).filter((k) => k !== "default"));

const missingInTypes = [...runtime].filter((k) => !declared.has(k)).sort();
const missingInRuntime = [...declared].filter((k) => !runtime.has(k)).sort();

if (missingInTypes.length || missingInRuntime.length) {
  if (missingInTypes.length) {
    console.error(`src/index.d.ts is missing declarations for: ${missingInTypes.join(", ")}`);
  }
  if (missingInRuntime.length) {
    console.error(`src/index.mjs is missing exports for: ${missingInRuntime.join(", ")}`);
  }
  process.exit(1);
}
console.log(`api exports in sync (${runtime.size} values)`);
