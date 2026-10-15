/**
 * Nightly parity — run one real example per family and compare the answers to
 * the stored probabilities, so a regression in a graph, a prompt layout, a
 * tokenizer, or the calibration math fails loudly instead of quietly shipping.
 *
 *   node scripts/parity-nightly.mjs                  # compare recorded families
 *   node scripts/parity-nightly.mjs --record         # (re)record baselines
 *   node scripts/parity-nightly.mjs --families=bekko,kev
 *   node scripts/parity-nightly.mjs --tolerance=1e-3
 *
 * Baselines live in parity/baselines.json. A family with no baseline is
 * SKIPPED (record it with --record); a mismatch or an unexpected error exits 1.
 * The run downloads the real pinned checkpoints, so it belongs in a scheduled
 * job (see .github/workflows/nightly.yml), not in unit CI.
 *
 * What is compared: every answer field (including the full probability map),
 * the truncation flag, the fused length, and — when a family reports it — the
 * decoded prompt. Timings are the only part of a DecideResult that is not
 * compared (they are wall-clock and never reproducible).
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createDecisionRuntime, listDecisionFamilies } from "../src/registry.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BASELINE_FILE = resolve(root, "parity", "baselines.json");
const BASELINE_SCHEMA = 1;

/**
 * The example every family must reproduce: one state, three questions — the
 * choice, noul, and score kinds the shared contract promises. Mirror of the
 * model card's banking example, kept small enough for a nightly CI run.
 */
const STATE =
  "I was charged twice for the same order and nobody answers my emails. I want my money back now.";
const QUESTIONS = [
  {
    type: "choice",
    instructions: "Which product area is the message about?",
    options: ["fees & charges", "pin & security", "refund & dispute", "top-up"],
  },
  { type: "noul", instructions: "The customer is asking for a refund." },
  {
    type: "score",
    instructions: "How positive is the sentiment of this message?",
    options: ["very negative", "negative", "neutral", "positive", "very positive"],
  },
];

// Per-family tweaks: runtime options for the headless (Node) run, or an
// example override when a family cannot answer the shared three. Keep the
// state/questions equal to STATE and QUESTIONS wherever possible so one
// recording pass covers every family.
const CASES = {
  // transformers.js' Node build (onnxruntime-node) rejects "wasm", while the
  // browser default is wasm — the headless run asks for cpu explicitly.
  "open-jev": { state: STATE, questions: QUESTIONS, options: { device: "cpu" } },
};

// ---------------------------------------------------------------------------
// Tiny CLI
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const out = { record: false, families: null, tolerance: 1e-4, file: BASELINE_FILE };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const eq = arg.startsWith("--") ? arg.indexOf("=") : -1;
    const name = eq === -1 ? arg : arg.slice(0, eq);
    const inline = eq === -1 ? null : arg.slice(eq + 1);
    const value = () => {
      if (inline != null) return inline;
      if (i + 1 >= argv.length) throw new Error(`${name} needs a value`);
      return argv[++i];
    };
    switch (name) {
      case "--record": out.record = true; break;
      case "--families":
        out.families = value().split(",").map((s) => s.trim()).filter(Boolean);
        break;
      case "--tolerance":
        out.tolerance = Number(value());
        if (!Number.isFinite(out.tolerance) || out.tolerance < 0) {
          throw new Error(`--tolerance must be a non-negative number, got "${inline}"`);
        }
        break;
      case "--file": out.file = resolve(root, value()); break;
      case "--help": case "-h": out.help = true; break;
      default: throw new Error(`unknown option "${arg}" (try --help)`);
    }
  }
  return out;
}

function usage() {
  return [
    "usage: node scripts/parity-nightly.mjs [--record] [--families=a,b] [--tolerance=1e-4] [--file=parity/baselines.json]",
    "",
    "Runs one real example per decision family and compares the answers to",
    "parity/baselines.json. Families without a baseline are skipped; pass",
    "--record to store fresh baselines for the families that run.",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Baseline IO
// ---------------------------------------------------------------------------

async function readBaselines(file) {
  let raw;
  try {
    raw = await readFile(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return { schema: BASELINE_SCHEMA, families: {} };
    throw error;
  }
  let data;
  try {
    data = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${file} is not valid JSON (${error.message})`);
  }
  if (data?.schema !== BASELINE_SCHEMA) {
    throw new Error(
      `${file} has schema ${JSON.stringify(data?.schema)}; this script writes schema ${BASELINE_SCHEMA} — re-record it`);
  }
  return { schema: BASELINE_SCHEMA, families: data.families ?? {} };
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

const fmt = (value) => {
  if (value === undefined) return "undefined";
  const json = JSON.stringify(value);
  return json === undefined ? String(value) : json;
};

/** Collect every difference between `actual` and the stored `expected`. */
function diffJson(actual, expected, tolerance, path, problems) {
  const bothNumbers = typeof actual === "number" && typeof expected === "number";
  if (bothNumbers) {
    if (!(Math.abs(actual - expected) <= tolerance)) {
      problems.push(`${path}: ${fmt(actual)} != stored ${fmt(expected)}`);
    }
    return problems;
  }
  if (actual === null || expected === null || typeof actual !== "object" || typeof expected !== "object") {
    if (actual !== expected) problems.push(`${path}: ${fmt(actual)} != stored ${fmt(expected)}`);
    return problems;
  }
  const actualArray = Array.isArray(actual);
  const expectedArray = Array.isArray(expected);
  if (actualArray !== expectedArray) {
    problems.push(`${path}: shape changed (${actualArray ? "array" : "object"} vs stored ${expectedArray ? "array" : "object"})`);
    return problems;
  }
  if (expectedArray) {
    if (actual.length !== expected.length) {
      problems.push(`${path}: length ${actual.length} != stored ${expected.length}`);
      return problems;
    }
    for (let i = 0; i < expected.length; i++) {
      diffJson(actual[i], expected[i], tolerance, `${path}[${i}]`, problems);
    }
    return problems;
  }
  const keys = [...new Set([...Object.keys(actual), ...Object.keys(expected)])].sort();
  for (const key of keys) {
    if (!(key in expected)) problems.push(`${path}.${key}: unexpected ${fmt(actual[key])}`);
    else if (!(key in actual)) problems.push(`${path}.${key}: missing (stored ${fmt(expected[key])})`);
    else diffJson(actual[key], expected[key], tolerance, `${path}.${key}`, problems);
  }
  return problems;
}

// ---------------------------------------------------------------------------
// Run one family
// ---------------------------------------------------------------------------

/** The shareable slice of a DecideResult: everything except wall-clock timings. */
function pickResult(result) {
  return JSON.parse(JSON.stringify({
    answers: result.answers,
    truncated: result.truncated,
    length: result.length,
    prompts: result.prompts ?? null,
  }));
}

async function runFamily(id, { tolerance, record, baseline }) {
  const decisionCase = CASES[id] ?? { state: STATE, questions: QUESTIONS };
  const runtime = await createDecisionRuntime({ family: id, ...(decisionCase.options ?? {}) });
  try {
    const result = await runtime.decide(decisionCase.state, decisionCase.questions);
    const current = {
      family: id,
      recorded: new Date().toISOString(),
      info: JSON.parse(JSON.stringify(runtime.info ?? {})),
      options: decisionCase.options ?? null,
      state: decisionCase.state,
      questions: decisionCase.questions,
      result: pickResult(result),
    };

    if (record || !baseline) {
      return { status: record ? "RECORDED" : "SKIPPED", entry: current };
    }

    // A stale baseline answers a different question — re-record, don't guess.
    if (baseline.state !== current.state
      || JSON.stringify(baseline.questions) !== JSON.stringify(current.questions)
      || JSON.stringify(baseline.options ?? null) !== JSON.stringify(current.options)) {
      return {
        status: "FAIL",
        problems: [
          "the stored example no longer matches the script's; re-record this family with --record",
        ],
      };
    }

    const problems = [];
    diffJson(current.info, baseline.info ?? {}, 0, "info", problems);
    diffJson(current.result, baseline.result, tolerance, "result", problems);
    return problems.length
      ? { status: "FAIL", problems: problems.slice(0, 12) }
      : { status: "PASS" };
  } finally {
    try { await runtime.dispose?.(); } catch { /* dispose is best-effort */ }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(argv) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(usage());
    return 0;
  }

  const known = listDecisionFamilies();
  const selected = args.families ?? known;
  const unknown = selected.filter((id) => !known.includes(id));
  if (unknown.length) throw new Error(`unknown family/families: ${unknown.join(", ")} (known: ${known.join(", ")})`);

  const baselineData = await readBaselines(args.file);
  const outcomes = [];

  for (const id of selected) {
    const baseline = baselineData.families[id];
    process.stdout.write(`[${id}] `);
    let outcome;
    try {
      outcome = await runFamily(id, { tolerance: args.tolerance, record: args.record, baseline });
    } catch (error) {
      outcome = { status: "ERROR", problems: [String(error?.stack ?? error)] };
    }
    outcomes.push({ id, outcome });

    switch (outcome.status) {
      case "PASS":
        console.log(`PASS (tolerance ${args.tolerance})`);
        break;
      case "RECORDED":
        console.log("RECORDED");
        break;
      case "SKIPPED":
        console.log("SKIP (no stored baseline; run with --record to create one)");
        break;
      default: // FAIL / ERROR
        console.log(outcome.status);
        for (const line of outcome.problems ?? []) console.log(`  ${line}`);
        break;
    }

    if (outcome.entry) baselineData.families[id] = outcome.entry;
  }

  if (args.record) {
    await mkdir(dirname(args.file), { recursive: true });
    await writeFile(args.file, `${JSON.stringify(baselineData, null, 2)}\n`);
    console.log(`baselines written to ${args.file}`);
  }

  const count = (status) => outcomes.filter((o) => o.outcome.status === status).length;
  console.log(
    `\n${count("PASS")} passed, ${count("FAIL")} failed, ${count("ERROR")} errored, `
    + `${count("RECORDED")} recorded, ${count("SKIPPED")} skipped`);
  return count("FAIL") + count("ERROR") > 0 ? 1 : 0;
}

main(process.argv.slice(2))
  .then((code) => { process.exitCode = code; })
  .catch((error) => {
    // Argument/baseline mistakes read better as a single line; set
    // PARITY_DEBUG=1 when the stack is what you want.
    console.error(process.env.PARITY_DEBUG ? (error?.stack ?? error) : (error?.message ?? error));
    process.exitCode = 1;
  });
