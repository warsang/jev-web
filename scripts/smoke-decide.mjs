// Local smoke test: run the reference model on the model card's banking
// example. Works in browsers via vite (see dev/), and in Node via
// onnxruntime-node (device: cpu, dtype: q4|fp16|fp32).
//
//   node scripts/smoke-decide.mjs [device] [dtype]
//
import { createDecider } from "../src/index.mjs";

const device = process.argv[2] ?? "cpu";
const dtype = process.argv[3] ?? "q4";

const decider = await createDecider({
  device,
  dtype,
  onProgress: (p) => {
    if (p.total && p.loaded) {
      process.stderr.write(`\r[${p.phase}] ${p.file ?? ""} ${(p.loaded / 1e6) | 0}/${(p.total / 1e6) | 0} MB`);
    }
  },
});
process.stderr.write("\n");
console.log("info:", decider.info);

const state =
  "I was charged twice for the same order and nobody answers my emails. I want my money back now.";
const { answers, truncated, length } = await decider.decide(state, [
  {
    type: "choice",
    instructions: "Which product area is the message about?",
    options: ["fees & charges", "pin & security", "refund & dispute", "top-up", "exchange & fiat", "atm & cash", "transfer", "card", "account & identity", "other"],
  },
  { type: "noul", instructions: "The customer is asking for a refund." },
  {
    type: "score",
    instructions: "How positive is the sentiment of this message?",
    options: ["very negative", "negative", "neutral", "positive", "very positive"],
  },
]);

console.log(`tokens: ${length}${truncated ? " (state truncated)" : ""}`);
for (const a of answers) console.log(JSON.stringify(a, null, 2));
