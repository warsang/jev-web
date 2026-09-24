// Local sanity check: load the real tokenizer from the model repo (no model
// weights) and confirm the three typed-decision markers encode to single
// tokens. Usage: node scripts/check-tokenizer.mjs [model] [revision]
import { AutoTokenizer } from "@huggingface/transformers";
import { typedMarkerIds, DEFAULT_MODEL, DEFAULT_REVISION } from "../src/index.mjs";

const model = process.argv[2] ?? DEFAULT_MODEL;
const revision = process.argv[3] ?? DEFAULT_REVISION;

const tokenizer = await AutoTokenizer.from_pretrained(model, { revision });
const encode = (text) => Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
const markers = typedMarkerIds(encode);
console.log("tokenizer ok:", model);
console.log("markers:", markers);
console.log("sample:", encode("I was charged twice.").slice(0, 12));
