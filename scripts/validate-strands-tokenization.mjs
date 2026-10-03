/**
 * Validate the chunked tokenisation scheme in src/strands.mjs against the real
 * Qwen3.5 tokenizer from the strands-decider checkpoint.
 *
 * Checks, for every demo preset question:
 *  1. concat(encode(chunk_i)) == encode(full question text) — the chunk
 *     boundaries coincide with pre-tokeniser piece boundaries, so positions
 *     derived from chunks are exact and the model sees training tokenisation.
 *  2. strandsOptionTokenIndices lands on each option line's last token: the
 *     token after it must decode to a newline-terminated piece.
 *
 * Needs network (one tokenizer download). Run: node scripts/validate-strands-tokenization.mjs
 */
import { AutoTokenizer } from "@huggingface/transformers";
import {
  renderStrandsQuestion,
  strandsOptionTokenIndices,
} from "../src/strands.mjs";
import { normalizeQuestions } from "../src/questions.mjs";
import { PRESETS } from "../demo/presets.mjs";

const tok = await AutoTokenizer.from_pretrained("StrandsAgents/strands-decider-2B-hobson-v19");
const encode = (text, addSpecial = false) =>
  Array.from(tok(text, { add_special_tokens: addSpecial }).input_ids.data, Number);
const decodeOne = (id) => tok.decode([id], { skip_special_tokens: false });

let checked = 0;
let failures = 0;
const fail = (msg) => { failures++; console.error(`FAIL: ${msg}`); };

for (const preset of PRESETS) {
  const normalized = normalizeQuestions(preset.questions);
  for (const q of normalized) {
    const rq = renderStrandsQuestion(q);
    const chunkIds = rq.chunks.map((c) => encode(c));
    const full = encode(rq.text);
    const concat = chunkIds.flat();
    checked++;
    if (concat.length !== full.length || concat.some((v, i) => v !== full[i])) {
      fail(`${preset.id}/${q.type}: chunked encoding != full encoding`);
      continue;
    }
    const k = rq.slotLabels.length;
    const lineChunks = chunkIds.slice(1, 1 + k);
    const lineIds = rq.chunks.slice(1, 1 + k).map((c) => encode(c.slice(0, -1)));
    let pos;
    try {
      pos = strandsOptionTokenIndices(lineIds, lineChunks);
    } catch (e) {
      fail(`${preset.id}/${q.type}: ${e.message}`);
      continue;
    }
    // Question-relative offset of each option chunk.
    let off = chunkIds[0].length;
    const chunkOffsets = [];
    for (let i = 0; i < k; i++) { chunkOffsets.push(off); off += lineChunks[i].length; }
    pos.forEach((p, i) => {
      const chunk = lineChunks[i];
      const scoredTok = chunk[p];
      const afterTok = chunk[p + 1];
      const scoredText = decodeOne(scoredTok);
      const line = rq.chunks[1 + i].slice(0, -1);
      // The scored token must be the tail of its option line…
      const tail = line.replace(/^[^\n]*?(\S+\s*)$/, "$1");
      if (!line.endsWith(scoredText.replace(/^[\sĠ]+/, "").replace(/^▁/, "")) && !scoredText.includes(tail.slice(-3))) {
        // …checked loosely: BPE spacing prefixes differ, so also accept the
        // token being a suffix of the line up to spacing.
        const stripped = scoredText.replace(/^[Ġ▁\s]+/, "");
        if (!line.endsWith(stripped)) fail(`${preset.id}/${q.type} opt${i}: scored token ${JSON.stringify(scoredText)} not a suffix of ${JSON.stringify(line)}`);
      }
      // …and the next token must terminate the line.
      if (afterTok === undefined) {
        fail(`${preset.id}/${q.type} opt${i}: scored token is the chunk's last token`);
      } else {
        const afterText = decodeOne(afterTok);
        if (!afterText.endsWith("\n")) {
          fail(`${preset.id}/${q.type} opt${i}: token after scored decodes to ${JSON.stringify(afterText)}, want a newline`);
        }
      }
      // Position must agree with the full-sequence layout.
      if (full[chunkOffsets[i] + p] !== scoredTok) {
        fail(`${preset.id}/${q.type} opt${i}: position mismatch vs full encoding`);
      }
    });
  }
}

console.log(`checked ${checked} questions, ${failures} failures`);
process.exit(failures ? 1 : 0);
