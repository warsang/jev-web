// jev-web — pure input builder for open-jev-shaped models.
//
// Layout produced (mirrors the reference export):
//   [CLS] [STATE] state [Q] instructions [OPT] option … [SEP]
//
// `seg` carries a span slot per token: option tokens get their pair index,
// question-instruction tokens get `totalPairs + question index`, everything
// else -1. `pair_q` / `pair_opt` list the question slot and pair slot for
// every (question, option) pair; `groups` groups pair indices per question.
// The ONNX graph consumes these directly (see the model's export docs).

export class DecisionInputTooLongError extends Error {
  constructor(message) {
    super(message);
    this.name = "DecisionInputTooLongError";
  }
}

// Runtime-independent: take pre-tokenized ids and lay out the sequence.
export function buildDecisionInput({
  stateIds,
  markers, // { cls, sep, state, question, option } single token ids
  questions, // [{ type, instructionIds: number[], optionIds: number[][] }]
  maxStateTokens = 256,
  maxLen = 512,
}) {
  if (!Array.isArray(stateIds)) throw new TypeError("stateIds must be an array of token ids");
  if (!Array.isArray(questions) || questions.length === 0) {
    throw new TypeError("questions must be a non-empty array of tokenized questions");
  }
  for (const key of ["cls", "sep", "state", "question", "option"]) {
    if (!Number.isInteger(markers?.[key])) throw new TypeError(`markers.${key} must be a token id`);
  }

  const truncated = stateIds.length > maxStateTokens;
  const state = truncated ? stateIds.slice(0, maxStateTokens) : stateIds;

  const inputIds = [markers.cls, markers.state, ...state];
  const seg = inputIds.map(() => -1);
  const pairQ = [];
  const pairOpt = [];
  const groups = [];
  const totalPairs = questions.reduce((n, q) => n + q.optionIds.length, 0);

  questions.forEach((q, qi) => {
    inputIds.push(markers.question, ...q.instructionIds);
    seg.push(-1, ...q.instructionIds.map(() => totalPairs + qi));
    groups.push(
      q.optionIds.map((ids) => {
        inputIds.push(markers.option, ...ids);
        seg.push(-1, ...ids.map(() => pairOpt.length));
        pairQ.push(totalPairs + qi);
        pairOpt.push(pairOpt.length);
        return pairOpt.length - 1;
      }),
    );
  });
  inputIds.push(markers.sep);
  seg.push(-1);

  if (inputIds.length > maxLen) {
    throw new DecisionInputTooLongError(
      `decision input is ${inputIds.length} tokens, over the ${maxLen}-token limit ` +
        `(state capped at ${maxStateTokens}) — shorten the questions or the state`,
    );
  }
  return { inputIds, seg, pairQ, pairOpt, groups, truncated, length: inputIds.length, totalPairs };
}

// Shared marker lookup: every marker must encode to exactly one token id.
// (Custom typed-decision exports add [STATE]/[Q]/[OPT] to the tokenizer; if
// they are missing, the model is not a typed-decision export.)
export function typedMarkerIds(encodeFn, { markers = ["[STATE]", "[Q]", "[OPT]"], cls = "[CLS]", sep = "[SEP]" } = {}) {
  const one = (text) => {
    const ids = encodeFn(text);
    if (!Array.isArray(ids) || ids.length !== 1) {
      throw new Error(`marker ${text} must encode to exactly one token (got ${ids?.length ?? 0})`);
    }
    return ids[0];
  };
  const [state, question, option] = markers.map(one);
  return { cls: one(cls), sep: one(sep), state, question, option };
}
