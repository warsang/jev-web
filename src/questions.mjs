// jev-web — question normalization for typed-decision models.
//
// The model answers three question kinds:
//   choice — pick one of 2..255 options
//   score  — ordered levels (2..10); the answer is the expected level index
//   noul   — yes/no; the answer is p(yes)

export const QUESTION_TYPES = new Set(["choice", "score", "noul"]);
export const NOUL_OPTIONS = ["no", "yes"];

export function normalizeQuestions(questions) {
  if (!Array.isArray(questions) || questions.length === 0) {
    throw new TypeError("questions must be a non-empty array");
  }
  return questions.map((q, i) => {
    const type = q?.type;
    if (!QUESTION_TYPES.has(type)) {
      throw new TypeError(`questions[${i}].type must be one of ${[...QUESTION_TYPES].join("|")}`);
    }
    const instructions = String(q?.instructions ?? "").trim();
    if (!instructions) throw new TypeError(`questions[${i}].instructions is required`);

    let options;
    if (type === "noul") {
      options = Array.isArray(q.options) && q.options.length ? q.options.map(String) : [...NOUL_OPTIONS];
      if (options.length !== 2) throw new TypeError(`questions[${i}]: noul takes exactly 2 options`);
    } else if (type === "choice") {
      options = (q.options ?? []).map(String);
      if (options.length < 2) throw new TypeError(`questions[${i}]: choice needs at least 2 options`);
      if (options.length > 255) throw new TypeError(`questions[${i}]: choice supports at most 255 options`);
    } else {
      options = (q.options ?? []).map(String);
      if (options.length < 2 || options.length > 10) {
        throw new TypeError(`questions[${i}]: score needs 2-10 ordered levels`);
      }
    }
    return { type, instructions, options, ...(q?.criteria !== undefined ? { criteria: q.criteria } : {}) };
  });
}
