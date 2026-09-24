// jev-web — score → answer decoding for typed-decision models.
//
// The graph returns one logit per (question, option) pair. Each question's
// options form a softmax group; a post-hoc temperature (fitted by the model
// authors) calibrates the distribution.

export function softmaxWithTemperature(logits, temperature = 1) {
  const t = Number.isFinite(temperature) && temperature > 0 ? temperature : 1;
  if (!Array.isArray(logits) || logits.length === 0) throw new TypeError("logits must be a non-empty array");
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp((Number(x) - max) / t));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function answersFromScores(scores, { questions, groups, temperature = 1 }) {
  if (!Array.isArray(scores)) throw new TypeError("scores must be an array");
  return questions.map((q, i) => {
    const pairIndices = groups[i];
    const probs = softmaxWithTemperature(pairIndices.map((p) => scores[p]), temperature);
    const probabilities = Object.fromEntries(q.options.map((option, j) => [option, probs[j]]));
    const confidence = Math.max(...probs);
    const best = probs.indexOf(confidence);

    if (q.type === "choice") {
      return { type: "choice", choice: q.options[best], index: best, probabilities, confidence };
    }
    if (q.type === "score") {
      const expected = probs.reduce((acc, p, j) => acc + p * j, 0);
      return { type: "score", score: expected, level: best, probabilities, confidence };
    }
    // noul: probabilities are { no, yes }; the answer is p(yes).
    const yes = probabilities[q.options[1]];
    return { type: "noul", noul: yes, probabilities, confidence: Math.max(yes, 1 - yes) };
  });
}
