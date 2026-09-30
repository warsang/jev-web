// Raw-text parsing baseline for the jev-web demo.
//
// This is deliberately a *fair* implementation of the way most people solve
// "get structured decisions out of text" without a model: hand-written keyword
// and regex rules, one rule set per question.
//
// It is here to be compared, not to be strawmanned. It is fast (microseconds),
// free, and works well on text you anticipated. It breaks in exactly the ways
// it always breaks:
//   * negation      — "this is NOT about a refund"
//   * contrast      — "the sound is fantastic, but the app crashes"
//   * mixed topics  — two intents in one state, one hard label out
//   * distribution  — a hard label carries no confidence to threshold on
//
// Every answer is tagged with the rule that produced it, so the demo can show
// *why* it agrees or disagrees with the model rather than just asserting that
// models are better.

const STOP = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "and", "or", "of",
  "for", "to", "in", "on", "at", "by", "with", "about", "this", "that", "it", "its",
  "my", "me", "i", "you", "your", "we", "our", "they", "them", "he", "she", "his", "her",
  "do", "does", "did", "have", "has", "had", "can", "could", "would", "should", "will",
  "shall", "may", "might", "must", "not", "no", "yes", "s", "t", "am",
]);

// A small, realistic synonym lexicon. Every real keyword parser has one; giving
// the baseline the same chance is the whole point of the comparison.
const LEXICON = {
  refund: ["refund", "reimburse", "money back", "chargeback", "reversed", "credited back", "charge back"],
  dispute: ["dispute", "disputed", "fraud", "unauthorised", "unauthorized", "stolen", "scammed", "wrong order"],
  card: ["card", "visa", "mastercard", "amex", "debit", "credit card", "blocked card"],
  fees: ["fee", "fees", "charge", "charges", "commission", "overcharged", "interest"],
  account: ["account", "login", "log in", "password", "2fa", "locked out", "verify", "kyc", "username"],
  transfer: ["transfer", "transferring", "sent", "wire", "paypal", "iban", "bacc"],
  atm: ["atm", "cash", "cashpoint", "withdraw", "withdrawal", "dispenser"],
  topup: ["top up", "topup", "topped up", "deposit", "add money", "add funds"],
  exchange: ["exchange", "rate", "converted", "conversion", "spread", "fx"],
  positive: ["great", "love", "excellent", "amazing", "fantastic", "brilliant", "perfect", "best", "wonderful", "recommend", "five stars", "10/10", "smooth", "fast"],
  negative: ["terrible", "awful", "hate", "broken", "buggy", "unusable", "worst", "refund", "complaint", "crash", "crashes", "crashing", "unresponsive", "disappointed", "furious", "ridiculous", "waste"],
  neutral: ["okay", "fine", "average", "alright", "mixed", "neither", "so-so"],
  // Keys are `very` / `mild` (intensifier buckets), not `strong`.
  negativeScore: {
    very: ["furious", "outrageous", "disaster", "worst", "unacceptable", "appalling"],
    mild: ["disappointed", "annoying", "slow", "buggy", "frustrating", "mediocre"],
  },
};

const NEGATORS = new Set(["not", "no", "never", "without", "isnt", "wasnt", "dont", "didnt", "doesnt", "cant", "cannot", "wont", "nobody"]);

function tokens(text) {
  return String(text ?? "").toLowerCase().replace(/[^a-z0-9\s'’-]/g, " ").split(/\s+/).filter(Boolean);
}

// Phrase-aware count: a lexicon entry can be a single token or a multiword phrase.
function countHits(haystack, terms) {
  const hits = [];
  for (const term of terms) {
    if (term.includes(" ")) {
      if (haystack.includes(term)) hits.push(term);
    } else {
      const n = haystack.split(/\s+/).filter((t) => t.startsWith(term)).length;
      if (n) hits.push(term);
    }
  }
  return hits;
}

// Negation window: "not a refund", "never got my money back" -> the 3 tokens
// before a hit flip its sign. Crude on purpose; this is what regex gets you.
function negated(text, hit) {
  const t = tokens(text);
  let i = t.findIndex((w) => w.startsWith(hit) || hit.startsWith(w));
  if (i <= 0) return false;
  return t.slice(Math.max(0, i - 3), i).some((w) => NEGATORS.has(w));
}

function sentimentScore(text) {
  const neg = countHits(text, LEXICON.negative).length;
  const pos = countHits(text, LEXICON.positive).length;
  if (pos === 0 && neg === 0) return null;
  return neg / (neg + pos); // 0 = all positive, 1 = all negative
}

/**
 * Answer one question with hand-written rules.
 * @returns {{type:string, value:any, rule:string, confidence:null, hits:string[], negated:boolean, ruled:boolean}}
 */
export function parseOne(state, question) {
  const text = ` ${String(state ?? "").toLowerCase()} `;
  const stems = tokens(question.instructions).filter((t) => !STOP.has(t));

  if (question.type === "noul") {
    // Try the stems of the instruction as the keyword set first
    // ("...asking for a refund" -> refund), then a yes/no lexicon.
    const own = countHits(text, stems).length;
    const fallbackTerms = stems.some((s) => /refund|money|return/.test(s)) ? LEXICON.refund
      : stems.some((s) => /urgent|immediately|asap/.test(s)) ? ["urgent", "immediately", "asap", "now", "right away"]
      : stems.some((s) => /blocked|locked/.test(s)) ? ["blocked", "locked", "locked out"]
      : [];
    const hits = own ? countHits(text, stems) : countHits(text, fallbackTerms);
    const flipped = hits.some((h) => negated(text, h));
    if (!hits.length) {
      return { type: "noul", value: null, rule: "no rule matched", confidence: null, hits: [], negated: false, ruled: false };
    }
    return {
      type: "noul",
      value: flipped ? 0 : 1,
      rule: `keyword ${flipped ? "(negated) " : ""}"${hits[0]}"`,
      confidence: null, hits, negated: flipped, ruled: true,
    };
  }

  if (question.type === "choice") {
    // Score each option by its own words + the matching lexicon entry.
    const scored = question.options.map((option) => {
      const optTokens = tokens(option).filter((t) => !STOP.has(t));
      const own = countHits(text, optTokens);
      let lex = [], lexKey = null;
      for (const [key, terms] of Object.entries(LEXICON)) {
        if (key === "positive" || key === "negative" || key === "neutral" || key === "negativeScore") continue;
        const h = countHits(text, terms);
        if (h.length) { lex = h; lexKey = key; break; }
      }
      const negatedHits = (own.length ? own : lex).filter((h) => negated(text, h));
      return { option, score: own.length + lex.length, own, lex, lexKey, negated: negatedHits.length > 0 };
    });
    const best = scored.reduce((a, b) => (b.score > a.score ? b : a), scored[0]);
    if (!best || best.score === 0) {
      return { type: "choice", value: null, rule: "no rule matched", confidence: null, hits: [], negated: false, ruled: false };
    }
    return {
      type: "choice",
      value: best.option,
      rule: `${best.lexKey ? `lexicon:${best.lexKey} ` : ""}keyword "${(best.own[0] ?? best.lex[0])}"${best.negated ? " (negated)" : ""}`,
      confidence: null,
      hits: best.own.length ? best.own : best.lex,
      negated: best.negated,
      ruled: true,
    };
  }

  // score
  const s = sentimentScore(text);
  if (s === null) {
    return { type: "score", value: null, rule: "no sentiment keywords", confidence: null, hits: [], negated: false, ruled: false };
  }
  const n = question.options.length;
  const intense = countHits(text, LEXICON.negativeScore.very).length;
  const idx = Math.min(n - 1, Math.max(0, Math.round(s * (n - 1))));
  return {
    type: "score",
    value: idx,
    rule: `${intense ? "intense+soft negative" : "keyword polarity"} (${countHits(text, LEXICON.negative).length} neg / ${countHits(text, LEXICON.positive).length} pos)`,
    confidence: null,
    hits: [...countHits(text, LEXICON.positive), ...countHits(text, LEXICON.negative)],
    negated: false,
    ruled: true,
  };
}

/**
 * One full "decision run" the naive way: a separate pass per question.
 * @returns {{answers:Array, passes:number, ms:number, matched:number}}
 */
export function parseAll(state, questions) {
  const t0 = performance.now();
  const answers = questions.map((q) => parseOne(state, q));
  const ms = performance.now() - t0;
  return {
    answers,
    passes: questions.length, // one rule set evaluated per question
    ms,
    matched: answers.filter((a) => a.ruled).length,
  };
}
