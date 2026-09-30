// Example inputs for the jev-web demo.
//
// Each preset is a "moment of truth" for keyword parsing:
//   easy        — a fair rule matches. Both approaches agree; the model adds confidence.
//   negation    — "NOT about a refund" inverts the obvious keyword.
//   contrast    — praise + complaint in one sentence. Keywords read the praise.
//   mixed       — two intents in one state; a hard label must pick one.
//   edge        — no in-domain signal at all. The rule parser has nothing to say.

export const PRESETS = [
  {
    id: "easy",
    label: "Banking triage",
    blurb: "easy — rules can win here",
    state:
      "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    questions: [
      {
        type: "choice",
        instructions: "Which product area is the message about?",
        options: ["fees & charges", "refund & dispute", "card", "other"],
      },
      { type: "noul", instructions: "The customer is asking for a refund." },
      {
        type: "score",
        instructions: "How negative is the message?",
        options: ["very negative", "negative", "neutral", "positive", "very positive"],
      },
    ],
  },
  {
    id: "negation",
    label: "Negation trap",
    blurb: "negation — the obvious keyword is inverted",
    state:
      "This is not about a refund. My card was stolen in Madrid and the bank refuses to help. I never received the money back I asked about.",
    questions: [
      {
        type: "choice",
        instructions: "Which product area is the message about?",
        options: ["fees & charges", "refund & dispute", "card", "other"],
      },
      { type: "noul", instructions: "The customer is asking for a refund." },
      { type: "noul", instructions: "The customer reports a compromised card." },
    ],
  },
  {
    id: "contrast",
    label: "Contradictory review",
    blurb: "contrast — praise and complaint in one sentence",
    state:
      "The sound quality of the speakers is fantastic, and the app is beautiful. But it crashes every single time I try to export a file, which is the only thing I bought it for.",
    questions: [
      {
        type: "score",
        instructions: "How positive is this review overall?",
        options: ["very negative", "negative", "neutral", "positive", "very positive"],
      },
      { type: "noul", instructions: "The reviewer would recommend this product to a friend." },
      { type: "noul", instructions: "The product has a functional problem." },
    ],
  },
  {
    id: "mixed",
    label: "Mixed intent",
    blurb: "mixed — two intents, one hard label",
    state:
      "My transfer is late again. Also my card was declined at the ATM yesterday even though the app says the balance is fine.",
    questions: [
      {
        type: "choice",
        instructions: "Which product area is the message about?",
        options: ["fees & charges", "refund & dispute", "card", "atm & cash", "transfer", "other"],
      },
      { type: "noul", instructions: "The customer is reporting a failed transfer." },
      { type: "noul", instructions: "The customer had a card declined at an ATM." },
    ],
  },
  {
    id: "edge",
    label: "Out of domain",
    blurb: "out of domain — rules have nothing to say",
    state:
      "We have been a customer for nine years and would like to discuss a partnership with your enterprise team.",
    questions: [
      {
        type: "choice",
        instructions: "Which product area is the message about?",
        options: ["fees & charges", "refund & dispute", "card", "other"],
      },
      { type: "noul", instructions: "The customer is asking for a refund." },
      { type: "noul", instructions: "The message expresses dissatisfaction." },
    ],
  },
];
