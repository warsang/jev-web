/**
 * Compile-only consumer for the shipped declarations (`src/index.d.ts`).
 *
 * `npm run typecheck` compiles this file against the package's own public
 * entry (self-referenced through the `exports` map) with `strict` and
 * `nodenext`, so a signature that drifts from the runtime — or a names export
 * that disappears — fails CI. Nothing here is executed.
 */
import {
  // core
  createDecider,
  type SessionDecider,
  type DecideResult,
  type Answer,
  type Question,
  type NormalizedQuestion,
  type Decider,
  // questions / encoding
  normalizeQuestions,
  QUESTION_TYPES,
  NOUL_OPTIONS,
  buildDecisionInput,
  typedMarkerIds,
  DecisionInputTooLongError,
  answersFromScores,
  softmaxWithTemperature,
  // families
  createLayaDecider,
  createStrandsDecider,
  createBekkoDecider,
  createDecision2Decider,
  createKevDecider,
  createVegaDecider,
  createClefDecider,
  createDecision3Decider,
  // registry
  createDecisionRuntime,
  registerDecisionFamily,
  listDecisionFamilies,
  getDecisionFamily,
  // per-family helpers
  buildLayaSequence,
  LAYA_DEFAULT_MAX_PREFIXES,
  renderBekkoRequest,
  tokenizeBekkoRequest,
  packKevSequence,
  KEV_VARIANTS,
  buildVegaPrompt,
  VEGA_DEFAULTS,
  VEGA_QTYPE,
  packDecision3Batch,
  decision3CandidateCodes,
  STRANDS_VARIANTS,
  BEKKO_VARIANTS,
  DECISION2_VARIANTS,
  DECISION3_VARIANTS,
  // config defaults
  DEFAULT_MODEL,
  DEFAULT_REVISION,
  DEFAULT_TEMPERATURE,
  resolveDevice,
  resolveDtype,
  detectWebGPU,
  loadModelConfig,
  // fetch / cache
  MODEL_CACHE,
  HttpError,
  JsonParseError,
  requestPersistentStorage,
  fetchCachedBytes,
  fetchCachedJson,
  fetchOnnxExternalData,
} from "jev-web";

const questions: Question[] = [
  {
    type: "choice",
    instructions: "Which product area is the message about?",
    options: ["fees & charges", "refund & dispute", "card", "other"],
    criteria: { card: "payment card issues" },
  },
  { type: "noul", instructions: "The customer is asking for a refund." },
  {
    type: "score",
    instructions: "How negative is the message?",
    options: ["very negative", "negative", "neutral", "positive"],
  },
];

// @ts-expect-error — choice questions require an options list
const invalid: Question = { type: "choice", instructions: "missing options" };
void invalid;

export async function useSessionDecider(): Promise<Answer[]> {
  const decider: SessionDecider = await createDecider({
    device: "auto",
    dtype: "q4",
    maxStateTokens: 256,
    onProgress: (p) => void `${p.phase} ${p.file ?? ""}`,
  });
  const result: DecideResult = await decider.decide("I was charged twice.", questions);
  await decider.decideMany([{ state: "again", questions }]);
  await decider.dispose();
  if (result.truncated) void result.length;
  return result.answers;
}

export async function useFamilies(): Promise<void> {
  const runtimes: Decider[] = await Promise.all([
    createLayaDecider({ maxPrefixes: LAYA_DEFAULT_MAX_PREFIXES }),
    createStrandsDecider({ variant: "qwen3.5-v1" }),
    createBekkoDecider({ variant: "68m", attentionBudget: 4096 }),
    createDecision2Decider({ variant: "sol" }),
    createKevDecider({ variant: "4b" }),
    createVegaDecider({ variant: "q8", alpha: "0.1", risk: "0.1" }),
    createClefDecider({ maxLen: 8192 }),
    createDecision3Decider({ batchSize: 4 }),
  ]);
  for (const runtime of runtimes) {
    const { answers, truncated, length, timings } = await runtime.decide("state", questions);
    for (const a of answers) {
      if (a.type === "choice") void [a.choice, a.index, a.confidence];
      else if (a.type === "score") void [a.score, a.level, a.normalizedScore];
      else void a.noul;
      void a.probabilities;
    }
    void [truncated, length, timings];
    await runtime.dispose();
  }
}

export async function useRegistry(): Promise<void> {
  registerDecisionFamily("consumer-test", {
    defaults: { model: "org/model" },
    create: async () => ({
      info: { family: "consumer-test" },
      decide: async (): Promise<DecideResult> => ({
        answers: [],
        truncated: false,
        length: 0,
      }),
      dispose: async () => {},
    }),
  });
  const runtime = await createDecisionRuntime({
    family: "consumer-test",
    fallback: ["laya"],
    onFallback: ({ family, error }) => void `${family} ${String(error)}`,
  });
  await runtime.decide("state", questions);
  listDecisionFamilies().map((id) => getDecisionFamily(id) !== null);
}

export function useEncoding(): void {
  const encode = (text: string) => [text.length];
  const markers = typedMarkerIds(encode, { markers: ["[STATE]", "[Q]", "[OPT]"] });
  const normalized: NormalizedQuestion[] = normalizeQuestions(questions);
  const input = buildDecisionInput({
    stateIds: [1, 2, 3],
    markers,
    questions: [{ type: "noul", instructionIds: [4], optionIds: [[5], [6]] }],
    maxStateTokens: 256,
  });
  if (input.truncated) void input.length;
  if (QUESTION_TYPES.size !== 3) throw new DecisionInputTooLongError("unexpected");
  const scored: Answer[] = answersFromScores([0.1, 0.9], {
    questions: normalized,
    groups: input.groups,
    temperature: DEFAULT_TEMPERATURE,
  });
  void [softmaxWithTemperature([1, 2]), scored, NOUL_OPTIONS];
}

export function useHelpers(): void {
  const encode = (text: string) => [...text].map((c) => c.charCodeAt(0) % 16);
  const laya = buildLayaSequence({
    encode,
    ids: { cls: 1, sep: 2, mask: 3 },
    state: "state",
    question: normalizeQuestions(questions)[0],
  });
  void laya.truncated;

  const bekko = renderBekkoRequest(normalizeQuestions(questions)[0], "state");
  void tokenizeBekkoRequest(bekko, encode, { candidate_length: 96 });
  void bekko.candidates.map((c) => c.text);

  const packed = packKevSequence({
    stateText: "state",
    questions: normalizeQuestions(questions),
    encode,
    delims: { state: 7, question: 8, option_start: 9, option_end: 10, decide: 11 },
    limits: { max_state_tokens: 8192, max_branch_tokens: 8192, max_options: 255 },
  });
  void [packed.tokens, packed.groups, packed.truncated, KEV_VARIANTS["4b"]];

  const prompt = buildVegaPrompt({
    encode,
    stateText: "state",
    qdef: { t: "choice", ins: "which?", crit: null },
    maxLen: VEGA_DEFAULTS.maxLen,
  });
  void [prompt.ids, prompt.spans.options, prompt.truncated, VEGA_QTYPE.choice];

  const batch = packDecision3Batch({ sequences: [[1, 2, 3]], counts: [2], padId: 0 });
  void [batch.B, batch.width, decision3CandidateCodes()];

  void [
    STRANDS_VARIANTS["qwen3.5-v1"],
    BEKKO_VARIANTS["400m"],
    DECISION2_VARIANTS.sol,
    DECISION3_VARIANTS.nano,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
  ];
}

export async function useConfigAndFetch(): Promise<void> {
  const device = resolveDevice({ device: "auto", webgpu: await detectWebGPU() });
  resolveDtype(device, "auto");
  const config = await loadModelConfig({ model: DEFAULT_MODEL, revision: DEFAULT_REVISION });
  void [config.temperature, config.markers, config.maxStateTokens, config.maxLen, config.pool, config.source];

  if (typeof caches !== "undefined") await caches.open(MODEL_CACHE);
  await requestPersistentStorage();
  try {
    const bytes: Uint8Array = await fetchCachedBytes("https://huggingface.co/x/resolve/a/config.json", {
      force: false,
      persist: true,
    });
    void bytes.byteLength;
    const json = await fetchCachedJson("https://huggingface.co/x/resolve/a/optional.json", { nullOn404: true });
    void json;
    const external = await fetchOnnxExternalData("https://huggingface.co/x/resolve/a/model.onnx");
    void external?.data.byteLength;
  } catch (error) {
    if (error instanceof HttpError) void error.status;
    if (error instanceof JsonParseError) void error.url;
  }
}
