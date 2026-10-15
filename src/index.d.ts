/**
 * jev-web — type definitions for the public API (`src/index.mjs`).
 *
 * Hand-maintained and verified by `npm run typecheck`, which compiles
 * `types-test/consumer.ts` against these declarations with `strict` and
 * `nodenext` resolution. Keep the named exports here in sync with
 * `src/index.mjs`.
 */

// ---------------------------------------------------------------------------
// Shared vocabulary
// ---------------------------------------------------------------------------

export type Device = "auto" | "wasm" | "webgpu";
export type QuestionType = "choice" | "score" | "noul";

/** A typed decision question, as accepted by `decide()` / `normalizeQuestions`. */
export interface ChoiceQuestion {
  type: "choice";
  /** Rendered into the prompt; any JSON value, strings pass through. */
  instructions: unknown;
  /** 2..255 options. */
  options: string[];
  /** Optional per-option descriptions ({label: description}). */
  criteria?: Record<string, unknown>;
  /** Optional field id (clef renders it into the record). */
  id?: string;
}

export interface NoulQuestion {
  type: "noul";
  instructions: unknown;
  /** `[no, yes]`; defaults to ["no", "yes"] when omitted. */
  options?: string[];
  criteria?: Record<string, unknown>;
  id?: string;
}

export interface ScoreQuestion {
  type: "score";
  instructions: unknown;
  /** 2..10 ordered levels. */
  options: string[];
  criteria?: Record<string, unknown>;
  id?: string;
}

export type Question = ChoiceQuestion | NoulQuestion | ScoreQuestion;

/** A question after `normalizeQuestions()`: instructions flattened, options defaulted. */
export interface NormalizedQuestion {
  type: QuestionType;
  instructions: string;
  options: string[];
  criteria?: unknown;
  id?: string;
}

// ---------------------------------------------------------------------------
// Answers and results
// ---------------------------------------------------------------------------

export interface AnswerCommon {
  type: QuestionType;
  confidence: number;
  probabilities?: Record<string, number>;
  /** Family-specific additions (vega: predictionSet/abstain/unbound/…). */
  [key: string]: unknown;
}

export interface ChoiceAnswer extends AnswerCommon {
  type: "choice";
  choice: string;
  index: number;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer extends AnswerCommon {
  type: "score";
  score: number;
  level: number;
  probabilities: Record<string, number>;
  /** Reference level texts (families that ship a legend). */
  legend?: Record<string, string>;
  /** bekko only: score mapped into [0, 1]. */
  normalizedScore?: number;
}

export interface NoulAnswer extends AnswerCommon {
  type: "noul";
  /** p(yes). */
  noul: number;
  probabilities: Record<string, number>;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface DecideResult {
  answers: Answer[];
  /**
   * True iff any user text (state, instructions or an option) was shortened
   * to fit the model. Families that cannot silently shorten (kev, strands
   * through its option list, decision2, decision3) throw instead.
   */
  truncated: boolean;
  /** Tokens the model actually consumed (family-dependent: one sequence, the batch total, …). */
  length: number;
  /** Decoded prompt(s); shape is family-specific (string or per-question object). */
  prompts?: unknown[];
  timings?: Record<string, number | number[] | null> | null;
  [key: string]: unknown;
}

export interface DeciderInfo {
  family: string;
  model?: string;
  revision?: string | null;
  device?: Device | string;
  [key: string]: unknown;
}

/**
 * The shared runtime contract: `info`, `decide()`, `dispose()`. The serial
 * queue inside each runtime makes overlapping `decide()` calls safe.
 */
export interface Decider {
  info: DeciderInfo;
  decide(state: unknown, questions: Question[]): Promise<DecideResult>;
  dispose(): Promise<void>;
}

/** open-jev session decider: adds helper `decideMany`. */
export interface SessionDecider extends Decider {
  decideMany(cases: Array<{ state: unknown; questions: Question[] }>): Promise<DecideResult[]>;
}

// ---------------------------------------------------------------------------
// Shared options
// ---------------------------------------------------------------------------

export interface ProgressEvent {
  phase: string;
  file?: string | null;
  loaded?: number | null;
  total?: number | null;
  status?: string | null;
  progress?: number | null;
  [key: string]: unknown;
}

export type ProgressCallback = (event: ProgressEvent) => void;
export type FetchImpl = (url: string, init?: RequestInit) => Promise<Response>;
export type SessionFactory =
  (url: string, sessionOptions?: Record<string, unknown>) => Promise<unknown>;

export interface BaseDeciderOptions {
  /** Injected onnxruntime-web (lazy import otherwise). */
  ort?: unknown;
  /** Injected @huggingface/transformers (lazy import otherwise). */
  transformers?: unknown;
  fetchImpl?: FetchImpl;
  /** ORT wasm/loader directory for self-hosted builds. */
  wasmPaths?: string | null;
  device?: Device | string;
  /** Custom session loading (local files, bundled weights, offline mirrors). */
  sessionFactory?: SessionFactory | null;
  onProgress?: ProgressCallback | null;
  /** Global object used for WebGPU detection (`navigator.gpu`). */
  scope?: typeof globalThis;
}

// ---------------------------------------------------------------------------
// Core session (open-jev)
// ---------------------------------------------------------------------------

export interface OpenJevOptions extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  dtype?: "auto" | "q4" | "q4f16" | "fp16" | "fp32" | "q8" | string;
  temperature?: number | null;
  maxStateTokens?: number | null;
}

export function createDecider(opts?: OpenJevOptions): Promise<SessionDecider>;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export interface DecisionFamilyDescriptor {
  defaults?: Record<string, unknown>;
  create: (opts: Record<string, unknown>) => Promise<Decider>;
}

export interface CreateDecisionRuntimeOptions {
  family?: string;
  fallback?: string[];
  onFallback?: ((info: { family: string; error: unknown }) => void) | null;
  [key: string]: unknown;
}

export function registerDecisionFamily(id: string, descriptor: DecisionFamilyDescriptor): string;
export function listDecisionFamilies(): string[];
export function getDecisionFamily(id: string): DecisionFamilyDescriptor | null;
export function createDecisionRuntime(opts?: CreateDecisionRuntimeOptions): Promise<Decider>;

// ---------------------------------------------------------------------------
// Questions / input building / decoding helpers
// ---------------------------------------------------------------------------

export const QUESTION_TYPES: Set<QuestionType>;
export const NOUL_OPTIONS: string[];
export function normalizeQuestions(questions: Question[]): NormalizedQuestion[];

export class DecisionInputTooLongError extends Error {}

export interface TokenizedQuestion {
  type?: QuestionType;
  instructionIds: number[];
  optionIds: number[][];
}

export interface TypedMarkers {
  cls: number;
  sep: number;
  state: number;
  question: number;
  option: number;
}

export function buildDecisionInput(spec: {
  stateIds: number[];
  markers: TypedMarkers;
  questions: TokenizedQuestion[];
  maxStateTokens?: number;
  maxLen?: number;
}): {
  inputIds: number[];
  seg: number[];
  pairQ: number[];
  pairOpt: number[];
  groups: number[][];
  truncated: boolean;
  length: number;
  totalPairs: number;
};

export function typedMarkerIds(
  encodeFn: (text: string) => number[],
  opts?: { markers?: string[]; cls?: string; sep?: string },
): TypedMarkers;

export function softmaxWithTemperature(logits: number[], temperature?: number): number[];

export function answersFromScores(
  scores: ArrayLike<number>,
  spec: { questions: NormalizedQuestion[]; groups: number[][]; temperature?: number },
): Answer[];

// ---------------------------------------------------------------------------
// open-jev config defaults
// ---------------------------------------------------------------------------

export const DEFAULT_MODEL: string;
export const DEFAULT_REVISION: string;
export const DEFAULT_TEMPERATURE: number;
export const DEFAULT_MARKERS: string[];
export const DEFAULT_MAX_STATE_TOKENS: number;
export const DEFAULT_MAX_LEN: number;

export function resolveDevice(opts?: { device?: string; webgpu?: boolean }): string;
export function resolveDtype(device: string, dtype?: string): string;
export function detectWebGPU(scope?: typeof globalThis): Promise<boolean>;
export function loadModelConfig(opts?: {
  model?: string;
  revision?: string;
  fetchImpl?: FetchImpl;
}): Promise<{
  temperature: number;
  markers: string[];
  maxStateTokens: number;
  maxLen: number;
  pool: string;
  source: string;
}>;

// ---------------------------------------------------------------------------
// fetch / cache helpers
// ---------------------------------------------------------------------------

export const MODEL_CACHE: string;

export class HttpError extends Error {
  url: string;
  status: number | undefined;
}

export class JsonParseError extends Error {
  url: string;
  cause?: unknown;
}

export function requestPersistentStorage(): Promise<boolean>;

export function fetchCachedBytes(url: string, opts?: {
  fetchImpl?: FetchImpl;
  onProgress?: ProgressCallback | null;
  cacheName?: string;
  force?: boolean;
  persist?: boolean;
}): Promise<Uint8Array>;

export function fetchCachedJson(url: string, opts?: {
  fetchImpl?: FetchImpl;
  onProgress?: ProgressCallback | null;
  cacheName?: string;
  force?: boolean;
  persist?: boolean;
  nullOn404?: boolean;
}): Promise<unknown | null>;

export function fetchOnnxExternalData(url: string, opts?: {
  fetchImpl?: FetchImpl;
  onProgress?: ProgressCallback | null;
  force?: boolean;
}): Promise<{ path: string; data: Uint8Array } | null>;

// ---------------------------------------------------------------------------
// Laya
// ---------------------------------------------------------------------------

export interface LayaOptions extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  subfolder?: string;
  encoderUrl?: string | null;
  headUrl?: string | null;
  tokenizerModel?: string | null;
  tokenizerSubfolder?: string | null;
  maxLen?: number;
  headMaxLen?: number;
  maxPrefixes?: number;
  temperatures?: number[];
  temperatureByOptions?: Record<string, number>;
}

export function createLayaDecider(opts?: LayaOptions): Promise<Decider>;

export function buildLayaSequence(spec: {
  encode: (text: string) => number[];
  ids: { cls: number; sep: number; mask: number };
  state: unknown;
  question: NormalizedQuestion;
  maxLen?: number;
  headMaxLen?: number;
  maxPrefixes?: number;
}): { ids: number[]; markers: number[]; qtype: number; truncated: boolean };

export function collateLayaItems(
  items: Array<{ ids: number[]; markers: number[]; qtype: number }>,
  padId?: number,
): {
  n: number;
  L: number;
  M: number;
  inputIds: BigInt64Array;
  attention: BigInt64Array;
  markerPos: BigInt64Array;
  markerMask: Uint8Array;
  qtype: BigInt64Array;
};

export function layaAnswersFromLogits(spec: {
  logits: ArrayLike<number>;
  questions: NormalizedQuestion[];
  markerCounts: number[];
  temperatures?: number[];
  temperatureByOptions?: Record<string, number>;
}): Answer[];

export function layaTokenizerAdapter(tok: unknown): {
  encode: (text: string) => number[];
  decode: (ids: number[]) => string;
  ids: { cls: number; sep: number; mask: number };
  pad: number;
};

export function renderLayaOptions(question: NormalizedQuestion): string[];
export function layaTempBucket(qtype: number, k: number): string;
export function clampTemperature(t: unknown, lo?: number, hi?: number): number;

export const LAYA_DEFAULT_MODEL: string;
export const LAYA_DEFAULT_REVISION: string;
export const LAYA_DEFAULT_SUBFOLDER: string;
export const LAYA_DEFAULT_TEMPERATURES: number[];
export const LAYA_DEFAULT_TEMPERATURES_BY_OPTIONS: Record<string, number>;
export const LAYA_DEFAULT_MAX_LEN: number;
export const LAYA_DEFAULT_HEAD_MAX_LEN: number;
export const LAYA_DEFAULT_MAX_PREFIXES: number;
export const LAYA_QTYPE: Record<QuestionType, number>;
export const LAYA_TEMP_MIN: number;
export const LAYA_TEMP_MAX: number;

// ---------------------------------------------------------------------------
// Strands Decider
// ---------------------------------------------------------------------------

export interface StrandsOptions extends BaseDeciderOptions {
  model?: string;
  revision?: string | null;
  modelFile?: string;
  modelUrl?: string | null;
  variant?: string | null;
  tokenizerModel?: string | null;
  tokenizerRevision?: string | null;
  calibrationModel?: string | null;
  calibrationRevision?: string | null;
  maxLen?: number | null;
  maxQuestionFraction?: number | null;
  temperaturesByKind?: Record<string, number> | null;
  temperature?: number | null;
  ordinalSmoothing?: number | null;
}

export function createStrandsDecider(opts?: StrandsOptions): Promise<Decider>;

export function renderStrandsQuestion(question: NormalizedQuestion): {
  text: string;
  chunks: string[];
  slotLabels: string[];
  slotDescriptions: unknown[];
  kind: string;
  optionSpans: Array<[number, number]>;
};

export function renderStrandsState(state: unknown): string;
export function renderStrandsContent(content: unknown): string;

export function buildStrandsBatch(spec: {
  encode: (text: string, addSpecial?: boolean) => number[];
  stateText: string;
  rendered: unknown[];
  maxLen?: number;
  maxQuestionFraction?: number;
}): {
  items: Array<{ ids: number[]; optIdx: number[]; kind: string }>;
  truncated: boolean;
};

export function collateStrandsItems(items: Array<{ ids: number[]; optIdx: number[] }>): {
  n: number;
  L: number;
  K: number;
  inputIds: BigInt64Array;
  attention: BigInt64Array;
  optionPos: BigInt64Array;
  answerPos: BigInt64Array;
};

export function strandsOptionTokenIndices(lineIds: number[], chunkIds: number[]): number[];

export function strandsAnswersFromLogits(spec: {
  logits: ArrayLike<number>;
  rowWidth: number;
  questions: NormalizedQuestion[];
  rendered: unknown[];
  temperaturesByKind?: Record<string, number>;
  temperature?: number;
  ordinalSmoothing?: number;
}): Answer[];

export function strandsChoiceConfidence(probabilities: number[]): number;
export function strandsScoreConfidence(probabilities: number[], ordinalSmoothing?: number): number;

export interface StrandsCalibration {
  temperature?: number;
  temperaturesByKind: Record<string, number>;
  ordinalSmoothing?: number;
  maxLen?: number;
}

export function parseStrandsCalibration(
  config: Record<string, unknown> | null | undefined,
  fallback?: Partial<StrandsCalibration>,
): StrandsCalibration;

export function loadStrandsCalibration(spec: {
  model: string;
  revision?: string | null;
  files?: string[];
  fetchImpl?: FetchImpl;
}): Promise<{ calibration: StrandsCalibration; source: string | null }>;

export const STRANDS_DEFAULT_MODEL: string;
export const STRANDS_DEFAULT_REVISION: string;
export const STRANDS_DEFAULT_FILE: string;
export const STRANDS_TEMPERATURE: number;
export const STRANDS_TEMPERATURE_BY_KIND: Record<string, number>;
export const STRANDS_ORDINAL_SMOOTHING: number;
export const STRANDS_DEFAULT_MAX_LEN: number;
export const STRANDS_MAX_QUESTION_FRACTION: number;
export const STRANDS_KIND: Record<string, string>;
export const STRANDS_CONFIG_FILES: string[];
export const STRANDS_VARIANTS: Record<string, { model: string; revision: string; description?: string }>;

// ---------------------------------------------------------------------------
// Bekko
// ---------------------------------------------------------------------------

export interface BekkoOptions extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  variant?: string | null;
  subdir?: string;
  manifestUrl?: string | null;
  modelUrl?: string | null;
  tokenizerModel?: string | null;
  tokenizerRevision?: string | null;
  system?: string;
  attentionBudget?: number;
}

export function createBekkoDecider(opts?: BekkoOptions): Promise<Decider>;

export interface BekkoRequest {
  task: string;
  instruction: string;
  state: string;
  layout: string;
  candidates: Array<{ id: string; text: string; value?: number }>;
  labels: string[];
  kind: string;
}

export function renderBekkoContent(content: unknown): string;
export function renderBekkoRequest(question: NormalizedQuestion, stateText: string): BekkoRequest;

export function tokenizeBekkoRequest(
  request: BekkoRequest,
  encode: (text: string) => number[],
  manifest: Record<string, unknown>,
  system?: string,
): { prefixIds: number[]; docIds: number[][]; truncated: boolean };

export function collateBekkoDocs(
  prefixIds: number[],
  docIds: number[][],
  manifest: Record<string, unknown>,
  ort: { Tensor: new (type: string, data: unknown, dims: number[]) => unknown },
): Record<string, unknown>;

export function bekkoSoftmax(logits: number[]): number[];
export function bekkoAnswerFromLogits(spec: {
  logits: ArrayLike<number>;
  task: string;
  request: BekkoRequest;
}): Answer;

export const BEKKO_DEFAULT_MODEL: string;
export const BEKKO_DEFAULT_REVISION: string;
export const BEKKO_DEFAULT_SUBDIR: string;
export const BEKKO_68M_MODEL: string;
export const BEKKO_68M_REVISION: string;
export const BEKKO_400M_MODEL: string;
export const BEKKO_400M_REVISION: string;
export const BEKKO_VARIANTS: Record<string, { model: string; revision: string; bytes?: number; description?: string }>;
export const BEKKO_ATTENTION_BUDGET: number;
export const BEKKO_DEFAULT_YES: string;
export const BEKKO_DEFAULT_NO: string;
export const BEKKO_NOUL_DESCRIPTIONS: Record<"true" | "false", string>;

// ---------------------------------------------------------------------------
// Decision 2.0
// ---------------------------------------------------------------------------

export interface Decision2Options extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  variant?: string | null;
  file?: string;
  configUrl?: string | null;
  maxLength?: number;
}

export function createDecision2Decider(opts?: Decision2Options): Promise<Decider>;

export function renderDecision2Content(content: unknown): string;
export function renderDecision2Options(question: NormalizedQuestion): string[];

export function buildDecision2Segments(
  question: NormalizedQuestion,
  stateText: string,
): { segments: unknown; optionEnds: unknown; keys: string[] };

export function tokenizeDecision2Segments(
  segments: unknown,
  optionEnds: unknown,
  encode: (text: string) => number[],
  maxLength: number,
): { ids: number[]; optionPos: number[]; answerPos: number };

export function decision2Softmax(logits: number[]): number[];
export function decision2AnswerFromLogits(spec: {
  logits: ArrayLike<number>;
  task: string;
  question: NormalizedQuestion;
  keys: string[];
  bias?: number | null;
}): Answer;

export const DECISION2_DEFAULT_MODEL: string;
export const DECISION2_DEFAULT_REVISION: string;
export const DECISION2_DEFAULT_FILE: string;
export const DECISION2_EOS_MODEL: string;
export const DECISION2_EOS_REVISION: string;
export const DECISION2_SOL_MODEL: string;
export const DECISION2_SOL_REVISION: string;
export const DECISION2_VARIANTS: Record<string, { model: string; revision: string; maxLength?: number; description?: string }>;
export const DECISION2_MAX_INPUT_TOKENS: number;
export const DECISION2_NOUL_DESCRIPTIONS: Record<string, string>;

// ---------------------------------------------------------------------------
// Kev
// ---------------------------------------------------------------------------

export interface KevOptions extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  variant?: string | null;
  modelFile?: string | null;
  configUrl?: string | null;
  modelUrl?: string | null;
  dtype?: "auto" | "q4" | "q4f16" | string;
}

export function createKevDecider(opts?: KevOptions): Promise<Decider>;

export function escapeKevText(text: unknown): string;

export function packKevSequence(spec: {
  stateText: string;
  questions: NormalizedQuestion[];
  encode: (text: string) => number[];
  delims: { state: number; question: number; option_start: number; option_end: number; decide: number };
  limits: { max_state_tokens: number; max_branch_tokens: number; max_options: number };
}): { tokens: number[]; groups: number[][]; truncated: boolean };

export const KEV_DEFAULT_MODEL: string;
export const KEV_DEFAULT_REVISION: string;
export const KEV_4B_MODEL: string;
export const KEV_4B_REVISION: string;
export const KEV_VARIANTS: Record<string, { model: string; revision: string; bytes?: number; description?: string }>;

// ---------------------------------------------------------------------------
// Vega
// ---------------------------------------------------------------------------

export interface VegaOptions extends BaseDeciderOptions {
  variant?: "fp16" | "q8" | string | null;
  format?: "fused" | "split" | null;
  model?: string;
  revision?: string;
  subfolder?: string | null;
  torsoFile?: string;
  engineFile?: string;
  embedFile?: string;
  decoderFile?: string;
  calibrationModel?: string | null;
  calibrationRevision?: string | null;
  maxLen?: number | null;
  alpha?: string | number;
  risk?: string | number | null;
}

export function createVegaDecider(opts?: VegaOptions): Promise<Decider>;

export function vegaQuestionDef(question: NormalizedQuestion): { t: string; ins: unknown; crit: unknown };
export function vegaOptionTexts(qdef: { t: string; crit: unknown }): string[];

export function buildVegaPrompt(spec: {
  encode: (text: string, addSpecial?: boolean) => number[];
  stateText: string;
  qdef: { t: string; ins: unknown; crit: unknown };
  maxLen: number;
  bosId?: number | null;
  queryFirstMaxState?: number;
  optMaxTokens?: number;
  insMaxTokens?: number;
  truncateState?: boolean;
}): {
  ids: number[];
  spans: {
    state: [number, number];
    question: [number, number];
    options: Array<[number, number]>;
    last: number;
  };
  truncated: boolean;
};

export function packVegaBatch(
  items: Array<{ ids: number[]; spans: unknown; qtype: number }>,
  opts?: { padId?: number; chunks?: number },
): {
  inputIds: BigInt64Array;
  attention: BigInt64Array;
  pool: Float32Array;
  poolDims: [number, number, number];
  n: number;
  L: number;
  K: number;
  qtypes: number[];
  optMask: Uint8Array;
  nseg: number;
};

export function packVegaChunks(
  pool: Float32Array,
  items: unknown[],
  L: number,
  opts?: { chunks?: number; chunkMinTokens?: number; chunkTokens?: number; nseg?: number },
): Uint8Array;

export function vegaThermalFeatures(spec: {
  kinetic: ArrayLike<number>;
  nearest: ArrayLike<number>;
  stepsUsed: ArrayLike<number>;
  nOpts: number[];
  maxSteps: number;
}): number[][];

export function vegaThermalProbabilities(spec: {
  logits: ArrayLike<number>;
  rowWidth: number;
  qtypes: number[];
  phys: ArrayLike<number>;
  nOpts: number[];
  theta?: number[][];
  thetaByRow?: number[][][] | null;
}): number[][];

export function vegaAnswers(spec: {
  probs: number[][];
  qtypes: number[];
  physics: { nearest: ArrayLike<number>; stepsUsed: ArrayLike<number> };
  rawLogits: ArrayLike<number>;
  rowWidth: number;
  calibration: Record<string, unknown>;
  questions?: NormalizedQuestion[];
  labels: Array<Array<string | number>>;
  alpha?: string;
  risk?: string;
  calibrations?: Array<Record<string, unknown> | null> | null;
  adapterNames?: string[] | null;
  gates?: number[][] | null;
  routes?: number[] | null;
}): Answer[];

export function vegaOptionLabels(questions: NormalizedQuestion[]): Array<Array<string | number>>;
export function renderVegaText(content: unknown): string;
export function vegaJsonDumps(value: unknown, ensureAscii?: boolean): string;
export function vegaRoute(gates: number[] | null | undefined, threshold?: number): number;
export function vegaTextPositions(L: number): BigInt64Array;
export function emptyVegaPast(
  session: { inputMetadata?: Array<{ name: string; shape?: unknown[]; isTensor?: boolean }> } | null,
  Tensor: new (type: string, data: unknown, dims: number[]) => unknown,
): Record<string, unknown>;

export function poolVegaHidden(spec: {
  h13: ArrayLike<number>;
  h19: ArrayLike<number>;
  spans: { state: [number, number]; question: [number, number]; options: Array<[number, number]> };
  d: number;
  chunks?: number;
  chunkMinTokens?: number;
  chunkTokens?: number;
  withChunks?: boolean;
}): { feats: Float32Array; opts: Float32Array; chunks: Float32Array; cmask: Uint8Array };

export const VEGA_DEFAULT_MODEL: string;
export const VEGA_DEFAULT_REVISION: string;
export const VEGA_TORSO_FILE: string;
export const VEGA_ENGINE_FILE: string;
export const VEGA_Q8_DEFAULT_MODEL: string;
export const VEGA_Q8_DEFAULT_REVISION: string;
export const VEGA_Q8_DEFAULT_SUBFOLDER: string;
export const VEGA_Q8_EMBED_FILE: string;
export const VEGA_Q8_DECODER_FILE: string;
export const VEGA_Q8_ENGINE_FILE: string;
export const VEGA_VARIANTS: Record<
  string,
  { model: string; revision: string; format?: string; subfolder?: string; description?: string }
>;
export const VEGA_QTYPE: Record<QuestionType | string, number>;
export const VEGA_QTYPE_NAMES: string[];
export const VEGA_DEFAULTS: {
  optMaxTokens: number;
  insMaxTokens: number;
  memoryChunks: number;
  memoryMinTokens: number;
  memoryChunkTokens: number;
  prefixMinTokens: number;
  queryFirstMaxState: number;
  maxLen: number;
  nSteps: number;
  risk: string;
  alpha: string;
};
export const VEGA_DEFAULT_THETA: number[][];

// ---------------------------------------------------------------------------
// clef
// ---------------------------------------------------------------------------

export interface ClefOptions extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  file?: string;
  tokenizerModel?: string | null;
  tokenizerRevision?: string | null;
  configUrl?: string | null;
  maxLen?: number | null;
  maxStateTokens?: number | null;
}

export function createClefDecider(opts?: ClefOptions): Promise<Decider>;

export function encodeClefRecord(spec: {
  encode: (text: string) => number[];
  stateText: string;
  questions: NormalizedQuestion[];
  maxLength?: number;
  maxStateTokens?: number | null;
}): {
  ids: number[];
  questions: Array<{
    id: string;
    type: number;
    instructions: unknown;
    optionIds: string[];
    questionSpan: [number, number];
    optionSpans: Array<[number, number]>;
  }>;
  length: number;
  truncated: boolean;
};

export function packClefBatch(record: {
  ids: number[];
  questions: Array<{ optionIds: string[]; questionSpan: [number, number]; optionSpans: Array<[number, number]>; type: number }>;
}): {
  Q: number;
  K: number;
  S: number;
  L: number;
  poolGlobal: Float32Array;
  poolQuestion: Float32Array;
  poolOption: Float32Array;
  slotQuestion: number[];
  typeIds: number[];
  optMask: Uint8Array;
  counts: number[];
};

export function clefQuestionOptions(question: NormalizedQuestion): { keys: string[]; descriptions: unknown[] };
export function clefAnswers(spec: { probs: number[][]; questions: NormalizedQuestion[] }): Answer[];
export function renderClefContent(content: unknown): string;

export const CLEF_DEFAULT_MODEL: string;
export const CLEF_DEFAULT_REVISION: string;
export const CLEF_DEFAULT_FILE: string;
export const CLEF_DEFAULT_MAX_LEN: number;
export const CLEF_SYSTEM_PROMPT: string;
export const CLEF_QTYPE: Record<QuestionType, number>;
export const CLEF_QTYPE_NAMES: string[];
export const CLEF_NOUL_DESCRIPTIONS: Record<"true" | "false", string>;
export const CLEF_DEFAULTS: { maxLen: number; maxStateTokens: number | null };

// ---------------------------------------------------------------------------
// Decision 3.0
// ---------------------------------------------------------------------------

export interface Decision3Options extends BaseDeciderOptions {
  model?: string;
  revision?: string;
  variant?: "edge" | "lite" | "nano" | string | null;
  torsoFile?: string;
  readoutFile?: string;
  tokenizerModel?: string | null;
  tokenizerRevision?: string | null;
  configUrl?: string | null;
  codes?: string[] | null;
  maxLen?: number | null;
  batchSize?: number;
  temperature?: number | null;
}

export function createDecision3Decider(opts?: Decision3Options): Promise<Decider>;

export function buildDecision3Prompt(state: unknown, rendered: unknown, codes: string[]): string;
export function decision3Messages(state: unknown, rendered: unknown, codes: string[]): Array<{ role: string; content: string }>;
export function renderDecision3Chat(system: string, user: string): string;

export function normalizeDecision3Question(question: NormalizedQuestion): {
  kind: string;
  rendered: { type: string; instructions: unknown; criteria?: Record<string, unknown> };
  keys: string[];
  descriptions: unknown[];
};

export function decision3OptionTexts(rendered: { type: string; criteria?: Record<string, unknown> }): string[];
export function decision3CandidateCodes(): string[];

export function decision3ProbsFromLogits(spec: {
  logits: ArrayLike<number>;
  rowWidth: number;
  counts: number[];
  temperature?: number;
}): number[][];

export function decision3Confidence(values: number[]): number;
export function decision3Answers(spec: { probs: number[][]; questions: NormalizedQuestion[] }): Answer[];

export function packDecision3Batch(spec: {
  sequences: number[][];
  counts: number[];
  padId: number;
}): {
  B: number;
  width: number;
  inputIds: BigInt64Array;
  attention: BigInt64Array;
  counts: number[];
};

export function renderDecision3Json(value: unknown): string;
export function renderDecision3Canonical(value: unknown): string;
export function describeDecision3Value(value: unknown): string;

export const DECISION3_DEFAULT_MODEL: string;
export const DECISION3_DEFAULT_REVISION: string;
export const DECISION3_TORSO_FILE: string;
export const DECISION3_READOUT_FILE: string;
export const DECISION3_DEFAULT_BATCH_SIZE: number;
export const DECISION3_DEFAULT_MAX_LEN: number;
export const DECISION3_MAX_OPTIONS: number;
export const DECISION3_SCORE_LEVELS: [number, number];
export const DECISION3_SYSTEM_PROMPT: string;
export const DECISION3_NOUL_DESCRIPTIONS: string[];
export const DECISION3_DEFAULT_INSTRUCTIONS: string;
export const DECISION3_DEFAULTS: { maxLen: number; batchSize: number; temperature: number };
export const DECISION3_VARIANTS: Record<
  string,
  { model: string; revision: string; maxLength?: number; description?: string }
>;
export const DECISION3_DEFAULT_CODES: string[];
