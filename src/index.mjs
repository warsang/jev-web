// jev-web — public API (typed decisions in the browser).
export { createDecider } from "./session.mjs";
export {
  registerDecisionFamily, createDecisionRuntime,
  listDecisionFamilies, getDecisionFamily,
} from "./registry.mjs";
export { normalizeQuestions, QUESTION_TYPES, NOUL_OPTIONS } from "./questions.mjs";
export { buildDecisionInput, typedMarkerIds, DecisionInputTooLongError } from "./encode.mjs";
export { answersFromScores, softmaxWithTemperature } from "./answers.mjs";
export {
  createLayaDecider,
  buildLayaSequence,
  collateLayaItems,
  layaAnswersFromLogits,
  layaTokenizerAdapter,
  renderLayaOptions,
  layaTempBucket,
  clampTemperature,
  LAYA_DEFAULT_MODEL,
  LAYA_DEFAULT_REVISION,
  LAYA_DEFAULT_SUBFOLDER,
  LAYA_DEFAULT_TEMPERATURES,
  LAYA_DEFAULT_TEMPERATURES_BY_OPTIONS,
  LAYA_DEFAULT_MAX_LEN,
  LAYA_DEFAULT_HEAD_MAX_LEN,
  LAYA_DEFAULT_MAX_PREFIXES,
  LAYA_QTYPE,
  LAYA_TEMP_MIN,
  LAYA_TEMP_MAX,
} from "./laya.mjs";
export {
  DEFAULT_MODEL,
  DEFAULT_REVISION,
  DEFAULT_TEMPERATURE,
  DEFAULT_MARKERS,
  DEFAULT_MAX_STATE_TOKENS,
  DEFAULT_MAX_LEN,
  resolveDevice,
  resolveDtype,
  detectWebGPU,
  loadModelConfig,
} from "./config.mjs";
