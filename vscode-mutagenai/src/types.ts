/** Shared types for the MutaGenAI VS Code extension. */

/** Events emitted by the Python sidecar over stdout (JSON lines). */
export type SidecarEvent =
  | StartedEvent
  | StatusEvent
  | LogEvent
  | SeedEvent
  | SeedCompleteEvent
  | GenerationEvent
  | StoppedEvent
  | DoneEvent
  | ErrorEvent;

export interface StartedEvent {
  type: "started";
  config: EvolutionConfig;
}

export interface StatusEvent {
  type: "status";
  message: string;
}

export interface LogEvent {
  type: "log";
  level: "info" | "warn" | "error";
  message: string;
}

export interface SeedEvent {
  type: "seed";
  index: number;
  total: number;
  hash: string;
  score: number;
  island: number;
  operation: string;
  template: string;
}

export interface SeedCompleteEvent {
  type: "seedComplete";
  bestScore: number;
  bestHash: string;
  candidateCount: number;
}

export interface CandidateInfo {
  hash: string;
  parentHashes: string[];
  score: number;
  island: number;
  operation: string;
  template: string;
  temperature: number;
  topP: number;
}

export interface GenerationEvent {
  type: "generation";
  generation: number;
  totalGenerations: number;
  bestScore: number;
  bestHash: string;
  genBestScore: number;
  improved: boolean;
  candidateCount: number;
  elapsed: number;
  candidates: CandidateInfo[];
  migrated: boolean;
  noImprovementCount: number;
}

export interface StoppedEvent {
  type: "stopped";
  generation: number;
  reason: "user" | "threshold" | "patience";
  message?: string;
}

export interface LineageNode {
  hash: string;
  parentHashes: string[];
  operation: string;
  generation: number;
  island: number;
  score: number;
  temperature: number;
  topP: number;
  template: string;
}

export interface DoneEvent {
  type: "done";
  bestPrompt: string;
  bestScore: number;
  bestTemperature: number;
  bestTopP: number;
  wallTime: number;
  totalCandidates: number;
  iterationsRun: number;
  lineage: LineageNode[];
}

export interface ErrorEvent {
  type: "error";
  message: string;
  traceback?: string;
}

/** Configuration sent to the sidecar. */
export interface EvolutionConfig {
  backend: string;
  model: string;
  iterations: number;
  populationSize: number;
  numIslands: number;
  eliteSize: number;
  mutationRate: number;
  crossoverRate: number;
  earlyStopThreshold: number;
  earlyStopPatience: number;
  seed: number;
  seeds?: string[];
  seedTemplate?: string;
  seedTemplatesDir?: string;
  tools?: ToolDef[];
  evalDataset?: EvalSampleDef[];
  promptText?: string;
  adaptiveMutations?: boolean;
  llmMutationRate?: number;
  refineAfterSplice?: boolean;
  /** "noeval" for wizard/no-eval mode, undefined for ground-truth. */
  mode?: string;
  taskDescription?: string;
  testInputs?: string[];
  problemType?: string;
  rubric?: string;
  mutations?: string[];
  strategies?: string[];
  proxyChecks?: Array<{ name: string; condition: string; weight: number }>;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: Record<string, string>;
}

export interface EvalSampleDef {
  query: string;
  expected_tool: string;
  expected_params: Record<string, string>;
}
