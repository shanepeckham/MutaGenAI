# Benchmark Candidates for MutaGenAI Cookbook

## Research Topics

1. Evaluate 13 agentic AI benchmarks for suitability as a new MutaGenAI cookbook example
2. Determine top 3 recommendations ranked by integration ease, dataset availability, and differentiation from existing examples

## Existing Cookbook Coverage

| Benchmark | Task Domain | Key File |
|---|---|---|
| BFCL V4 | Function calling (simple, multiple, parallel) | prompt_evolution_bfcl.py |
| xLAM 60k | Function calling (60k examples, 3673 APIs) | prompt_evolution_xlam.py |
| API-Bank | Multi-level tool use (call, retrieval, planning) | prompt_evolution_apibank.py |
| ToolBench | REST API tool use (G1/G2/G3 tiers) | prompt_evolution_toolbench.py |
| τ-bench | Conversational customer-service agents | prompt_evolution_tau_bench.py |
| Browser Agent | Multi-turn browser tool calling (tickets, forms) | prompt_evolution_browser_agent.py |
| BrowserGym/WebLINX | Web action prediction (click, fill, navigate) | browsergym/prompt_evolution_browsergym.py |
| Entity Classification | 6-class agentic entity classification | prompt_evolution_entity_classification.py |
| AssetOpsBench | DAG planning for industrial asset ops | prompt_evolution_assetops.py |
| Agentic (synthetic) | Customer support, code-assist, data-pipeline | prompt_evolution_agentic.py |
| No-eval strategies | 7 label-free scoring strategies | prompt_evolution_no_eval.py |
| GSM8K | Math reasoning (log exists, no cookbook script) | logs/gsm8k_run1_experiment_log.json |

**Integration pattern**: Each cookbook loads data → defines `score_*(response, ground_truth) → float[0,1]` → wraps in PromptEvolver loop → logs JSON.

## Benchmark Evaluations

### 1. GSM8K — Grade School Math 8K

- **Task**: Solve grade-school math word problems with chain-of-thought reasoning. Output format: step-by-step reasoning followed by `#### <number>`.
- **Default system prompt**: Yes — clear and well-defined: "You are a math problem solver. Given a grade-school math word problem, solve it step by step and provide the final numerical answer..." (already exists in `logs/gsm8k_run1_experiment_log.json`)
- **Dataset**: Publicly available on HuggingFace: `openai/gsm8k`. Train split (7,473 examples) + test split (1,319 examples). Each has `question` and `answer` fields with `#### <number>` final answer format.
- **Scoring**: Exact match accuracy on the extracted numerical answer. Very clean — parse `####` line, compare to ground truth number.
- **Integration ease**: **Very high**. Single LLM call per sample. Clear ground truth extraction. Already has an experiment log proving the concept works.
- **Differentiation**: Tests **reasoning/math** — fundamentally different from all existing examples which focus on tool calling, function calling, or classification. Chain-of-thought prompt optimization is a compelling use case.
- **Status**: Log exists but **no cookbook script** — this is a gap that could be filled.
- **Verdict**: ⭐⭐⭐⭐⭐ Excellent candidate — simplest integration, proven concept, unique task domain.

### 2. HumanEval / MBPP — Code Generation

- **Task**: Generate Python functions from docstring specifications (HumanEval: 164 problems) or from natural language descriptions (MBPP: 974 problems).
- **Default system prompt**: Yes — natural starting point: "You are a Python programmer. Given a function signature and docstring, complete the implementation. Return only the function body."
- **Dataset**: HumanEval available at `openai/openai_humaneval` on HuggingFace. MBPP at `google-research-datasets/mbpp`. Both have prompt + canonical solution + test cases.
- **Scoring**: pass@k metric — execute generated code against unit tests. Binary pass/fail per problem.
- **Integration ease**: **Medium-High**. Requires code execution (sandboxed `exec()` with test assertions). Single LLM call per problem, but scoring requires running Python code. The MutaGenAI pattern of `score_*(response, ground_truth) → float` maps well — just needs a subprocess/exec wrapper.
- **Differentiation**: Tests **code generation** — a completely new domain. All existing examples test tool/function calling or classification, not code synthesis.
- **Considerations**: Code execution adds complexity (security sandboxing). MBPP is easier to integrate (simpler test assertions). HumanEval is the more recognized benchmark.
- **Verdict**: ⭐⭐⭐⭐ Strong candidate — high differentiation, moderate integration complexity.

### 3. MINT (Multi-turn Interactive)

- **Task**: Multi-turn agent interactions requiring tool use across multiple exchanges. Agent must reason, use tools (code interpreter, web browser), and refine responses.
- **Default system prompt**: Partially defined — the framework provides agent scaffolding but the system prompt is embedded in a multi-turn harness.
- **Dataset**: Available on GitHub (xingyaoww/mint-bench). Contains tasks from GSM8K, MATH, HumanEval, MBPP, TheoremQA (with tool augmentation).
- **Scoring**: Success rate (task completion accuracy).
- **Integration ease**: **Low**. Multi-turn interaction loop is fundamentally different from MutaGenAI's single-call scoring pattern. Would need a conversation simulator wrapper.
- **Differentiation**: Multi-turn interaction, but the underlying tasks overlap with GSM8K and HumanEval.
- **Verdict**: ⭐⭐ Poor fit — multi-turn architecture doesn't map to the single-call evolutionary loop.

### 4. AgentBench — Comprehensive Agent Evaluation

- **Task**: Evaluate agents across 8 environments: OS shell, database, knowledge graph, digital card game, lateral thinking, house-holding (ALFWorld), web shopping, web browsing.
- **Default system prompt**: Each environment has its own system prompt template, but they are tightly coupled to interactive environment simulators.
- **Dataset**: Available on GitHub (THUDM/AgentBench). Requires environment setup (Docker containers, simulators).
- **Scoring**: Per-environment success metrics (task completion rate, F1, etc.).
- **Integration ease**: **Very low**. Each environment requires its own simulator/runtime. Cannot score with a single LLM call — requires interactive stateful execution.
- **Differentiation**: Very broad but impractical for prompt evolution.
- **Verdict**: ⭐ Poor fit — requires live environment simulators, cannot be reduced to single-call scoring.

### 5. ToolBench v2

- **Task**: Same domain as existing ToolBench coverage (REST API tool use).
- **Differentiation**: Already covered by `prompt_evolution_toolbench.py`.
- **Verdict**: ❌ Already covered — skip.

### 6. SWE-bench Lite — Software Engineering

- **Task**: Given a GitHub issue, apply a code patch to a real repository. SWE-bench Lite has 300 curated problems from 12 Python repos.
- **Default system prompt**: No standard starting prompt — the task framing varies by agent framework (SWE-Agent, AutoCodeRover, etc.).
- **Dataset**: Available on HuggingFace: `princeton-nlp/SWE-bench_Lite`. Contains repo, issue text, gold patch, and test commands.
- **Scoring**: % of issues resolved (test suite passes after applying the patch).
- **Integration ease**: **Very low**. Requires cloning Git repos, applying patches, running test suites. Each evaluation takes minutes. Fundamentally a multi-step process with environment setup.
- **Differentiation**: Unique task (real-world software engineering) but impractical for evolutionary loop.
- **Verdict**: ⭐ Poor fit — evaluation is too heavyweight for an evolutionary fitness function.

### 7. GAIA — General AI Assistants

- **Task**: Questions requiring multi-step reasoning with tools (web search, file reading, code execution). 3 difficulty levels (L1: 1-step, L2: 5+ steps, L3: requires domain expertise).
- **Default system prompt**: No standardized system prompt — varies by framework.
- **Dataset**: Available on HuggingFace: `gaia-benchmark/GAIA`. Validation set has 165 questions with annotated answers and steps.
- **Scoring**: Exact-match accuracy on final answer. Very clean — compare extracted answer to ground truth string/number.
- **Integration ease**: **Low-Medium**. L1 questions could work as single-call reasoning, but most require tool access (web, files). Scoring is simple (exact match) but the tasks need real tool access.
- **Differentiation**: General reasoning + tool use, but tool requirements make it hard to run offline.
- **Verdict**: ⭐⭐ Moderate — L1 subset could work but limited without real tools.

### 8. Nexus Function Calling (NexusRaven)

- **Task**: Single-turn function calling — given a natural language query and function signatures, produce the correct function call with arguments.
- **Default system prompt**: Yes — the NexusRaven model uses a clear prompt template: "Function: ... User Query: ... " with a system instruction for function calling.
- **Dataset**: Available on HuggingFace: `Nexusflow/NexusRaven_API_evaluation`. Contains user queries, function definitions, and ground-truth function calls.
- **Scoring**: Function name match accuracy + argument correctness.
- **Integration ease**: **Very high**. Single LLM call, clear ground truth, simple parsing. Almost identical to BFCL/xLAM pattern.
- **Differentiation**: **Low** — very similar to BFCL and xLAM which are already covered. Function calling with argument matching is well-represented.
- **Verdict**: ⭐⭐⭐ Easy to integrate but too similar to existing function-calling examples.

### 9. HotpotQA — Multi-hop Question Answering

- **Task**: Answer questions that require reasoning across 2+ Wikipedia paragraphs. Models must find and combine information from multiple sources.
- **Default system prompt**: Yes — clear framing: "You are a question-answering assistant. Answer the following question using step-by-step reasoning. Provide a short, direct answer."
- **Dataset**: Publicly available on HuggingFace: `hotpotqa/hotpot_qa`. Full dataset: 113K train, 7.4K dev. Distractor setting provides 10 paragraphs (2 gold + 8 distractors) with supporting facts annotated.
- **Scoring**: Exact match (EM) and F1 on answer span. Clean binary/continuous scoring.
- **Integration ease**: **Very high**. Single LLM call — provide context paragraphs + question → extract answer → compare to ground truth. EM gives binary score, F1 gives continuous score. No external tools needed.
- **Differentiation**: Tests **multi-hop reasoning over provided context** — completely different from tool calling. The prompt evolution angle is compelling: can you evolve a system prompt that makes the model better at combining information from multiple paragraphs?
- **Considerations**: The "distractor" setting is ideal — 10 paragraphs with 2 gold, so the model must figure out which are relevant. This is a reasoning + information extraction task that benefits strongly from prompt engineering.
- **Verdict**: ⭐⭐⭐⭐⭐ Excellent candidate — simple integration, unique reasoning domain, clean metrics.

### 10. ALFWorld — Text-based Household Agent

- **Task**: Complete household tasks (clean, heat, put, examine, cool, look) in a text-based interactive environment derived from ALFRED.
- **Default system prompt**: ReAct-style prompts are common: "Interact with a household to solve a task. You can take actions..."
- **Dataset**: Available on GitHub (alfworld/alfworld). Requires installing the ALFWorld environment.
- **Scoring**: Task success rate (binary: completed or not).
- **Integration ease**: **Low**. Requires interactive text game environment. Multi-step action sequence — not a single LLM call. Each task involves 10-50+ interaction turns.
- **Differentiation**: Embodied agent reasoning, but the interactive requirement makes it unsuitable.
- **Verdict**: ⭐⭐ Poor fit — requires multi-turn interactive environment simulation.

### 11. WebArena — Web Navigation

- **Task**: Complete realistic web tasks on self-hosted web applications (shopping, forums, GitLab, maps, Wikipedia).
- **Default system prompt**: Environment-specific system prompts with action space definitions.
- **Dataset**: 812 tasks across 5 web environments. Requires self-hosting the web applications.
- **Scoring**: Task completion rate (functional correctness).
- **Integration ease**: **Very low**. Requires self-hosted web infrastructure. Multi-step browser interactions. Too heavyweight.
- **Differentiation**: Web navigation is partially covered by BrowserGym/WebLINX.
- **Verdict**: ⭐ Poor fit — requires infrastructure, partially overlaps with existing coverage.

### 12. ToolACE — Tool-Augmented Agent

- **Task**: Function/tool calling with a focus on data diversity — 26k samples generated using a synthetic pipeline with self-evolution.
- **Default system prompt**: The dataset includes system prompts as part of the conversation format.
- **Dataset**: Available on HuggingFace: `Team-ACE/ToolACE`. Contains function definitions, queries, and expected tool calls.
- **Scoring**: Function name match + argument accuracy (similar to BFCL).
- **Integration ease**: **Very high**. Same pattern as BFCL/xLAM.
- **Differentiation**: **Low** — another function calling benchmark. Too similar to existing coverage.
- **Verdict**: ⭐⭐⭐ Easy to integrate but redundant with BFCL, xLAM, API-Bank.

### 13. MetaTool — Tool Selection

- **Task**: Given a user query, select the correct tool from a large toolset. Focuses on tool awareness (knowing when to use a tool) and tool selection (choosing the right one).
- **Default system prompt**: Yes — straightforward: "You are an AI assistant with access to the following tools. Select the most appropriate tool for the user's request."
- **Dataset**: Available on GitHub (HowieHwong/MetaTool). Contains tool descriptions + queries + expected tool selections.
- **Scoring**: Tool selection accuracy (exact match on tool name).
- **Integration ease**: **High**. Single LLM call — provide tools + query → extract selected tool → compare. Very simple scoring.
- **Differentiation**: **Low-Medium** — tool selection is a subset of what BFCL and xLAM already test. API-Bank Level 2 specifically tests API retrieval + selection.
- **Verdict**: ⭐⭐⭐ Easy but overlaps with existing tool-calling coverage.

## Top 3 Recommendations

### Rank 1: HotpotQA — Multi-hop Reasoning

**Why it's the best choice:**
- **Completely new task domain**: Multi-hop reasoning over provided context is fundamentally different from every existing example (all are tool calling, function calling, or classification).
- **Trivial integration**: Single LLM call. Provide paragraphs + question → extract answer → EM/F1 scoring. No external dependencies, no tool access, no code execution.
- **Large, clean, public dataset**: `hotpotqa/hotpot_qa` on HuggingFace, 113K+ examples. The "distractor" setting (10 paragraphs, 2 gold) is ideal.
- **Clear default prompt**: "Answer the following question based on the provided context. Think step by step, identify the relevant paragraphs, and provide a short, direct answer."
- **Compelling evolution story**: Evolving prompts that make models better at multi-hop reasoning is a high-impact use case. The system prompt can guide: how to identify relevant paragraphs, how to chain evidence, how to format the final answer.
- **Scoring**: F1 gives continuous fitness signal (better than binary EM for evolution). HuggingFace `evaluate` library has built-in F1/EM metrics for QA.

**Starting prompt:**
```
You are a question-answering assistant. You will be given several context
paragraphs and a question. Some paragraphs are relevant, others are
distractors.

Instructions:
- Read all paragraphs carefully
- Identify the paragraphs relevant to the question
- Reason step by step, combining information from multiple paragraphs
- Provide a short, direct answer (a few words, not a full sentence)

Answer:
```

**Dataset loading:**
```python
from datasets import load_dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
# ds[i]["question"], ds[i]["context"], ds[i]["answer"]
```

### Rank 2: HumanEval (Code Generation)

**Why it's a strong choice:**
- **New task domain**: Code generation/synthesis — no existing example covers this.
- **Recognized benchmark**: Industry-standard for measuring code generation capability.
- **Clear default prompt**: "You are a Python programmer. Complete the function implementation based on the docstring. Return only the function body, no explanation."
- **Clean scoring**: pass@1 — execute generated code against unit tests. Binary per problem but averages to a continuous metric over the dataset.
- **Public dataset**: `openai/openai_humaneval` on HuggingFace. 164 problems with test cases.

**Integration complexity**: Medium. Requires code execution, which adds:
- Sandboxed `exec()` with timeout
- Test case extraction and execution
- Security considerations (though the test cases are well-known and safe)

**Starting prompt:**
```
You are an expert Python programmer. Given a function signature and
docstring, write the complete function implementation.

Rules:
- Return ONLY the function body (no signature, no docstring)
- Use standard Python libraries only
- Handle edge cases (empty input, None, etc.)
- Write clean, efficient code
```

**Dataset loading:**
```python
from datasets import load_dataset
ds = load_dataset("openai/openai_humaneval", split="test")
# ds[i]["prompt"], ds[i]["canonical_solution"], ds[i]["test"]
```

### Rank 3: GSM8K (Math Reasoning) — Complete the Missing Cookbook

**Why it ranks third:**
- **Proven concept**: An experiment log already exists showing the pipeline works (93% evolved vs 95% default on gpt-4.1).
- **Simplest possible integration**: Single LLM call, `####` number extraction, exact match scoring.
- **Public dataset**: `openai/gsm8k` on HuggingFace.
- **Unique domain**: Math reasoning with chain-of-thought.

**Why not higher:**
- The existing log shows the evolved prompt actually performed *slightly worse* (-2%) than the default on gpt-4.1. This suggests GPT-4.1 is already near-ceiling on GSM8K. For a compelling cookbook example, you'd want to show improvement (possibly with a smaller model like Ollama llama3.2, where there's more room to optimize).
- Less "agentic" than the user's stated preference — it's pure reasoning, not tool use.

**Starting prompt** (from existing log):
```
You are a math problem solver. Given a grade-school math word
problem, solve it step by step and provide the final numerical
answer.

Rules:
- Show your reasoning step by step.
- After your reasoning, write the final answer on its own line
  in the format: #### <number>
- The final answer must be a single number (integer or decimal).
- Do NOT include units, dollar signs, or commas in the final answer.
- Example final line: #### 42
```

**Dataset loading:**
```python
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
# ds[i]["question"], ds[i]["answer"] (answer has #### <number> at end)
```

## Follow-on Questions

1. Should the cookbook example target Ollama (local, small model) or Azure OpenAI (cloud, large model)? Smaller models have more room for prompt improvement.
2. Is there a preference for examples that demonstrate improvement over the default (GSM8K showed regression on gpt-4.1)?
3. For HumanEval, is sandboxed code execution acceptable complexity, or should we prefer a benchmark that requires no execution at all?

## Key Discoveries

- **HotpotQA is the strongest candidate overall** — it fills a genuine gap (reasoning over context), has trivial integration, massive public dataset, and continuous F1 scoring ideal for evolutionary search.
- **GSM8K has a proven pipeline but no cookbook script** — filling this gap is low-hanging fruit.
- **Multi-turn/interactive benchmarks (MINT, AgentBench, ALFWorld, WebArena, SWE-bench) are poor fits** — MutaGenAI's evolutionary loop scores candidates with a single-call fitness function; multi-turn interaction requires fundamentally different architecture.
- **Function-calling benchmarks (Nexus, ToolACE, MetaTool) are redundant** — BFCL, xLAM, API-Bank, and ToolBench already provide thorough coverage.
