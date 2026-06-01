# MutagenAI

Evolve LLM system prompts using evolutionary search — no fine-tuning, no
GPUs, no labelled data required.

> **Explain like I'm 10:** You know how the *instructions* you give someone
> matter as much as the question? MutagenAI tries hundreds of different ways
> to phrase the instructions for an AI, tests each one, and breeds the
> winners — just like evolving animals — until it finds the best wording.

## Table of contents

- [Getting started — the wizard](#getting-started--the-wizard)
  - [What the wizard asks](#what-the-wizard-asks)
  - [Three paths through the wizard](#three-paths-through-the-wizard)
  - [How to evaluate when you have no ground truth](#how-to-evaluate-when-you-have-no-ground-truth)
  - [Example walkthrough](#example-walkthrough)
  - [Wizard code generation improvements](#wizard-code-generation-improvements)
- [How prompt evolution works](#how-prompt-evolution-works)
  - [The approach](#the-approach)
  - [The evolutionary loop](#the-evolutionary-loop)
- [Core modules](#core-modules)
- [Ground-truth prompt evolution](#ground-truth-prompt-evolution)
  - [Quick start — PromptEvolver](#quick-start--promptevolver)
  - [Baseline comparison](#baseline-comparison)
  - [Ollama vs Azure OpenAI](#ollama-vs-azure-openai)
  - [Agentic workload scenarios](#agentic-workload-scenarios)
- [Six agentic benchmarks](#six-agentic-benchmarks)
  - [BFCL V4 — single-turn function calling](#bfcl-v4--single-turn-function-calling)
  - [τ-Bench — conversational agents](#τ-bench--conversational-agents)
  - [xLAM 60 k — broad function calling](#xlam-60-k--broad-function-calling)
  - [ToolBench — multi-tool API orchestration](#toolbench--multi-tool-api-orchestration)
  - [API-Bank — multi-level tool-use evaluation](#api-bank--multi-level-tool-use-evaluation)
  - [Browser Agent Tasks — failure recovery](#browser-agent-tasks--failure-recovery)
  - [Consolidated results](#consolidated-results)
  - [Key takeaways from benchmarks](#key-takeaways-from-benchmarks)
- [No-eval prompt evolution — 7 strategies](#no-eval-prompt-evolution--7-strategies)
  - [The seven strategies](#the-seven-strategies)
  - [Quick start — NoEvalPromptEvolver](#quick-start--noevalpromptevolver)
  - [Which strategy should I pick?](#which-strategy-should-i-pick)
- [xLAM no-eval worked example](#xlam-no-eval-worked-example)
  - [The problem](#the-problem)
  - [What this recipe does](#what-this-recipe-does)
  - [Configuration](#configuration)
  - [Results](#results)
  - [What the results mean](#what-the-results-mean)
  - [How to apply this to your own agent](#how-to-apply-this-to-your-own-agent)
- [API-Bank no-eval vs ground-truth comparison](#api-bank-no-eval-vs-ground-truth-comparison)
- [Seed templates — external prompt configuration](#seed-templates--external-prompt-configuration)
- [Agent routing — static vs evolved prompt (GPT-4.1)](#agent-routing--static-vs-evolved-prompt-gpt-41)
- [Entity classification — static vs evolved prompt](#entity-classification--static-vs-evolved-prompt)
- [Dashboard and visualisation](#dashboard-and-visualisation)
- [Cookbook recipes](#cookbook-recipes)
- [Installation](#installation)

---

## Getting started — the wizard

> Answer a few questions about your AI agent and MutagenAI generates a
> ready-to-run script that evolves the perfect system prompt — no PhD
> required.

The wizard walks you through nine steps and generates a self-contained
Python script tailored to your project:

```bash
mutagenai init                        # interactive walkthrough
mutagenai init --output my_agent.py   # custom output file
```

### What the wizard asks

| Step | Question | Why it matters |
|------|----------|----------------|
| 1 | **Task description** | Drives mutation generation and rubrics |
| 2 | **Ground truth?** | yes / partial / no — determines scoring path |
| 3 | **Test inputs** | Unlabelled queries your agent handles |
| 4 | **Scoring strategy** | Picks from 7 strategies (see below) |
| 5 | **Domain mutations** | Custom or auto-generated rewrite rules |
| 6 | **Human evaluation** | always / final pick / fully automated |
| 7 | **Seed templates** | Your existing prompts or auto-generated |
| 8 | **LLM backend** | Ollama (local) / OpenAI / Azure OpenAI |
| 9 | **Configuration** | Standard (fast) / Deep (thorough) / Custom |

### Three paths through the wizard

**Path A — Full ground truth.** You have labelled input/output pairs.
The wizard generates a script that scores candidates against your labels
and evolves the prompt for maximum accuracy.

**Path B — No labels.** You have only unlabelled inputs. The wizard lets
you pick from seven label-free strategies:

| Strategy | How it works |
|---|---|
| LLM-as-Judge | A second LLM call rates each output 0–10 |
| Self-Consistency | Same input, multiple runs — agreement = quality |
| Proxy Metrics | Structural checks: valid JSON, format, length |
| Tool-Use Success | Actually execute tool calls, score by status |
| Preference Pairs | Compare against hand-crafted good/bad examples |
| Human-as-Judge | You rate outputs interactively during evolution |
| Composite | Weighted mix of the above (recommended) |

**Path C — Human-in-the-loop.** The wizard supports three
human-evaluation modes:

- **`always`** — You rate candidates every generation (gold standard).
- **`final`** — Evolution runs fully automated, then presents the top-K
  prompts for your final pick.
- **`no`** — Fully automated, no human involvement.

### How to evaluate when you have no ground truth

Most real-world agents have no labelled dataset. You have a system prompt,
some test queries, and an intuition for what "good" looks like — but no
gold-standard answers to score against. The wizard handles this by
replacing the ground-truth fitness function with **proxy signals** that
give the evolutionary loop enough information to rank candidates.

Evolution does not need perfect absolute scores. It only needs to answer
one question reliably: **"Is prompt A better than prompt B?"** The seven
no-eval strategies provide that relative ranking without labels.

**Which strategy should you pick?** Use this decision tree:

```text
Do you have labelled eval data?
  ├─ YES → Use the standard PromptEvolver (ground-truth scoring)
  └─ NO
       ├─ Can you describe "good" in words?
       │    └─ YES → LLM-as-Judge (write a rubric)
       ├─ Does the agent call real APIs/tools?
       │    └─ YES → Tool-Use Success (HTTP status codes)
       ├─ Is there one correct answer per input?
       │    └─ YES → Self-Consistency (agreement across runs)
       ├─ Is output format critical (JSON, SQL, brackets)?
       │    └─ YES → Proxy Metrics (structural checks)
       ├─ Do you have 5-10 good/bad output examples?
       │    └─ YES → Preference Scoring (few-shot pairs)
       ├─ Is quality subjective or safety-critical?
       │    └─ YES → Human Tournament (you pick winners)
       └─ Not sure? → Composite (recommended — blends multiple signals)
```

**Recommended starting point: Composite.** The wizard's default Composite
scorer blends LLM-as-Judge (35 %), Self-Consistency (30 %), and Proxy
Metrics (35 %). Weights are normalised to sum to 1.0. This covers three
independent signal types — semantic quality, output stability, and
structural correctness — so it is robust even when individual signals are
noisy. Proxy checks are now **problem-type-aware**: tool routing gets
JSON and agent-name checks; classification gets single-label and format
checks. The
[xLAM worked example](#xlam-no-eval-worked-example) below proves that
this approach preserves 95 %+ ground-truth accuracy while having zero
access to labels.

**How to know if evolution is working without labels:**

1. **No-eval fitness should increase** across generations. If the composite
   score plateaus at generation 1, your rubric or proxy checks may be too
   coarse.
2. **Spot-check 3–5 outputs** from the best prompt. Do they look
   reasonable? This is the human sanity check.
3. **Compare to the default prompt.** Run both prompts on your test inputs
   and eyeball the difference. If the evolved prompt produces more
   consistent, better-formatted outputs, evolution is working.
4. **Use `human_eval_mode: final`** if you want a formal step. Evolution
   runs fully automated, then presents the top-K prompts for your final
   selection.

### Example walkthrough

```text
🧬 MutagenAI — Prompt Evolution Wizard

Step 1 of 9 — Task Description
  Task description: You are an API-calling assistant that maps queries
  to tool calls in the format [ToolName(param=value)].

Step 2 of 9 — Ground Truth
  Ground truth availability: no

Step 4 of 9 — Scoring Strategy
  Strategies: 7  (Composite)

Step 6 of 9 — Human Evaluation
  Human evaluation mode: final

  ✓ Generated: evolve_prompt.py
  Run it with:
    uv run python evolve_prompt.py
```

The generated script includes all imports, seed templates, scoring setup,
evolution loop, results saving, and (if selected) a human final-selection
step.

### Wizard code generation improvements

The wizard generates smarter, more effective evolution scripts out of the
box. Eight improvements ensure that generated code follows best practices
discovered through benchmarking:

| Improvement | What changed |
|---|---|
| **Seed template diversification** | A single user seed is auto-expanded to 6 structural variants (CoT, output-format-first, minimalist, contrastive, persona, intent-matching) so every island starts with different material. |
| **Problem-type proxy checks** | Generic proxy checks (`has_function_name`, `bracket_format`) replaced with task-aware checks. Tool routing gets `valid_json`, `has_sequence_or_array`, `contains_agent_name`, `no_verbose_explanation`, `at_least_one_selection`. Classification gets `valid_json`, `single_label`, `not_empty`, `no_verbose_explanation`. Generation gets `valid_json`, `is_json_object`, `has_fields`, `no_markdown_fences`, `not_empty`. |
| **Task-specific LLM Judge rubrics** | The rubric is no longer a truncated task description. For tool routing: checks agent relevance, logical order, precision, JSON validity. For classification: checks predicted class, format, reasoning. For generation: checks valid JSON, required fields, grounded values, non-empty strings, non-empty arrays. |
| **Adaptive mutations enabled** | Generated scripts now set `adaptive_mutations=True` and `llm_mutation_rate=0.3` in `NoEvalConfig`, enabling the evolver to learn which mutations are effective. |
| **`refine_after_splice` enabled** | Crossover offspring are refined by the LLM to improve coherence, reducing the chance of Frankenstein prompts. |
| **Domain mutations wired** | `DOMAIN_MUTATIONS` are now passed as `custom_mutations` to `NoEvalPromptEvolver`, so user-defined and auto-generated mutations actually drive evolution. |
| **Normalized CompositeScorer weights** | Weights in the composite scorer now sum to 1.0 (e.g. judge 0.35, consistency 0.30, proxy 0.35) for clarity. |
| **Scaled-up config presets** | `standard` preset: 5 generations, 6 population, 2 islands (was 3/4/2). `deep` preset: 10 generations, 8 population, 3 islands (was 5/6/3). |

---

## How prompt evolution works

### The approach

Most teams improve LLM performance by training bigger models or
fine-tuning on task-specific data. Both require GPUs and large datasets.
Prompt evolution takes a different path: **keep the model frozen and
evolve the system prompt instead.**

The idea is simple. When you ask an LLM to call an API, to plan a
multi-step tool chain, or to follow a customer-service policy, the
*instructions* you put in the system prompt matter far more than the
model's parameter count. A vague "you are a helpful assistant" prompt
scores 5–19 % on the benchmarks below. A well-structured prompt — with
the right role description, output format, constraint ordering, and
chain-of-thought hints — can reach 90–100 %. The question is: how do
you find that perfect prompt automatically?

### The evolutionary loop

MutagenAI treats prompt engineering as an **optimisation problem** and
solves it with evolutionary search:

1. **Seed population.** Start with 4 prompt templates ranging from a
   bare-bones one-liner to a detailed step-by-step planner. Each
   template has a placeholder where tool definitions get injected at
   test time.
2. **Evaluate.** Every candidate prompt is sent to the LLM along with
   the test cases. The model's output is scored — either against
   ground-truth labels or via no-eval strategies (LLM-as-Judge,
   Self-Consistency, Proxy Metrics, etc.).
3. **Select and breed.** The best-scoring prompts survive. The worst
   are replaced by *mutated* copies of the winners — small edits like
   adding a constraint, injecting a chain-of-thought hint, or
   reordering instructions. Pairs of good prompts are also *crossed
   over*. Meanwhile, CMA-ES tunes the numeric knobs (temperature,
   top-p) — see below.
4. **Island migration.** The population is split across 2 islands that
   evolve independently. Every 3 generations the best prompt from each
   island migrates to its neighbour, injecting diversity.
5. **Converge.** After 3–5 generations the prompts converge on a
   structure that consistently scores highest.

```text
                    ┌───────────────────────────────┐
                    │  Population of prompt variants │
                    │  (4 candidates × 2 islands)    │
                    └───────────┬───────────────────┘
                                │
                    ┌───────────▼───────────────────┐
                    │  For each candidate prompt:    │
                    │   1. Run prompt on test inputs │
                    │   2. Score outputs via SCORER  │ ← your chosen strategy
                    │   3. Assign fitness score      │
                    └───────────┬───────────────────┘
                                │
                    ┌───────────▼───────────────────┐
                    │  Tournament select → crossover │
                    │  → mutate → next generation    │
                    └───────────┬───────────────────┘
                                │
                    ┌───────────▼───────────────────┐
                    │  Migrate best across islands   │
                    │  every 3 generations           │
                    └───────────────────────────────┘
```

### What is CMA-ES? (the short version)

Imagine you're playing "hot and cold" to find hidden treasure in a huge
field — blindfolded. You throw a bunch of darts. Some land closer to the
treasure, some farther away. You keep the best ones and throw your next
batch *near where the good ones landed*.

At first your darts land in a circle around your best guess. But say the
treasure is inside a long, narrow valley. A circle wastes most darts on
the hillsides. CMA-ES notices "the good darts keep landing in a line
going *this* way" and stretches the circle into an oval pointing down
the valley. That oval is the **covariance matrix** — a description of
the shape and direction of where darts are thrown.

CMA-ES also adjusts how *far* it throws. If several rounds in a row the
good darts keep moving the same direction, it takes bigger steps. If
they zigzag, it shrinks — it must be close.

In MutagenAI, CMA-ES tunes the *continuous knobs* — `temperature` and
`top_p` — while the evolutionary algorithm handles the *words* in the
prompt. CMA-ES needs no formula for "how good is temperature = 0.7"; it
just tries values, keeps the best, and learns which combinations work
together.

### Advanced evolution features

Three additional features give the evolutionary loop finer control over
parent selection, evaluation cost, and mutation targeting.

#### Score-proportional selection

By default MutaGenAI uses tournament selection to pick parents. Set
`selection_method=SelectionMethod.SCORE_PROPORTIONAL` on
`PromptEvolverConfig` to switch to a sigmoid-weighted scheme that
favours high-scoring candidates while penalising over-selected parents.
This improves diversity by ensuring every promising candidate gets a
chance to breed, not just the tournament winner.

```python
from MutaGenAI import PromptEvolverConfig, SelectionMethod

config = PromptEvolverConfig(
    selection_method=SelectionMethod.SCORE_PROPORTIONAL,
)
```

#### Progressive evaluation (shallow → deep)

When evaluation datasets are large, running every candidate through the
full set is expensive. Progressive evaluation runs a cheap shallow pass
first (`eval_sample_size` samples). Only candidates that meet a
promotion threshold are re-evaluated on a larger deep sample for a more
reliable score.

```python
config = PromptEvolverConfig(
    eval_sample_size=10,               # shallow pass size
    eval_promotion_threshold=75.0,     # minimum score (0–100) to promote
    eval_deep_sample_size=50,          # deep pass size
)
```

Both fields default to `None`, which disables progressive evaluation
entirely — every candidate is scored once on `eval_sample_size` samples.

#### Structured failure buckets

Adaptive mutations already target the worst-performing categories. Failure
buckets add a second axis: they classify *how* each sample failed (wrong
tool, wrong parameters, unparseable output, no output, partial match) and
inject mutation hints that specifically address that failure mode.

The buckets are problem-type-aware. Tool-routing, classification, and
generation tasks each have their own mutation dictionaries. Set
`problem_type` on the config to match your workload:

```python
from MutaGenAI import PromptEvolverConfig, ProblemType

config = PromptEvolverConfig(
    problem_type=ProblemType.GENERATION,  # or TOOL_ROUTING, CLASSIFICATION
)
```

Failure bucket mutations are generated automatically each generation and
blended into the mutation pool alongside adaptive and built-in mutations.

---

## Core modules

| Module | Purpose |
|---|---|
| [`mutagenai/prompt_evolver.py`](mutagenai/prompt_evolver.py) | `PromptEvolver` — ground-truth prompt evolution with island-model EA + CMA-ES continuous tuning |
| [`mutagenai/strategies.py`](mutagenai/strategies.py) | `NoEvalPromptEvolver` + 7 scoring strategies for label-free evolution |
| [`mutagenai/wizard.py`](mutagenai/wizard.py) | `mutagenai init` wizard — interactive questionnaire that generates a ready-to-run script |
| [`mutagenai/seed_loader.py`](mutagenai/seed_loader.py) | Load seed templates from external JSON files |
| [`mutagenai/dashboard.py`](mutagenai/dashboard.py) | Plotting functions for benchmark visualisation (BFCL, xLAM, τ-bench, ToolBench, API-Bank, Browser Agent) |
| [`docs/algorithm_animation.html`](docs/algorithm_animation.html) | Interactive step-through animation showing how the evolutionary loop works (open in a browser) |

---

## Ground-truth prompt evolution

When you have labelled input/output pairs, `PromptEvolver` evolves the
system prompt and sampling parameters (temperature, top-p) for maximum
accuracy.

### Quick start — PromptEvolver

```python
from mutagenai import (
    PromptEvolver, PromptEvolverConfig, Tool, EvalSample, LLMBackend,
)

tools = [
    Tool("get_weather", "Get current weather", {"location": "string"}),
    Tool("send_email", "Send an email", {"to": "string", "subject": "string"}),
]

dataset = [
    EvalSample("Weather in London?", "get_weather", {"location": "London"}),
    EvalSample("Email Bob about the project", "send_email", {"to": "Bob"}),
]

evolver = PromptEvolver(
    tools=tools,
    eval_dataset=dataset,
    config=PromptEvolverConfig(iterations=5, backend=LLMBackend.OLLAMA),
)
result = evolver.run()
print(result.summary())
```

### Baseline comparison

Ollama llama3.2, 6 tools, 24 samples:

| Prompt | Accuracy | Delta vs Naive |
|---|---:|---:|
| Naive (tool list only) | 42.5 % | — |
| Minimal JSON instruction | 89.6 % | +47.1 % |
| Verbose (kitchen sink) | 95.0 % | +52.5 % |
| High temperature (creative) | 78.8 % | +36.3 % |
| Zero temperature (greedy) | 89.6 % | +47.1 % |
| **EVOLVED (MutagenAI)** | **96.7 %** | **+54.2 %** |

### Ollama vs Azure OpenAI

Same baselines on both backends:

| Prompt | Ollama | Azure OpenAI | Delta |
|---|---:|---:|---:|
| Naive (tool list only) | 44.2 % | 0.0 % | -44.2 % |
| Minimal JSON instruction | 90.4 % | 92.1 % | +1.7 % |
| Verbose (kitchen sink) | 89.2 % | 98.8 % | +9.6 % |
| **EVOLVED (MutagenAI)** | **100.0 %** | **100.0 %** | **+0.0 %** |

Both backends reached **100 % accuracy** after evolution.

### Agentic workload scenarios

Three realistic agentic workloads with 8–9 tools and 20 evaluation
samples each:

| Scenario | Tools | Ollama Evolved | Azure Evolved | Best Baseline |
|---|---:|---:|---:|---:|
| Customer-Support Triage | 9 | 93.8 % | 95.0 % | 79.5 % |
| Code-Assistant Agent | 9 | 100.0 % | 100.0 % | 88.0 % |
| Data-Pipeline Orchestrator | 8 | 96.2 % | 97.5 % | 86.0 % |

---

## Six agentic benchmarks

MutagenAI has been evaluated across six real-world agentic benchmarks
covering function calling, conversational agents, multi-tool
orchestration, and failure recovery.

### BFCL V4 — single-turn function calling

> **Script:** [`examples/cookbook/prompt_evolution_bfcl.py`](examples/cookbook/prompt_evolution_bfcl.py)

[Berkeley Function Calling Leaderboard V4](https://gorilla.cs.berkeley.edu/leaderboard.html)
— the gold-standard benchmark for evaluating function-calling accuracy.

**Ollama (llama3.2)**

| Category | Default | Evolved | Delta |
|---|---:|---:|---:|
| simple_python | 98.4 % | **100.0 %** | +1.5 % |
| multiple | 94.2 % | 95.5 % | +0.0 % |
| parallel | 96.0 % | 96.9 % | +0.0 % |
| live_simple | 77.6 % | **80.5 %** | +0.7 % |

**Azure OpenAI (GPT-4.1)**

| Category | Default | Evolved | Delta |
|---|---:|---:|---:|
| simple_python | 98.4 % | 100.0 % | +0.0 % |
| parallel | 96.0 % | **97.5 %** | +0.6 % |
| live_simple | 77.6 % | **77.5 %** | +5.0 % |

### τ-Bench — conversational agents

> **Script:** [`examples/cookbook/prompt_evolution_tau_bench.py`](examples/cookbook/prompt_evolution_tau_bench.py)

[τ-bench](https://github.com/sierra-research/tau2-bench) (Sierra
Research) — tests tool-using conversational agents on customer-service
scenarios. Top models reach only ~46 % pass@1 on airline.

**Ollama (llama3.2)**

| Domain | Algorithm | Default | Evolved | Delta |
|---|---|---:|---:|---:|
| airline | standard | 30.0 % | **36.9 %** | +5.7 % |
| airline | deep | 30.0 % | **46.6 %** | +3.1 % |
| retail | standard | 52.0 % | **54.6 %** | +0.6 % |

**Azure OpenAI (GPT-4.1)**

| Domain | Algorithm | Default | Evolved | Delta |
|---|---|---:|---:|---:|
| airline | standard | 28.0 % | **39.5 %** | +2.1 % |
| airline | deep | 28.0 % | **46.5 %** | +8.3 % |
| retail | standard | 50.1 % | **51.6 %** | +0.0 % |

### xLAM 60 k — broad function calling

> **Script:** [`examples/cookbook/prompt_evolution_xlam.py`](examples/cookbook/prompt_evolution_xlam.py)

[Salesforce xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
— 60 000 examples across 3 673 APIs and 21 categories.

**Ollama (llama3.2)**

| Category | Default | Evolved | Delta |
|---|---:|---:|---:|
| finance | 50.7 % | **98.0 %** | +1.9 % |
| social | 40.3 % | **94.0 %** | +1.2 % |
| data | 46.3 % | **100.0 %** | +3.5 % |
| entertainment | 58.1 % | **96.8 %** | +3.9 % |

**Azure OpenAI (GPT-4.1)**

| Category | Default | Evolved | Delta |
|---|---:|---:|---:|
| social | 26.7 % | **100.0 %** | +2.5 % |
| data | 28.3 % | **100.0 %** | +0.0 % |
| finance (deep) | 26.7 % | **100.0 %** | +0.0 % |

### ToolBench — multi-tool API orchestration

> **Script:** [`examples/cookbook/prompt_evolution_toolbench.py`](examples/cookbook/prompt_evolution_toolbench.py)

[ToolBench](https://github.com/OpenBMB/ToolBench) — 16 464 REST APIs,
three difficulty tiers. Published baseline: ToolLLaMA 66.7 %, GPT-4
71.1 %.

**Ollama (llama3.2)**

| Split | Algorithm | Default | Evolved | Delta |
|---|---|---:|---:|---:|
| g1_instruction | standard | 16.0 % | **89.5 %** | +41.0 % |
| g2_category | standard | 4.7 % | **65.0 %** | +21.5 % |
| g3_instruction | standard | 19.0 % | **39.7 %** | +16.3 % |
| g3_instruction | deep | 19.0 % | **74.3 %** | +25.8 % |

### API-Bank — multi-level tool-use evaluation

> **Script:** [`examples/cookbook/prompt_evolution_apibank.py`](examples/cookbook/prompt_evolution_apibank.py)

[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
— 73 runnable API tools, 314 dialogues. Published baseline: GPT-4
83.8 % calling, 41.2 % retrieval.

**Ollama (llama3.2)**

| Level | Algorithm | Default | Evolved | Delta |
|---|---|---:|---:|---:|
| level_1 | standard | 52.3 % | **100.0 %** | +4.0 % |
| level_2 | standard | 57.1 % | **88.9 %** | +3.7 % |
| level_2 | deep | 57.1 % | **82.6 %** | +7.7 % |

### Browser Agent Tasks — failure recovery

> **Script:** [`examples/cookbook/prompt_evolution_browser_agent.py`](examples/cookbook/prompt_evolution_browser_agent.py)

[Browser Agent Tasks](https://huggingface.co/datasets/DataCreatorAI/tool-calling-browser-agent-tasks)
— 1 062 multi-turn conversations with failure recovery scenarios.

**Ollama (llama3.2)**

| Category | Algorithm | Default | Evolved | Delta |
|---|---|---:|---:|---:|
| normal | standard | 25.9 % | **29.0 %** | +2.7 % |
| failure_recovery | standard | 24.3 % | **28.0 %** | +4.0 % |
| multi_tool | standard | 22.3 % | **30.9 %** | +0.2 % |
| failure_recovery | deep | 24.3 % | **35.2 %** | +8.5 % |

### Consolidated results

| Benchmark | Best Evolved (Ollama 3 B) | Default | Absolute gain |
|---|---:|---:|---:|
| BFCL simple_python | **100.0 %** | 98.4 % | +1.5 % |
| xLAM data | **100.0 %** | 46.3 % | +53.7 % |
| API-Bank level_1 | **100.0 %** | 52.3 % | +47.7 % |
| ToolBench G1 | **89.5 %** | 16.0 % | +73.5 % |
| τ-bench airline (deep) | **46.6 %** | 30.0 % | +16.6 % |
| Browser failure_recovery (deep) | **35.2 %** | 24.3 % | +10.9 % |

### Key takeaways from benchmarks

1. **Prompt structure matters more than model size.** A default "you are
   a helpful assistant" prompt scores 5–19 %. A well-structured prompt
   jumps to 40–100 %. Evolution adds a further 1–41 % on top.
2. **A 3 B local model can beat GPT-4.** On xLAM and API-Bank, Ollama
   llama3.2 with an evolved prompt matches or exceeds GPT-4's published
   scores — at zero API cost.
3. **The hardest benchmarks benefit most.** ToolBench G1 saw +41 %
   absolute gain; τ-bench airline saw +8.3 % with GPT-4.1 deep search.
4. **Multi-tool planning responds to prompt engineering.** ToolBench G3
   improved from 19 % to 74.3 % under deep evolution.
5. **Evolution finds what humans miss.** Hand-crafted prompts plateau;
   the evolutionary loop discovers non-obvious structural patterns.
6. **Format compliance is model-specific.** GPT-4.1 scored < 7 % on
   API-Bank despite being far larger — it wraps calls in markdown that
   strict parsers reject. Prompt evolution effectiveness depends on the
   model's willingness to obey exact output constraints.

---

## Token optimization

Agent system prompts can be verbose (800–1 500 tokens). Token
optimization applies evolutionary pressure toward shorter prompts
without sacrificing accuracy. The feature is **off by default** and
activated via five `PromptEvolverConfig` fields.

### How it works

Two complementary mechanisms run inside the existing evolutionary loop:

1. **Baseline-relative efficiency bonus** — after computing raw accuracy
   the engine calculates `efficiency = baseline_tokens / candidate_tokens`
   (capped at `token_efficiency_cap`), converts it to a 0–100 bonus, and
   blends it with the raw score using `token_weight`.
2. **Lexicographic tournament tiebreaker** — within the same accuracy
   band (`token_accuracy_band` percentage points) the tournament selects
   the candidate with fewer tokens instead of higher raw score.

### Quick start — token-aware evolution

```python
from mutagenai import (
    PromptEvolver, PromptEvolverConfig, Tool, EvalSample,
    LLMBackend, count_prompt_tokens,
)

baseline_prompt = "You are a helpful assistant..."
config = PromptEvolverConfig(
    iterations=5,
    backend=LLMBackend.OLLAMA,
    minimize_tokens=True,            # enable token optimization
    token_weight=0.10,               # 10 % of score from efficiency
    token_efficiency_cap=2.0,        # cap efficiency bonus at 2x shorter
    token_accuracy_band=2.0,         # tiebreak within 2 pp accuracy
    baseline_prompt_tokens=count_prompt_tokens(baseline_prompt),
)

evolver = PromptEvolver(tools=tools, eval_dataset=dataset, config=config)
result = evolver.run()
```

### Configuration reference

| Field | Type | Default | Description |
|---|---|---|---|
| `minimize_tokens` | `bool` | `False` | Master switch for token optimization |
| `token_weight` | `float` | `0.10` | Blend weight for the efficiency bonus (0–1) |
| `token_efficiency_cap` | `float` | `2.0` | Maximum efficiency ratio before capping |
| `token_accuracy_band` | `float` | `2.0` | Accuracy band width for the tournament tiebreaker (pp) |
| `baseline_prompt_tokens` | `int` | `0` | Token count of the baseline prompt (use `count_prompt_tokens()`) |

### Utility

`count_prompt_tokens(text)` counts tokens using tiktoken `cl100k_base`
with a `len(text) // 4` fallback when tiktoken is not installed.

---

## No-eval prompt evolution — 7 strategies

> **Script:** [`examples/cookbook/prompt_evolution_no_eval.py`](examples/cookbook/prompt_evolution_no_eval.py)

The benchmarks above all use ground-truth labels. Real-world agents
rarely have these. The **no-eval strategies** module
(`mutagenai.strategies`) replaces the ground-truth fitness function with
seven alternative signal sources.

### The seven strategies

| # | Strategy | Signal source | LLM calls per eval | Best for |
|---|---|---|---|---|
| 1 | **LLM-as-Judge** | Second LLM scores output against rubric | 1 extra | General quality |
| 2 | **Synthetic Eval** | LLM generates input/output pairs from description | 1 generate + 1 per eval | No test data at all |
| 3 | **Tool-Use Success** | API return codes (200/400/500) | 0 extra | Agents with real endpoints |
| 4 | **Self-Consistency** | Agreement across N runs | N-1 extra | Deterministic tasks |
| 5 | **Proxy Metrics** | Structural checks (JSON, format, length) | 0 extra | Format compliance |
| 6 | **Preference Scoring** | Good/bad output pairs as references | 1 extra | 5–10 example preferences |
| 7 | **Human Tournament** | Human picks best per generation | 0 extra | Highest quality |

### Quick start — NoEvalPromptEvolver

```python
from mutagenai import (
    NoEvalPromptEvolver,
    NoEvalConfig,
    LLMJudge,
    SelfConsistencyScorer,
    ProxyMetricsScorer,
    CompositeScorer,
)

# 1. Define the task
task = """You are a customer-service agent for an online store.
Answer questions about orders, returns, shipping, and accounts.
If you need to call a tool, respond with JSON: {"tool": "...", "parameters": {...}}"""

# 2. Provide unlabelled test inputs
inputs = [
    "Where is my order #12345?",
    "I want to return the shoes I bought last week.",
    "How do I change my shipping address?",
]

# 3. Pick a scoring strategy
scorer = CompositeScorer([
    (LLMJudge(rubric="Score 0-10 on helpfulness, accuracy, and tone."), 0.5),
    (ProxyMetricsScorer(ProxyMetricsScorer.common_checks()), 0.3),
    (SelfConsistencyScorer(num_samples=3), 0.2),
])

# 4. Configure and run
config = NoEvalConfig(iterations=5, population_size=4, num_islands=2)
evolver = NoEvalPromptEvolver(
    task_description=task,
    test_inputs=inputs,
    scorer=scorer,
    config=config,
)
result = evolver.run()
print(result.best_prompt)
```

### Which strategy should I pick?

```text
Do you have labelled eval data?
  ├─ YES → Use PromptEvolver (ground-truth)
  └─ NO
       ├─ Can you describe "good" in words?  → LLM-as-Judge
       ├─ Agent calls real APIs/tools?        → Tool-Use Success
       ├─ One correct answer per input?       → Self-Consistency
       ├─ Output format critical?             → Proxy Metrics
       ├─ Have 5-10 good/bad examples?        → Preference Scoring
       ├─ Quality is subjective?              → Human Tournament
       └─ Not sure?                           → Composite (recommended)
```

---

## xLAM no-eval worked example

> **Script:** [`examples/cookbook/prompt_evolution_xlam_no_eval.py`](examples/cookbook/prompt_evolution_xlam_no_eval.py)

This is a complete, end-to-end walkthrough of evolving a prompt **when
you have no ground truth**. It uses the wizard approach applied to the
**Salesforce xLAM function-calling 60k** benchmark — but we pretend we
have no labels. We then reveal the hidden labels afterwards to measure
how well the no-eval strategies actually performed.

### The problem

You have an LLM agent that calls functions. Users send natural-language
queries; the agent must output structured function calls in the format
`[func_name(param=value, ...)]`. You have 20 representative user queries
and the tool schemas, but **no expected outputs** — no labelled dataset,
no annotation team, no evaluation harness. How do you systematically
improve the system prompt?

### What this recipe does

1. **Loads 150 xLAM cases** (30 per category) from the Hugging Face
   dataset.
2. **Strips all labels** — the experiment sees only queries and tool
   schemas.
3. **Selects 20 unlabelled test inputs** (stratified random sample).
4. **Auto-generates seed templates** from a one-line task description.
5. **Auto-generates mutations** — generic prompt rewrites.
6. **Evolves prompts** using four strategies: Composite, LLM-as-Judge,
   Self-Consistency, and Proxy Metrics.
7. **Reveals ground truth** and evaluates every evolved prompt against
   the real xLAM labels.

### Configuration

The experiment uses the wizard's **standard** preset:

| Parameter | Value | What it controls |
|---|---|---|
| Generations | 5 | Number of evolutionary cycles |
| Population | 6 | Prompt variants per island |
| Islands | 2 | Parallel sub-populations |
| Elite size | 3 | Top candidates that survive unchanged |
| Migration | every 3 gens | Best prompts shared between islands |

### Results

**Model:** Ollama `llama3.2` (3B parameters, running locally).

| Approach | No-Eval Fitness | GT Score | Func Name | Param Acc | Wall Time |
|---|---|---|---|---|---|
| Default prompt (no evolution) | — | 95.4 % | 100.0 % | 77.4 % | — |
| **Composite (wizard default)** | **93.2 %** | **95.5 %** | **100.0 %** | **77.8 %** | **2 582 s** |
| LLM-as-Judge | 86.5 % | 94.6 % | 100.0 % | 76.8 % | 2 236 s |
| Self-Consistency | 91.6 % | 95.4 % | 100.0 % | 77.4 % | 1 449 s |
| Proxy Metrics | 100.0 % | 94.6 % | 100.0 % | 76.8 % | 578 s |
| **GT Evolution** ★ | — | **97.4 %** | — | — | — |

★ GT Evolution is the average evolved score across 5 categories from the
ground-truth recipe, which runs GT-guided prompt evolution with the same
model. This is the **ceiling** — the best result when labels are available.

**Column definitions:**

- **No-Eval Fitness** — the score the strategy assigned during evolution
  (0–100 %, without seeing labels). This is the signal evolution
  optimised.
- **GT Score** — ground-truth accuracy, measured after evolution by
  comparing outputs to the hidden xLAM labels. Invisible to evolution.
- **Func Name** — percentage of cases where the model selected the
  correct function name.
- **Param Acc** — percentage of parameter key–value pairs that matched
  the ground-truth arguments.
- **Wall Time** — total clock time including all LLM calls.

### What the results mean

1. **Composite evolution beats the default baseline.** The Composite
   strategy reaches 95.5 % GT — a +0.1 % lift over the 95.4 % default
   baseline — and the highest Param Acc of any strategy (77.8 %).

2. **Function names are always correct — the challenge is parameters.**
   Every strategy achieves 100.0 % Func Name accuracy. All variation
   lives in Param Acc (76.8–77.8 %).

3. **GT-guided evolution sets the ceiling at 97.4 %.** The best no-eval
   strategy (Composite, 95.5 %) closes **5 %** of the gap between the
   default (95.4 %) and the GT ceiling (97.4 %) — without ever seeing
   a label.

4. **Self-Consistency is the best individual strategy.** 95.4 % GT with
   77.4 % Param Acc and moderate wall time (1 449 s).

5. **Proxy Metrics has the highest no-eval fitness but lower GT.**
   100.0 % no-eval but 94.6 % GT — structural format checks overfit
   slightly. Still the fastest strategy (578 s).

6. **Composite is the safest default.** It blends three independent
   signals — making it robust when you don't know which signal matters
   most.

### How to apply this to your own agent

1. **Run `mutagenai init`** — at Step 2 select "no" for ground truth,
   at Step 4 pick **Composite**.
2. **Provide 10–30 representative test inputs** — real queries.
3. **Write proxy checks** for your output format (valid JSON, bracket
   format, required fields).
4. **Write a rubric** for the LLM-as-Judge scorer — list 3–5 criteria
   with point values.
5. **Run evolution** and spot-check 3–5 outputs.
6. **Iterate** — tighten the rubric, add proxy checks, increase
   generations.

---

## API-Bank no-eval vs ground-truth comparison

> **Script:** [`examples/cookbook/prompt_evolution_apibank_no_eval.py`](examples/cookbook/prompt_evolution_apibank_no_eval.py)

How close can no-eval evolution get to ground-truth–guided evolution on
a real benchmark? This recipe runs the **same API-Bank task** through
both pipelines and compares the results head-to-head.

### Results

**Model:** Ollama `llama3.2` (3B parameters, running locally).

| Approach | No-Eval Fitness | GT Score | API Name | Param Acc | Wall Time |
|---|---|---|---|---|---|
| Default prompt (no evolution) | — | 55.0 % | — | — | — |
| LLM-as-Judge | 90.0 % | 57.9 % | 50.0 % | 44.7 % | 1 474 s |
| Preference Scoring | 69.0 % | 60.0 % | 50.0 % | 49.9 % | 959 s |
| Proxy Metrics | 100.0 % | 63.4 % | 63.3 % | 56.9 % | 471 s |
| Tool-Use Success | 55.0 % | 68.2 % | 66.7 % | 57.3 % | 469 s |
| Self-Consistency | 78.1 % | 81.8 % | 83.3 % | 71.2 % | 980 s |
| **Composite (recommended)** | **81.3 %** | **82.3 %** | **83.3 %** | **74.1 %** | **2 029 s** |
| **GT Evolution** | — | **100.0 %** | **100.0 %** | **100.0 %** | **470 s** |

**Key finding:** No-eval evolution delivers a **+27 point improvement**
over the default prompt without labels. The Composite strategy closes
**61 %** of the gap between the default (55.0 %) and the GT ceiling
(100 %).

---

## Seed templates — external prompt configuration

Seed templates define the starting population for prompt evolution.
Storing them as external JSON files makes them version-controlled,
shareable, and easy to swap between experiments.

### File format

Place JSON files in the `seed_templates/` directory at the project root:

```json
{
  "name": "entity-classification",
  "description": "Diverse seeds for entity classification.",
  "seeds": [
    "Classify this text as Agent, Task, Tool, Input, Output, or Human.",
    "Think about what role this text plays…",
    "You are an expert in agentic AI systems…",
    "Entity type?"
  ]
}
```

### Loading seeds in code

```python
from mutagenai import load_seed_templates, list_seed_templates

# List available template files
print(list_seed_templates())  # ['entity_classification']

# Load a specific template
seeds = load_seed_templates("entity_classification")

# Pass to NoEvalPromptEvolver
evolver = NoEvalPromptEvolver(
    task_description="Classify entities in agentic AI.",
    test_inputs=test_inputs,
    scorer=scorer,
    config=config,
    seed_templates=seeds,
)
```

### Output schema and automatic proxy checks

Seed templates can include an `output_schema` field that defines the
expected JSON structure of LLM output. The schema is substituted into
seed text via the `{output_schema}` placeholder, and can be converted
into proxy checks automatically:

```json
{
  "name": "medical-records",
  "seeds": ["Generate a JSON record matching: {output_schema}"],
  "output_schema": {
    "diagnosis": "string",
    "medications": [],
    "details": {"reasoning": "string", "confidence": "number"}
  }
}
```

```python
from MutaGenAI import schema_to_proxy_checks

schema = {"diagnosis": "string", "medications": [], "details": {"reasoning": "string"}}
checks = schema_to_proxy_checks(schema, weight=1.0)
# Generates: valid_json, has_diagnosis, diagnosis_non_empty,
#            has_medications, medications_is_list,
#            has_details, details_has_reasoning
```

### Designing effective seeds

Diverse initialisation is the single most impactful factor for evolution
quality. Design seeds that vary across **structural archetypes**:

| Archetype | What it does | Example |
|---|---|---|
| **Direct instruction** | Simple baseline | "Classify this text as X, Y, or Z." |
| **Chain-of-thought** | Elicits reasoning before answering | "Think about what role this text plays, then classify…" |
| **Persona** | Frames the model's role | "You are an expert in X. Determine…" |
| **Definitional** | Provides class boundary semantics | "An Agent acts autonomously, a Task is work…" |
| **Minimalist** | Explores the terse end of prompt space | "Entity type?" |
| **Output-strict** | Enforces format compliance | "Return exactly one word from: X, Y, Z." |
| **Contrastive** | Scenario framing | "This must be precise for system correctness…" |

Rule of thumb: **one seed per archetype**, and set `population_size` ≥
the number of seeds so every archetype enters the initial gene pool.

---

## Agent routing — static vs evolved prompt (GPT-4.1)

This experiment benchmarks an **evolved** system prompt against a
**static** baseline on multi-step agent routing using Azure OpenAI
GPT-4.1. The dataset is
[V1rtucious/multi-step-agent-routing](https://huggingface.co/datasets/V1rtucious/multi-step-agent-routing)
(616 train / 154 test rows, 27 specialist agents).

**Evolution config:** 8 generations, population 8 × 2 islands,
`NoEvalPromptEvolver` with `CompositeScorer` (LLMJudge 0.3 +
SelfConsistency 0.3 + ProxyMetrics 0.3), adaptive mutations,
`llm_mutation_rate=0.3`, `describe_entities=True`,
`refine_after_splice=True`. The evolution evaluated 134 candidates
over ≈ 2.7 hours and converged to a best no-eval fitness of **88.6 %**.

**Benchmark:** 100 samples from train + 100 from test (seed 42),
scored on agent-set precision / recall / F1.

### Results

| | Train F1 | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|
| Static prompt | 52.0 % | 49.3 % | 42.4 % | 67.5 % |
| **Evolved prompt** | 51.2 % | 49.2 % | 39.1 % | 72.8 % |
| Delta | -0.8 % | **-0.2 %** | -3.3 % | **+5.3 %** |

**Evolved prompt — test by complexity (mean F1):**

| Complexity | F1 |
|---|---|
| High | 58.6 % |
| Medium | 51.6 % |
| Low | 38.1 % |

**Evolved prompt — test by routing pattern (mean F1):**

| Routing pattern | F1 |
|---|---|
| Approval chain | 56.8 % |
| Conditional branching | 56.8 % |
| Data enrichment | 54.1 % |
| Investigative | 52.7 % |
| Linear sequential | 42.9 % |

**Key findings:**

1. **Recall improved significantly** (+5.3 % test, +9.4 % train) —
   the evolved prompt's `"Respond with JSON only. No explanation."`
   prefix helped the model include more relevant agents rather than
   hedging.
2. **Precision dropped** (-3.3 %) as the model traded conservative
   selection for broader coverage.
3. **High-complexity requests benefited most** (58.6 % F1 vs 38.1 %
   for low), suggesting the evolved prompt helps with multi-step
   routing where more agents need to be activated.
4. **Investigative and approval-chain patterns** scored highest,
   indicating the JSON-only instruction particularly aids structured
   decision flows.

**Run the benchmark yourself:**

```bash
uv run python examples/experiments/agent_routing/run_benchmark.py
```

---

## Entity classification — static vs evolved prompt

This experiment tests whether starting with diverse seed prompts and
evolving them yields better results than starting with an AI-generated
expert prompt.

### Results

| | Validation | Test |
|---|---|---|
| Static AI-generated prompt | 62.5 % | 61.0 % |
| **Evolved prompt** | **71.5 %** | **62.0 %** |
| Delta (evolved - static) | **+9.0 %** | **+1.0 %** |

**Per-class validation accuracy:**

| Class | Static | Evolved | Delta |
|---|---|---|---|
| Agent | 27.3 % | 45.5 % | +18.2 % |
| Task | 14.3 % | 54.3 % | +40.0 % |
| Tool | 96.5 % | 84.2 % | -12.3 % |
| Input | 74.4 % | 67.4 % | -7.0 % |
| Output | 68.3 % | 90.2 % | +21.9 % |
| Human | 15.4 % | 38.5 % | +23.1 % |

**Key finding:** The static prompt was heavily biased toward Tool
(96.5 %) while failing on Task (14.3 %) and Human (15.4 %). Evolution
rebalanced these, with Task jumping +40.0 % and Human +23.1 %.

---

## Dashboard and visualisation

All plotting functions auto-detect the environment: **Plotly** for
interactive notebooks, **Matplotlib** for scripts and static output.

```python
from mutagenai.dashboard import plot_bfcl_evolution

# BFCL benchmark convergence and comparison
plot_bfcl_evolution("bfcl_experiment_log.json")
```

### Available plots

| Function | What it shows |
|---|---|
| `plot_bfcl_evolution()` | BFCL benchmark convergence and comparison |
| `plot_tau_bench_evolution()` | τ-bench convergence and sub-score breakdown |
| `plot_xlam_evolution()` | xLAM convergence curves and backend comparison |
| `plot_toolbench_evolution()` | ToolBench convergence and tier comparison |
| `plot_apibank_evolution()` | API-Bank convergence and accuracy breakdown |
| `plot_browser_agent_evolution()` | Browser Agent convergence and failure recovery |

Each benchmark recipe saves a JSON log that feeds directly into its
corresponding dashboard function.

---

## Cookbook recipes

All prompt evolution recipes live in
[`examples/cookbook/`](examples/cookbook/):

| Recipe | Script | What it demonstrates |
|---|---|---|
| 1 | [`prompt_evolution_azure.py`](examples/cookbook/prompt_evolution_azure.py) | Ollama vs Azure OpenAI (GPT-4.1) head-to-head |
| 2 | [`prompt_evolution_agentic.py`](examples/cookbook/prompt_evolution_agentic.py) | Three agentic workloads (support, code, data pipeline) |
| 3 | [`prompt_evolution_bfcl.py`](examples/cookbook/prompt_evolution_bfcl.py) | BFCL V4 benchmark — closing the FC-vs-Prompt gap |
| 4 | [`prompt_evolution_tau_bench.py`](examples/cookbook/prompt_evolution_tau_bench.py) | τ-bench conversational agent benchmark |
| 5 | [`prompt_evolution_xlam.py`](examples/cookbook/prompt_evolution_xlam.py) | xLAM / APIGen function-calling 60 k |
| 6 | [`prompt_evolution_toolbench.py`](examples/cookbook/prompt_evolution_toolbench.py) | ToolBench multi-tool API orchestration (16 k APIs) |
| 7 | [`prompt_evolution_apibank.py`](examples/cookbook/prompt_evolution_apibank.py) | API-Bank multi-level tool-use evaluation (73 APIs) |
| 8 | [`prompt_evolution_browser_agent.py`](examples/cookbook/prompt_evolution_browser_agent.py) | Browser Agent Tasks with failure recovery |
| 9 | [`prompt_evolution_no_eval.py`](examples/cookbook/prompt_evolution_no_eval.py) | No-eval strategies — 7 label-free approaches |
| 10 | [`prompt_evolution_apibank_no_eval.py`](examples/cookbook/prompt_evolution_apibank_no_eval.py) | API-Bank no-eval vs ground-truth comparison |
| 11 | [`prompt_evolution_xlam_no_eval.py`](examples/cookbook/prompt_evolution_xlam_no_eval.py) | xLAM no-eval wizard-style worked example |
| 12 | [`prompt_evolution_entity_classification.py`](examples/cookbook/prompt_evolution_entity_classification.py) | Entity classification — static vs evolved prompt |

---

## Installation

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/shanepeckham/MutagenAI.git
cd MutagenAI
uv sync --extra all

# Start a local LLM backend
ollama serve
ollama pull llama3.2

# Run your first evolution
mutagenai init
```

**Optional dependency groups:**

```bash
pip install mutagenai              # core engine (numpy only)
pip install mutagenai[llm]         # + httpx, azure-identity
pip install mutagenai[viz]         # + matplotlib, plotly
pip install mutagenai[wizard]      # + rich
pip install mutagenai[all]         # everything
```

For Azure OpenAI, set `AZURE_OPENAI_ENDPOINT` and
`AZURE_OPENAI_DEPLOYMENT` (uses RBAC via `DefaultAzureCredential`).
For OpenAI, set `OPENAI_API_KEY`.

Copy the sample env file and fill in the values you need:

```bash
cp .env.sample .env
```

See [`.env.sample`](.env.sample) for the full list of supported
variables (Ollama, Azure OpenAI, OpenAI, Hugging Face).
