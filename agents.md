# MutaGenAI — agent instructions

## What MutaGenAI is

MutaGenAI is a standalone Python library extracted from the EvoSim
evolutionary computing framework. It evolves LLM system prompts using
evolutionary search — no fine-tuning, no GPUs, no labelled data required.

The library treats prompt engineering as an optimisation problem. It
maintains a population of prompt variants across isolated islands, scores
each variant against benchmark data (or proxy signals when labels are
unavailable), selects winners via tournament selection, applies mutations
and crossover, and migrates elites between islands. CMA-ES tunes
continuous parameters (temperature, top-p) alongside the discrete prompt
search.

## Origin and extraction history

MutaGenAI was extracted from
[EvoSim](https://github.com/shanepeckham/EvoSim) in April 2026. The
extraction was clean because the prompt evolution code had **zero hard
coupling** to EvoSim's evolutionary algorithm engine:

- `prompt_evolver.py` had zero internal EvoSim imports (self-contained
  island-model EA + CMA-ES)
- `strategies.py` only imported from `prompt_evolver`
- `seed_loader.py` had zero internal imports (stdlib only)

All `from evosim.` imports were rewritten to `from MutaGenAI.`. The
wizard was rebranded (banners, CLI examples, generated code references).
The dashboard was rebuilt from scratch using only the 6 benchmark plot
families — all EA-specific plots (`plot_convergence`, `plot_pareto`,
`plot_map_elites`, `Dashboard` class) were excluded.

109 tests pass. The package installs cleanly via `uv sync`.

## Repository structure

```text
MutaGenAI/
├── MutaGenAI/                 # Package source
│   ├── __init__.py            # Public API — all exports
│   ├── prompt_evolver.py      # Core engine: PromptEvolver, island EA, CMA-ES, LLM backends
│   ├── strategies.py          # 7 no-eval scoring strategies + NoEvalPromptEvolver
│   ├── seed_loader.py         # Load seed templates from JSON files
│   ├── wizard.py              # Interactive CLI wizard (9-step questionnaire)
│   └── dashboard.py           # 6 benchmark plot families (Plotly + Matplotlib fallback)
├── docs/                      # Interactive animation (algorithm_animation.html)
├── logs/                      # Experiment logs from benchmark runs (JSON + CSV)
├── operator_prompts/          # LLM operator prompts (assess, generate, judge, roadmap, constitution)
├── seed_templates/            # JSON seed template files
├── examples/
│   ├── cookbook/               # 12 runnable recipes (BFCL, xLAM, τ-bench, ToolBench, API-Bank, etc.)
│   └── experiments/           # Multi-run experiment scripts (apibank, xlam, entity_classification)
├── tests/                     # 109 tests (pytest)
├── pyproject.toml             # Build config (hatchling), dependencies, CLI entry point
├── README.md                  # Full documentation with benchmark results
└── agents.md                  # This file
```

## Key modules and their responsibilities

### prompt_evolver.py (~1 539 LOC)

The core evolutionary prompt engine. Contains:

- **`PromptEvolver`** — main class for ground-truth prompt evolution.
  Implements island-model EA with tournament selection, crossover,
  mutation, and elite preservation. Runs CMA-ES in parallel to tune
  temperature and top-p.
- **`PromptEvolverConfig`** — configuration dataclass (iterations,
  population_size, num_islands, mutation_rate, crossover_rate,
  migration_interval, elite_size, standard/deep presets).
- **`PromptCandidate`** — a single prompt variant with score, hash,
  generation, island_id, lineage (parent_hashes, operation).
- **`LLMBackend`** / **`LLMClient`** — enum + HTTP client for Ollama,
  OpenAI, and Azure OpenAI. Lazy imports of `httpx` and
  `azure-identity`.
- **`Tool`**, **`EvalSample`** — data classes for tool schemas and
  labelled evaluation samples.
- **`ProblemType`** — enum: tool_calling, classification, generation,
  code, conversation.
- **`ErrorProfile`** — tracks common failure modes during evolution.
- **`evolve_prompt_with_cmaes()`** — CMA-ES continuous parameter tuning.
- **`generate_adaptive_mutations()`** — generates domain-specific
  mutations based on error analysis.
- **`get_mutations_for_problem_type()`** — returns the 18 built-in
  mutation operators for a given problem type.

Only hard dependency: `numpy>=1.26`. LLM backends require
`httpx>=0.27` (optional).

### strategies.py (~1 290 LOC)

Seven label-free scoring strategies for when you have no ground truth:

| Class | What it does |
|---|---|
| `LLMJudge` | Second LLM scores output against a rubric (0–10) |
| `SyntheticEvalGenerator` / `SyntheticEvalScorer` | LLM generates synthetic input/output pairs |
| `ToolSuccessScorer` / `ToolResult` | Scores by HTTP status code from real API calls |
| `SelfConsistencyScorer` | Agreement across N runs of the same input |
| `ProxyMetricsScorer` / `ProxyCheck` | Structural checks (valid JSON, format, length) |
| `PreferenceScorer` / `PreferencePair` | Few-shot good/bad output comparison |
| `HumanTournament` | Interactive human selection per generation |
| `CompositeScorer` | Weighted blend of multiple strategies |

**`NoEvalPromptEvolver`** wraps `PromptEvolver` with any `Scorer`
subclass, routing fitness through the chosen strategy instead of
ground-truth labels.

### wizard.py (~1 030 LOC)

Interactive 9-step CLI wizard. Generates a ready-to-run Python script
tailored to the user's agent. Uses Rich for terminal UI (graceful
fallback to plain input if Rich is not installed).

Entry point: `MutaGenAI init` (mapped in pyproject.toml to
`MutaGenAI.wizard:run_wizard`).

### seed_loader.py (~62 LOC)

Loads seed template JSON files from the `seed_templates/` directory.
Two public functions: `load_seed_templates(name)` and
`list_seed_templates()`.

### dashboard.py (~1 100 LOC)

Six benchmark-specific plot families. Each has three sibling functions:

- `plot_<benchmark>_evolution()` — public entry point, auto-detects env
- `_plot_<benchmark>_plotly()` — interactive Plotly version
- `_plot_<benchmark>_mpl()` — static Matplotlib fallback

Benchmarks: BFCL, τ-bench, xLAM, ToolBench, API-Bank, Browser Agent.

Four utility functions: `_has_plotly()`, `_has_matplotlib()`,
`_has_ipywidgets()`, `_in_notebook()`.

No imports from any other MutaGenAI module — fully standalone.

## Dependencies

| Group | Packages | When needed |
|---|---|---|
| Core | `numpy>=1.26` | Always |
| LLM | `httpx>=0.27`, `azure-identity>=1.15` | Any LLM backend call |
| Viz | `matplotlib>=3.8`, `plotly>=5.18` | Dashboard plots |
| Wizard | `rich>=13.0` | Pretty CLI wizard |

Install groups: `pip install MutaGenAI[llm]`, `[viz]`, `[wizard]`,
`[all]`.

## Build and test

```bash
uv sync                        # install all deps including dev
uv run pytest tests/           # 109 tests, ~72 s
uv run pytest tests/ -x -q     # fail-fast, quiet
```

Build system: **hatchling**. Python >=3.11.

## Coding conventions

- All LLM-dependent imports are **lazy** — `httpx` and `azure-identity`
  are imported inside functions, not at module level. This keeps the core
  engine importable without network dependencies.
- Optional viz deps (`plotly`, `matplotlib`) are detected at call time
  via `_has_plotly()` / `_has_matplotlib()`. Functions degrade gracefully.
- The wizard generates code with `from MutaGenAI import ...` — never
  `from evosim`.
- No circular imports. Dependency direction:
  `strategies → prompt_evolver`, `wizard → prompt_evolver + strategies`,
  `seed_loader → (none)`, `dashboard → (none)`.
- Tests are self-contained — they mock all LLM calls. No network access
  required.

## Agent workflow rules

When working on this codebase, follow these rules:

### Before editing

1. **Read the module you're modifying.** Each of the 5 modules has a
   specific responsibility. Don't mix concerns.
2. **Check the import graph.** `dashboard.py` and `seed_loader.py` have
   no internal imports — keep them that way.
3. **Run the tests** before and after changes: `uv run pytest tests/ -x`.

### Adding a new scoring strategy

1. Add the class to `strategies.py` as a subclass of `Scorer`.
2. Export it from `__init__.py` and add to `__all__`.
3. Add tests in `tests/test_strategies.py`.
4. Update the wizard's strategy selection in `wizard.py` if it should be
   user-facing.

### Adding a new benchmark dashboard

1. Add three functions to `dashboard.py`:
   - `plot_<name>_evolution(log_path, ...)` — public entry point
   - `_plot_<name>_plotly(data, ...)` — Plotly implementation
   - `_plot_<name>_mpl(data, ...)` — Matplotlib fallback
2. Follow the existing pattern: load JSON, extract series, render.
3. No imports from other MutaGenAI modules.

### Adding a new cookbook example

1. Create `examples/cookbook/prompt_evolution_<name>.py`.
2. Use `from MutaGenAI import ...` — never `from MutaGenAI.prompt_evolver`.
3. Each example should be runnable standalone:
   `uv run python examples/cookbook/prompt_evolution_<name>.py`.
4. Save results to a JSON log that the dashboard can consume.

### Adding a new LLM backend

1. Add the enum value to `LLMBackend` in `prompt_evolver.py`.
2. Add the client logic to `LLMClient` with lazy imports.
3. Update the wizard's backend selection in `wizard.py`.
4. Add tests that mock the HTTP calls.

### Seed templates

1. Create `seed_templates/<name>.json` with the schema:
   `{"name": "...", "description": "...", "seeds": ["...", ...]}`.
2. Design seeds across diverse structural archetypes (direct, CoT,
   persona, definitional, minimalist, output-strict, contrastive).
3. Set `population_size >= len(seeds)` so every archetype enters the
   initial gene pool.

## What NOT to change

- **Don't add EvoSim imports.** This is a standalone package. Any
  reference to `evosim` is a bug.
- **Don't add EA engine features** (pareto fronts, map-elites, surrogate
  models, landscape analysis). Those belong in EvoSim.
- **Don't make `dashboard.py` import from other MutaGenAI modules.** It
  must stay self-contained for independent use.
- **Don't add hard dependencies beyond numpy.** All other deps must
  remain optional with lazy imports and graceful fallbacks.

## Verified benchmark claims

The README contains benchmark results from six agentic benchmarks.
These numbers were produced by running the cookbook scripts against
real datasets. Key verified claims:

| Benchmark | Evolved (Ollama 3 B) | Default | Gain |
|---|---:|---:|---:|
| BFCL simple_python | 100.0 % | 98.4 % | +1.5 % |
| xLAM data | 100.0 % | 46.3 % | +53.7 % |
| API-Bank level_1 | 100.0 % | 52.3 % | +47.7 % |
| ToolBench G1 | 89.5 % | 16.0 % | +73.5 % |
| τ-bench airline (deep) | 46.6 % | 30.0 % | +16.6 % |
| Browser failure_recovery (deep) | 35.2 % | 24.3 % | +10.9 % |

No-eval Composite strategy reaches 82.3 % on API-Bank (vs 55.0 %
default) and 95.5 % on xLAM (vs 95.4 % default) — without labels.

## Quick reference — public API

```python
# Ground-truth evolution
from MutaGenAI import PromptEvolver, PromptEvolverConfig, Tool, EvalSample, LLMBackend

# No-eval evolution
from MutaGenAI import NoEvalPromptEvolver, NoEvalConfig
from MutaGenAI import LLMJudge, SelfConsistencyScorer, ProxyMetricsScorer, CompositeScorer
from MutaGenAI import ToolSuccessScorer, PreferenceScorer, HumanTournament
from MutaGenAI import SyntheticEvalGenerator, SyntheticEvalScorer

# Seed templates
from MutaGenAI import load_seed_templates, list_seed_templates

# Dashboard
from MutaGenAI.dashboard import (
    plot_bfcl_evolution,
    plot_tau_bench_evolution,
    plot_xlam_evolution,
    plot_toolbench_evolution,
    plot_apibank_evolution,
    plot_browser_agent_evolution,
)

# Wizard
from MutaGenAI import run_wizard
```
