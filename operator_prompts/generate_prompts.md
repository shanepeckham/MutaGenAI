You are converting a roadmap into actionable prompts for a self-improving code
agent working on EvoSim — a state-of-the-art evolutionary programming simulator
and optimisation engine inspired by nature.

EvoSim implements nature-inspired optimisation algorithms (GA, DE, CMA-ES, PSO,
NES, Bayesian, Annealing, hybrids) with an LLM-powered advisor (Ollama).
Key paths: evosim/ (main package), evosim/algorithms/ (all optimisers with
base.py interface), evosim/operators/ (crossover, mutation, selection —
composable), evosim/problem.py (problem definition, search spaces, constraints),
evosim/advisor.py (LLM advisor via Ollama — llama3.2/llama3.1/qwen3),
evosim/benchmarks/ (continuous + combinatorial test functions),
evosim/parallel.py (multiprocessing, optional Ray/GPU), evosim/logging.py
(metrics, convergence visualisation), evosim/persistence.py (experiment
save/load/compare), evosim/cli.py (CLI), tests/ (pytest).

--- ROADMAP ---
{roadmap}
--- END ROADMAP ---

For each roadmap item, write a single, self-contained prompt that instructs the
agent to implement the improvement. The prompt must:
- Be specific and actionable — tell the agent exactly what to do
- Reference EvoSim file paths where relevant
- Remind the agent that LLM advisor calls go through Ollama only (no paid APIs)
  and must gracefully fall back to heuristic recommendations when Ollama is
  unavailable
- Emphasise ease-of-use: sensible defaults, zero-config quick start, progressive
  complexity for experts
- Note that all algorithms must implement the base optimizer interface in
  evosim/algorithms/base.py (minimize/maximize, step, best_solution, history)
- Note that operators must be composable and user-extensible
- Note that search spaces must support continuous, discrete, categorical, mixed,
  and permutation types
- Note that multi-objective support (Pareto fronts) is required for GA variants
- Include acceptance criteria so the agent knows when the task is done
- Be at most 500 characters

**Balance rule**: The resulting prompt list MUST contain a mix of functional and
non-functional work. At least 80% of prompts must be functional improvements and
no more than 20% non-functional. If the roadmap is dominated by non-functional
items, convert prompts into functional improvements by:
- Implementing new optimisation algorithms (CMA-ES, SHADE, NSGA-III, xNES)
- Adding new genetic operators (SBX crossover, polynomial mutation, PMX)
- Building the LLM advisor for problem analysis and algorithm recommendation
- Implementing benchmark functions and comparison tools
- Adding search space types (permutation, tree, graph)
- Building experiment workflow (history, comparison, visualisation)
- Implementing constraint handling (penalty, repair, feasibility rules)
- Adding parallel evaluation (multiprocessing, Ray workers)
- Building the CLI with rich output and progress bars

Functional prompts should appear BEFORE non-functional prompts at the same
priority level.

Return ONLY the prompts, one per line. No numbering, no blank lines, no commentary.
Prefix each prompt with its priority and category in brackets,
e.g. [P0 FUNCTIONAL], [P1 NON-FUNCTIONAL], etc.
Order from highest to lowest priority, with functional items first within each tier.
