You are a product owner defining a development roadmap for EvoSim — a
state-of-the-art evolutionary programming simulator and optimisation engine.

EvoSim is designed to be the greatest and most performant evolutionary
optimisation solution. Users bring ANY problem (continuous, discrete,
combinatorial, mixed, single/multi-objective, constrained) and EvoSim finds
the best approach. Features an LLM-powered advisor via local Ollama models
(llama3.2, llama3.1, qwen3) for problem analysis, algorithm recommendation,
hyperparameter tuning, and result interpretation. Stack: Python, NumPy,
Pydantic, optional GPU/Ray parallelism, CLI + optional FastAPI, Ollama HTTP API.

Key paths: evosim/ (main package), evosim/algorithms/ (GA, NSGA-II/III, DE,
SHADE, CMA-ES, PSO, SNES, xNES, OpenAI-ES, Bayesian, Annealing, hybrids),
evosim/operators/ (crossover, mutation, selection — composable),
evosim/problem.py (problem definition, search spaces, constraints),
evosim/advisor.py (LLM advisor via Ollama), evosim/benchmarks/ (standard
test functions), evosim/parallel.py (multiprocessing, Ray, GPU),
evosim/logging.py (metrics, convergence plots), evosim/persistence.py
(experiment save/load/compare), evosim/cli.py (CLI), tests/ (pytest).

Based on the following assessment of the codebase at {local_path}:

--- ASSESSMENT ---
{assessment}
--- END ASSESSMENT ---

Create a prioritised roadmap of improvements. For each item:
1. Title (short, actionable)
2. Priority (P0 = critical, P1 = high, P2 = medium, P3 = low)
3. Category: **FUNCTIONAL** or **NON-FUNCTIONAL**
4. Effort estimate (S / M / L / XL)
5. Description of what needs to be done
6. Acceptance criteria

**Balance guideline**: At least 80% of roadmap items MUST be functional
improvements: new algorithms, operator implementations, problem space support,
LLM advisor features, benchmark functions, performance optimisations,
experiment workflow, visualisation, CLI UX, API design.

Non-functional items (tests, docs, refactors, security) should be no more
than 20%.

Prioritisation guidance for EvoSim:
- P0: Anything that prevents a user from defining a problem and running an
  optimisation (broken install, no working algorithm, no problem definition
  API, fitness evaluation failures)
- P1: Core algorithm gaps (missing CMA-ES, missing DE/SHADE, no multi-
  objective support, no constraint handling, no LLM advisor, no benchmarks)
- P2: Advanced features (Bayesian optimisation, hybrid methods, experiment
  comparison, GPU acceleration, advanced operators, visualisation)
- P3: Polish (docs, performance tuning, additional benchmarks, edge cases)

Inspiration from best-in-class libraries:
- EvoTorch: GPU-accelerated, CMA-ES/SNES/xNES/PGPE, MAP-Elites, GymNE
- PyGAD: Easy API, lifecycle callbacks, multi-objective, Keras/PyTorch integration
- Google Vizier: Bayesian optimisation, study config, distributed client-server
- eaopt: Composable models (generational, steady-state, ring), speciation, migration
- PySR: Symbolic regression, Pareto-optimal expressions
- DEAP: Rich operator library, multi-objective, coevolution, GP

North star: make EvoSim the most powerful yet easiest-to-use evolutionary
optimisation engine — adaptable to any problem, performant at scale, with
intelligent LLM-powered guidance using free local Ollama models.

Order by priority then effort (smallest first for quick wins).
Return at most 15 items.
Format each item clearly so it can be converted into a prompt.
