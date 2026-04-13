# EvoSim Constitution — First Principles

These principles govern ALL decisions made by the Operator and Agent.
They are non-negotiable and must never be violated.

## 1. User-First Design
EvoSim must be trivially easy to use. A user with zero evolutionary
computation experience must be able to define a problem and get results
within minutes. Sensible defaults everywhere. Progressive complexity —
simple API for simple problems, full control for experts.

## 2. Problem-Agnostic Architecture
EvoSim adapts to the user's problem, not the other way around. Support
continuous, discrete, categorical, mixed, and combinatorial search
spaces. Single-objective, multi-objective, and constrained optimization.
The user brings the problem; EvoSim finds the best approach.

## 3. Nature-Inspired, Science-Backed
Every algorithm must be grounded in peer-reviewed research. Implement
state-of-the-art variants (CMA-ES, SHADE, NSGA-III, xNES, Bayesian
Optimization) not just textbook versions. Performance matters — use
NumPy vectorization and optional GPU acceleration.

## 4. LLM-Powered Intelligence (Ollama Only)
The advisor system uses local Ollama models (llama3.2, llama3.1,
qwen3) to analyse problems, recommend algorithms, tune hyperparameters,
and interpret results. No paid API dependencies. Graceful fallback to
heuristic recommendations when Ollama is unavailable.

## 5. Experiment-Driven Workflow
Users must be able to compare algorithms side-by-side, track experiment
history, reproduce runs with seeds, and visualise convergence. Every
run is logged and saveable. The system learns from experiment history
to improve future recommendations.

## 6. Performance at Scale
Parallel fitness evaluation via multiprocessing. Optional GPU
acceleration for large populations. Vectorized operations throughout.
Must handle populations of 10,000+ and dimensions of 1,000+ efficiently.

## 7. Composability
Operators (crossover, mutation, selection) are mix-and-match. Algorithms
can be composed with custom operators. Users can define custom operators,
custom algorithms, and custom search spaces without modifying core code.

## 8. No Magic, Full Transparency
Every decision the system makes (algorithm selection, hyperparameter
choices, operator selection) must be explainable. Logging at every
level. The user always knows what's happening and why.

## 9. Robust Testing
Every algorithm must have correctness tests against known benchmark
functions. Integration tests for the full pipeline. Property-based
tests for operators. Regression tests for performance.

## 10. Clean Python, No Bloat
Pure Python with NumPy. Minimal dependencies. Type hints everywhere.
Pydantic for configuration validation. No framework lock-in.
Optional dependencies (matplotlib, ray, torch) are truly optional.
