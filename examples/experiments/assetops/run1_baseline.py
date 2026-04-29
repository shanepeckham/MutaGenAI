#!/usr/bin/env python3
"""
AssetOpsBench Track 1 — Baseline Experiment: Ground-Truth Evolution
=====================================================================

Establishes baseline results for Track 1 planning prompt evolution
using ground-truth scoring against AssetOpsBench scenarios from
HuggingFace.

Runs both standard and deep configurations, saves results for
comparison with subsequent no-eval and adaptive experiments.

Design:
  - standard: 3 iterations, pop 6, 2 islands
  - deep:     5 iterations, pop 8, 3 islands
  - Scoring:  format (0.15) + server (0.30) + tool (0.30)
              + dependency (0.15) + completeness (0.10)
  - Seeds:    6 diverse archetypes from assetops_planning.json

Usage::

    uv sync --extra llm
    uv run python examples/experiments/assetops/run1_baseline.py

    # Deep mode:
    uv run python examples/experiments/assetops/run1_baseline.py --deep

    # Azure OpenAI backend:
    uv run python examples/experiments/assetops/run1_baseline.py --backend azure_openai --model gpt-4.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root, ".env"))

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptEvolverConfig,
)

# Reuse the cookbook module
sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))
from prompt_evolution_assetops import (
    load_assetops_scenarios,
    run_assetops_evolution,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AssetOpsBench Track 1 — Baseline Ground-Truth Experiment"
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="Also run the deep configuration",
    )
    parser.add_argument(
        "--backend", default="ollama",
        choices=["ollama", "openai", "azure_openai"],
    )
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--max-scenarios", type=int, default=50)
    args = parser.parse_args()

    backend_map = {
        "ollama": LLMBackend.OLLAMA,
        "openai": LLMBackend.OPENAI,
        "azure_openai": LLMBackend.AZURE_OPENAI,
    }
    backend = backend_map[args.backend]

    scenarios = load_assetops_scenarios(max_scenarios=args.max_scenarios)
    if not scenarios:
        print("  ✗ No scenarios loaded. Exiting.")
        return

    print(f"\n  Loaded {len(scenarios)} scenarios")
    print(f"  Backend: {args.backend} / {args.model}")

    configs = {
        "standard": PromptEvolverConfig(
            iterations=3,
            population_size=6,
            num_islands=2,
            backend=backend,
            ollama_model=args.model,
        ),
    }
    if args.deep:
        configs["deep"] = PromptEvolverConfig(
            iterations=5,
            population_size=8,
            num_islands=3,
            backend=backend,
            ollama_model=args.model,
        )

    all_results = []

    for alg_name, config in configs.items():
        print(f"\n  Running {alg_name} configuration...")
        client = LLMClient(config)
        experiment = run_assetops_evolution(
            scenarios=scenarios,
            client=client,
            config=config,
            algorithm_name=alg_name,
        )
        all_results.append({
            "experiment": "run1_baseline",
            "category": experiment.category,
            "algorithm": experiment.algorithm,
            "backend": experiment.backend,
            "n_scenarios": experiment.n_scenarios,
            "baseline_score": round(experiment.baseline_score, 2),
            "evolved_score": round(experiment.evolved_score, 2),
            "improvement": round(
                experiment.evolved_score - experiment.baseline_score, 2
            ),
            "server_accuracy": round(experiment.server_accuracy, 2),
            "tool_accuracy": round(experiment.tool_accuracy, 2),
            "best_temperature": round(experiment.best_temperature, 4),
            "best_top_p": round(experiment.best_top_p, 4),
            "iterations": experiment.iterations,
            "wall_time": round(experiment.wall_time, 1),
            "history": experiment.history,
            "prompt_evolution": experiment.prompt_evolution,
            "best_prompt_template": experiment.best_prompt_template,
        })

    log_path = Path(_root) / "logs"
    log_path.mkdir(exist_ok=True)
    out_file = log_path / "assetops_run1_baseline_log.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Log saved to {out_file}")


if __name__ == "__main__":
    main()
