#!/usr/bin/env python3
"""
AssetOpsBench Track 1 — No-Eval Composite Experiment
======================================================

Tests no-eval prompt evolution for Track 1 planning prompts using
a Composite scorer (LLMJudge + ProxyMetrics + SelfConsistency).

This mirrors the Phase 2 generalization scenario where the challenge
evaluates on 10 unseen scenarios from new asset classes — no ground
truth is available.

Design:
  - LLMJudge (0.5):  scores plan quality against a rubric
  - ProxyMetrics (0.3): structural checks (#TaskN, #ServerN, #ToolN tags)
  - SelfConsistency (0.2): agreement across 3 runs
  - Population: 8, Islands: 2, Iterations: 5

Prior art (API-Bank no-eval): Composite reached 82.3% GT score
without labels, vs 55.0% default.

Usage::

    uv sync --extra llm
    uv run python examples/experiments/assetops/run2_noeval_composite.py

    # More iterations:
    uv run python examples/experiments/assetops/run2_noeval_composite.py --iterations 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
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

sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))
from prompt_evolution_assetops import run_assetops_no_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AssetOpsBench Track 1 — No-Eval Composite Experiment"
    )
    parser.add_argument(
        "--backend", default="ollama",
        choices=["ollama", "openai", "azure_openai"],
    )
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    args = parser.parse_args()

    backend_map = {
        "ollama": LLMBackend.OLLAMA,
        "openai": LLMBackend.OPENAI,
        "azure_openai": LLMBackend.AZURE_OPENAI,
    }
    backend = backend_map[args.backend]

    config = PromptEvolverConfig(
        iterations=args.iterations,
        population_size=args.population,
        num_islands=2,
        backend=backend,
        ollama_model=args.model,
    )
    client = LLMClient(config)

    print(f"\n  Backend: {args.backend} / {args.model}")
    print(f"  Config: {args.iterations} iterations, pop={args.population}")

    result = run_assetops_no_eval(client, config)

    # Add experiment metadata
    result["experiment"] = "run2_noeval_composite"
    result["backend"] = args.backend
    result["model"] = args.model
    result["iterations"] = args.iterations
    result["population_size"] = args.population

    log_path = Path(_root) / "logs"
    log_path.mkdir(exist_ok=True)
    out_file = log_path / "assetops_run2_noeval_log.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Log saved to {out_file}")


if __name__ == "__main__":
    main()
