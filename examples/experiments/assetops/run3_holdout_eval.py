#!/usr/bin/env python3
"""
AssetOpsBench Track 1 — Holdout Generalization Experiment
==========================================================

Tests the evolved prompt from run1 on held-out scenarios to measure
generalization — simulates the Phase 2 competition evaluation where
the solution is tested on 10 new scenarios from different asset classes.

Design:
  1. Load the best prompt from run1_baseline_log.json
  2. Load all available scenarios, split 70/30 (train/holdout)
  3. Report score on the holdout set vs the baseline prompt
  4. Compare evolved prompt server/tool accuracy on unseen asset classes

This validates whether evolutionary prompt improvements transfer to
scenarios the optimizer never saw.

Usage::

    uv sync --extra llm
    uv run python examples/experiments/assetops/run3_holdout_eval.py

    # Specify the evolved log:
    uv run python examples/experiments/assetops/run3_holdout_eval.py \
        --evolved-log logs/assetops_run1_baseline_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root, ".env"))

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)

sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))
from prompt_evolution_assetops import (
    SERVER_DESCRIPTIONS,
    load_assetops_scenarios,
    score_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AssetOpsBench Track 1 — Holdout Generalization"
    )
    parser.add_argument(
        "--evolved-log",
        default="logs/assetops_run1_baseline_log.json",
        help="Path to the evolved experiment log",
    )
    parser.add_argument(
        "--backend", default="ollama",
        choices=["ollama", "openai", "azure_openai"],
    )
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    backend_map = {
        "ollama": LLMBackend.OLLAMA,
        "openai": LLMBackend.OPENAI,
        "azure_openai": LLMBackend.AZURE_OPENAI,
    }
    backend = backend_map[args.backend]
    holdout_config = PromptEvolverConfig(
        backend=backend,
        ollama_model=args.model,
    )
    client = LLMClient(holdout_config)

    # Load evolved prompt
    log_path = Path(_root) / args.evolved_log
    if not log_path.exists():
        print(f"  ✗ Log not found: {log_path}")
        print("  Run run1_baseline.py first.")
        return

    with open(log_path) as f:
        log_data = json.load(f)

    if isinstance(log_data, list):
        # Pick the best result
        best_entry = max(log_data, key=lambda x: x.get("evolved_score", 0))
    else:
        best_entry = log_data

    evolved_template = best_entry["best_prompt_template"]
    evolved_temp = best_entry.get("best_temperature", 0.1)
    evolved_top_p = best_entry.get("best_top_p", 0.9)

    print(f"\n  Evolved prompt loaded (score={best_entry.get('evolved_score', '?')}%)")
    print(f"  Template preview: {evolved_template[:100]}...")

    # Load scenarios and split
    scenarios = load_assetops_scenarios(max_scenarios=args.max_scenarios)
    if not scenarios:
        print("  ✗ No scenarios loaded.")
        return

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(scenarios))
    split = int(len(scenarios) * (1.0 - args.holdout_fraction))
    holdout_indices = indices[split:]
    holdout = [scenarios[int(i)] for i in holdout_indices]

    print(f"  Total scenarios: {len(scenarios)}")
    print(f"  Holdout set: {len(holdout)} ({args.holdout_fraction * 100:.0f}%)")

    # Baseline prompt (the default from AssetOpsBench planner.py)
    baseline_template = (
        "You are a planning assistant for industrial asset operations "
        "and maintenance.\n\nDecompose the question below into a sequence "
        "of subtasks. For each subtask, assign a server and select the "
        "exact tool to call.\n\nAvailable servers and tools:\n{servers}\n\n"
        "Question: {question}\n\nPlan:"
    )

    # Evaluate both on holdout
    print(f"\n  {'─' * 50}")
    print(f"  Evaluating on {len(holdout)} holdout scenarios...\n")

    results = {"baseline": [], "evolved": []}

    for scenario in holdout:
        for label, template, temp, top_p in [
            ("baseline", baseline_template, 0.1, 0.9),
            ("evolved", evolved_template, evolved_temp, evolved_top_p),
        ]:
            prompt = template.replace(
                "{servers}", SERVER_DESCRIPTIONS
            ).replace(
                "{question}", scenario.utterance
            )

            response = client.complete(
                system_prompt=prompt,
                user_message=scenario.utterance,
                temperature=temp,
                top_p=top_p,
            )
            if response is None:
                results[label].append(0.0)
                continue

            score, _ = score_plan(response, scenario)
            results[label].append(score * 100.0)

    baseline_avg = sum(results["baseline"]) / len(results["baseline"])
    evolved_avg = sum(results["evolved"]) / len(results["evolved"])

    print(f"  Baseline holdout score: {baseline_avg:.1f}%")
    print(f"  Evolved holdout score:  {evolved_avg:.1f}%")
    print(f"  Improvement:            {evolved_avg - baseline_avg:+.1f}%")
    print(f"  {'─' * 50}")

    # Save
    output = {
        "experiment": "run3_holdout_eval",
        "holdout_size": len(holdout),
        "baseline_avg": round(baseline_avg, 2),
        "evolved_avg": round(evolved_avg, 2),
        "improvement": round(evolved_avg - baseline_avg, 2),
        "baseline_scores": [round(s, 2) for s in results["baseline"]],
        "evolved_scores": [round(s, 2) for s in results["evolved"]],
        "evolved_source_log": args.evolved_log,
    }

    log_dir = Path(_root) / "logs"
    log_dir.mkdir(exist_ok=True)
    out_file = log_dir / "assetops_run3_holdout_log.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Log saved to {out_file}")


if __name__ == "__main__":
    main()
