#!/usr/bin/env python3
"""Model migration demo: llama3.2:latest -> qwen3:8b on entity classification.

Seeds evolution with the experiment's known winning prompt, measures the
three-way migration (A_old / A_transfer / A_evolved) with the Phase 1-2
migration utilities, and reports which samples regressed.

Run:  python examples/experiments/entity_classification/migrate_llama_to_qwen.py
Needs: Ollama with `llama3.2` and `qwen3:8b` pulled; network for the dataset.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from MutaGenAI.prompt_evolver import (
    EvalSample,
    LLMBackend,
    PromptEvolver,
    PromptEvolverConfig,
    ProblemType,
    Tool,
)
from MutaGenAI.migration import MigrationReport, evaluate_prompt, make_client

from examples.experiments.entity_classification import ENTITY_TYPES

# The known-good prompt for the previous model (llama3.2), from the experiment.
WINNING_PROMPT = """\
You are an AI assistant.

## Task
You are an agent that classifies inbound text into one of Agent, Task, Tool, Input, Output, Human

## Output
Provide a clear, direct response."""

SRC_MODEL = "llama3.2"
TGT_MODEL = "qwen3:8b"
N = int(os.environ.get("N", "24"))
ITERATIONS = int(os.environ.get("ITERATIONS", "2"))
POPULATION = int(os.environ.get("POPULATION", "2"))
ISLANDS = int(os.environ.get("ISLANDS", "1"))
# EARLY_STOP: "preserve" stops once A_old is matched; "off" runs all generations.
EARLY_STOP = os.environ.get("EARLY_STOP", "preserve")
TEMP, TOP_P = 0.7, 0.95


def load_samples(n: int) -> list[EvalSample]:
    from datasets import load_dataset

    ds = load_dataset("holistic-ai/entity-classification-agentic-ai")
    rows = [
        {"content": r["content"], "expected": r["expected_entity"]}
        for r in ds["validation"]
    ]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [EvalSample(rows[int(i)]["content"], rows[int(i)]["expected"]) for i in idx]


def main() -> None:
    tools = [Tool(e, f"The '{e}' entity type") for e in ENTITY_TYPES]
    samples = load_samples(N)
    print(f"Loaded {len(samples)} eval samples\n")

    src = make_client(SRC_MODEL, LLMBackend.OLLAMA, max_tokens=10)
    # qwen3 is a reasoning model: disable the think phase for clean labels.
    tgt = make_client(
        TGT_MODEL, LLMBackend.OLLAMA, max_tokens=16, ollama_think=False,
        timeout=120.0,
    )
    if not src.is_available() or not tgt.is_available():
        print("Ollama or a required model is not reachable.")
        return

    print("[1/4] A_old — llama3.2 + winning prompt")
    a_old = evaluate_prompt(
        WINNING_PROMPT, tools, samples, src, temperature=TEMP, top_p=TOP_P
    )
    print(f"      {a_old.accuracy:.1%} ({a_old.num_correct}/{a_old.total})\n")

    print("[2/4] A_transfer — qwen3:8b + winning prompt (naive swap)")
    a_transfer = evaluate_prompt(
        WINNING_PROMPT, tools, samples, tgt, temperature=TEMP, top_p=TOP_P
    )
    print(f"      {a_transfer.accuracy:.1%} "
          f"({a_transfer.num_correct}/{a_transfer.total})\n")

    print("[3/4] Evolve on qwen3:8b (warm-started from winning prompt)…")
    early = None if EARLY_STOP == "off" else a_old.accuracy * 100.0
    config = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=TGT_MODEL,
        ollama_think=False,
        max_tokens=16,
        timeout=120.0,
        iterations=ITERATIONS,
        population_size=POPULATION,
        num_islands=ISLANDS,
        problem_type=ProblemType.CLASSIFICATION,
        early_stop_score=early,  # preserve the old-model bar, or None to optimize
    )
    evolver = PromptEvolver(
        tools, samples, config, seed_templates=[WINNING_PROMPT], verbose=True
    )
    t0 = time.perf_counter()
    result = evolver.run()
    print(f"      evolved in {time.perf_counter() - t0:.0f}s "
          f"(generations run: {result.iterations_run})\n")

    print("[4/4] A_evolved — qwen3:8b + evolved prompt")
    a_evolved = evaluate_prompt(
        result.best_prompt, tools, samples, tgt,
        temperature=result.best_temperature, top_p=result.best_top_p,
    )
    print(f"      {a_evolved.accuracy:.1%} "
          f"({a_evolved.num_correct}/{a_evolved.total})\n")

    report = MigrationReport.build(
        source_eval=a_old,
        transfer_eval=a_transfer,
        evolved_eval=a_evolved,
        source_model=f"ollama:{SRC_MODEL}",
        target_model=f"ollama:{TGT_MODEL}",
    )
    print(report.summary())
    print("\nEvolved prompt:\n" + "-" * 56 + f"\n{result.best_prompt}\n" + "-" * 56)

    out = Path(_root) / "logs" / "migration_llama_to_qwen_entity.json"
    out.write_text(
        json.dumps(
            {
                "source_model": report.source_model,
                "target_model": report.target_model,
                "a_old": report.a_old,
                "a_transfer": report.a_transfer,
                "a_evolved": report.a_evolved,
                "delta_vs_old": report.delta_vs_old,
                "delta_vs_transfer": report.delta_vs_transfer,
                "preserved": report.preserved,
                "transfer_regressions": report.transfer_regressions,
                "recovered": report.recovered,
                "remaining_regressions": report.remaining_regressions,
                "best_temperature": result.best_temperature,
                "best_top_p": result.best_top_p,
                "iterations_run": result.iterations_run,
                "evolved_prompt": result.best_prompt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
