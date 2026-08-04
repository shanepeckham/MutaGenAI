#!/usr/bin/env python3
"""Model migration demo: llama3.2:latest -> qwen3:8b on API-Bank (level 1).

Seeds evolution with the experiment's llama3.2 winning prompt (run2), measures
the three-way migration (A_old / A_transfer / A_evolved) with the Phase 1-2
migration utilities, and reports which cases regressed.

Framing: API-Bank normally injects the per-case API description + conversation
into the system prompt via placeholders. Here the winning prompt stays as the
system prompt (the rules) and each case's description + conversation are sent
as the user message, with the candidate API names as `tools`. All three
anchors use this identical framing, so the comparison is internally consistent.

Run:  python examples/experiments/apibank/migrate_llama_to_qwen.py
Needs: Ollama with `llama3.2` and `qwen3:8b`; API-Bank cache in .apibank_cache/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

from examples.cookbook.prompt_evolution_apibank import load_apibank_dataset

# llama3.2 winning prompt for API-Bank level 1 (from apibank_run2 experiment).
WINNING_PROMPT = """\
You are an API-calling assistant. Given a conversation and API
descriptions, generate the correct API request in the format
[ApiName(key1='value1', key2='value2', ...)].

descriptions, generate the correct API request in the format

Rules:
- Output EXACTLY ONE API call in the bracket format shown above.
Consider the user's full intent before choosing an API.
- Match API names exactly as described.
- Extract parameter values from the conversation — do NOT invent values.
- Include ALL required parameters.
- Use single quotes around parameter values.
- Output ONLY the API call, nothing else.

Respond accurately with the correct API call."""

SRC_MODEL = "llama3.2"
TGT_MODEL = os.environ.get("TGT_MODEL", "qwen3:8b")
# THINK: "false"/"true" set Ollama's think flag; "none" omits it (for
# non-reasoning targets like llama3.2:1b).
_THINK = {"false": False, "true": True}.get(
    os.environ.get("THINK", "false").lower(), None
)
N = int(os.environ.get("N", "24"))
ITERATIONS = int(os.environ.get("ITERATIONS", "2"))
POPULATION = int(os.environ.get("POPULATION", "2"))
ISLANDS = int(os.environ.get("ISLANDS", "1"))
# EARLY_STOP: "preserve" stops once A_old is matched; "off" runs all generations.
EARLY_STOP = os.environ.get("EARLY_STOP", "preserve")
TEMP, TOP_P = 0.7, 0.95


def build_eval() -> tuple[list[Tool], list[EvalSample]]:
    by_level = load_apibank_dataset(max_per_level=N, levels=["level_1"])
    cases = by_level.get("level_1", [])
    samples: list[EvalSample] = []
    names: set[str] = set()
    for c in cases:
        api = c.expected_api_name
        if not api:
            continue
        names.add(api)
        query = f"{c.instruction}\n\n{c.input_text}\n\nGenerate API Request:"
        samples.append(EvalSample(query, api, c.expected_params))
    tools = [Tool(n, f"API endpoint {n}") for n in sorted(names)]
    return tools, samples


def main() -> None:
    tools, samples = build_eval()
    print(f"Loaded {len(samples)} API-Bank level-1 cases "
          f"({len(tools)} distinct APIs)\n")

    src = make_client(SRC_MODEL, LLMBackend.OLLAMA, max_tokens=64)
    tgt = make_client(
        TGT_MODEL, LLMBackend.OLLAMA, max_tokens=64, ollama_think=_THINK,
        timeout=120.0,
    )
    if not src.is_available() or not tgt.is_available():
        print("Ollama or a required model is not reachable.")
        return

    print(f"[1/4] A_old — {SRC_MODEL} + winning prompt")
    a_old = evaluate_prompt(
        WINNING_PROMPT, tools, samples, src, temperature=TEMP, top_p=TOP_P
    )
    print(f"      {a_old.accuracy:.1%} ({a_old.num_correct}/{a_old.total})\n")

    print(f"[2/4] A_transfer — {TGT_MODEL} + winning prompt (naive swap)")
    a_transfer = evaluate_prompt(
        WINNING_PROMPT, tools, samples, tgt, temperature=TEMP, top_p=TOP_P
    )
    print(f"      {a_transfer.accuracy:.1%} "
          f"({a_transfer.num_correct}/{a_transfer.total})\n")

    print(f"[3/4] Evolve on {TGT_MODEL} (warm-started from winning prompt)…")
    early = None if EARLY_STOP == "off" else a_old.accuracy * 100.0
    config = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=TGT_MODEL,
        ollama_think=_THINK,
        max_tokens=64,
        timeout=120.0,
        iterations=ITERATIONS,
        population_size=POPULATION,
        num_islands=ISLANDS,
        problem_type=ProblemType.TOOL_ROUTING,
        early_stop_score=early,
    )
    evolver = PromptEvolver(
        tools, samples, config, seed_templates=[WINNING_PROMPT], verbose=True
    )
    t0 = time.perf_counter()
    result = evolver.run()
    print(f"      evolved in {time.perf_counter() - t0:.0f}s "
          f"(generations run: {result.iterations_run})\n")

    print(f"[4/4] A_evolved — {TGT_MODEL} + evolved prompt")
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

    safe_tgt = TGT_MODEL.replace(":", "_").replace("/", "_")
    out = Path(_root) / "logs" / f"migration_llama_to_{safe_tgt}_apibank.json"
    out.write_text(
        json.dumps(
            {
                "source_model": report.source_model,
                "target_model": report.target_model,
                "level": "level_1",
                "n_cases": len(samples),
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
