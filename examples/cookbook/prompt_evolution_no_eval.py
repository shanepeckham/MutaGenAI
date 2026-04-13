#!/usr/bin/env python3
"""
Cookbook Recipe 48 — No-Eval Prompt Evolution: 7 Strategies
==========================================================

Evolves agent prompts **without labelled evaluation data** using seven
pluggable fitness strategies.  Each strategy provides a different signal
source for the evolutionary loop.

Strategies
----------
1. **LLM-as-Judge**        — A second LLM scores outputs against a rubric.
2. **Synthetic Eval**      — Auto-generate test cases from a description.
3. **Tool-Use Success**    — Use API return codes as fitness signals.
4. **Self-Consistency**    — Score prompts by output agreement across runs.
5. **Proxy Metrics**       — Structural checks (valid JSON, required fields).
6. **Preference Scoring**  — Score with good/bad output examples.
7. **Human Tournament**    — Human selects the best output per generation.

Each strategy runs a short island-model evolution (3 generations) on a
customer-service agent task so you can see the differences.

Usage::

    # Run all automated strategies (1-6):
    uv run python examples/cookbook/prompt_evolution_no_eval.py

    # Run ONLY the human-in-the-loop strategy (interactive):
    uv run python examples/cookbook/prompt_evolution_no_eval.py --human

    # Run a specific strategy by number:
    uv run python examples/cookbook/prompt_evolution_no_eval.py --strategy 1

    # Combine strategies with weights:
    uv run python examples/cookbook/prompt_evolution_no_eval.py --composite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from prompture.strategies import (
    CompositeScorer,
    HumanTournament,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    PreferencePair,
    PreferenceScorer,
    ProxyCheck,
    ProxyMetricsScorer,
    Scorer,
    SelfConsistencyScorer,
    SyntheticEvalGenerator,
    SyntheticEvalScorer,
    ToolResult,
    ToolSuccessScorer,
)
from prompture.prompt_evolver import LLMBackend, LLMClient, PromptEvolverConfig

# ─────────────────────────────────────────────────────────────────────────
# Shared task: Customer-service agent (no eval labels needed)
# ─────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = textwrap.dedent("""\
    You are a helpful customer-service agent for an online store.
    Answer customer questions about orders, returns, shipping, and
    account management.  Be polite, concise, and accurate.
    If you need to call a tool, respond with JSON:
    {"tool": "<name>", "parameters": {<key>: <value>}}
""").strip()

# Unlabelled test inputs — these have NO expected outputs
TEST_INPUTS = [
    "Where is my order #12345?",
    "I want to return the shoes I bought last week.",
    "How do I change my shipping address?",
    "Can I get a refund for order #67890?",
    "What's your return policy?",
    "My package arrived damaged, what should I do?",
    "I forgot my password and can't log in.",
    "Do you ship internationally?",
    "I was charged twice for the same order.",
    "Can I cancel my order before it ships?",
]


# ─────────────────────────────────────────────────────────────────────────
# Helper: run a strategy and print results
# ─────────────────────────────────────────────────────────────────────────


def run_strategy(
    name: str,
    scorer: Scorer,
    config: NoEvalConfig,
    test_inputs: list[str] | None = None,
) -> dict[str, Any]:
    """Run one strategy and return a result dict."""
    inputs = test_inputs or TEST_INPUTS
    print(f"\n{'━' * 62}")
    print(f"  Strategy: {name}")
    print(f"  Scorer:   {scorer.name()}")
    print(f"  Inputs:   {len(inputs)} (unlabelled)")
    print(f"{'━' * 62}\n")

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=inputs,
        scorer=scorer,
        config=config,
        seed=42,
        verbose=True,
    )

    t0 = time.perf_counter()
    result = evolver.run()
    wall = time.perf_counter() - t0

    print(f"\n  ✓ Best score: {result.best_score:.1f}%")
    print(f"    Temp: {result.best_temperature:.3f}  Top-p: {result.best_top_p:.3f}")
    print(f"    Wall time: {wall:.1f}s")
    print(f"\n  Best prompt (first 200 chars):")
    print(f"    {result.best_prompt[:200]}...")
    print()

    return {
        "strategy": name,
        "best_score": result.best_score,
        "best_temperature": result.best_temperature,
        "best_top_p": result.best_top_p,
        "wall_time": wall,
        "prompt_preview": result.best_prompt[:300],
        "iterations": result.iterations_run,
        "candidates_tried": len(result.all_candidates),
        "history": result.history,
    }


# ─────────────────────────────────────────────────────────────────────────
# Strategy 1: LLM-as-Judge
# ─────────────────────────────────────────────────────────────────────────


def demo_llm_judge(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 1 — Use a second LLM call to score against a rubric.

    The judge sees the task, input, and output, then rates on a rubric.
    No labelled data needed — just define what "good" looks like.
    """
    scorer = LLMJudge(
        rubric=textwrap.dedent("""\
            Score the response 0-10 on these criteria:
            - Helpfulness: Does it address the customer's actual need? (0-4)
            - Accuracy: Is the information correct and not hallucinated? (0-3)
            - Tone: Is it polite and professional? (0-2)
            - Conciseness: Is it direct without unnecessary filler? (0-1)
        """),
        max_score=10.0,
    )
    return run_strategy("1. LLM-as-Judge", scorer, config)


# ─────────────────────────────────────────────────────────────────────────
# Strategy 2: Synthetic Eval Generation
# ─────────────────────────────────────────────────────────────────────────


def demo_synthetic_eval(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 2 — Auto-generate test cases from the task description.

    An LLM creates input/output pairs, then evolution scores against
    those synthetic "expected" outputs.
    """
    print("\n  Generating synthetic eval cases...")
    llm_config = PromptEvolverConfig(backend=config.backend)
    client = LLMClient(llm_config)

    generator = SyntheticEvalGenerator(
        task_description=TASK_DESCRIPTION,
        num_cases=10,
    )
    cases = generator.generate(client)

    if not cases:
        print("  ⚠ Could not generate synthetic cases (LLM unavailable?)")
        print("  Using fallback hardcoded cases for demo purposes.")
        cases = [
            {"input": inp, "expected_output": f"Thank you for contacting us about '{inp[:30]}...'. Let me help you with that."}
            for inp in TEST_INPUTS[:5]
        ]

    print(f"  Generated {len(cases)} synthetic test cases:")
    for i, c in enumerate(cases[:3]):
        print(f"    [{i+1}] Input: {c['input'][:60]}...")
        print(f"        Expected: {c['expected_output'][:60]}...")
    if len(cases) > 3:
        print(f"    ... and {len(cases) - 3} more")

    scorer = SyntheticEvalScorer(cases)
    inputs = [c["input"] for c in cases]
    return run_strategy("2. Synthetic Eval", scorer, config, test_inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# Strategy 3: Tool-Use Success Signals
# ─────────────────────────────────────────────────────────────────────────


# Simulated tool executor for demo purposes
_KNOWN_TOOLS = {
    "lookup_order", "process_return", "update_address",
    "process_refund", "search_faq", "report_damage",
    "reset_password", "check_shipping", "cancel_order",
}


def _mock_tool_executor(tool_name: str, params: dict[str, Any]) -> ToolResult:
    """Simulated tool execution — returns success if tool is known and has params."""
    if tool_name not in _KNOWN_TOOLS:
        return ToolResult(success=False, return_code=404, output=f"Unknown tool: {tool_name}")
    if not params:
        return ToolResult(success=False, return_code=400, output="Missing parameters")
    return ToolResult(success=True, return_code=200, output=f"{tool_name} executed OK")


def demo_tool_success(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 3 — Score based on tool execution success/failure.

    The scorer parses tool calls from the model output and runs them
    through a (simulated) executor.  Real APIs give binary fitness
    signals for free.
    """
    scorer = ToolSuccessScorer(tool_executor=_mock_tool_executor)
    # Use inputs that should trigger tool calls
    tool_inputs = [
        "Look up order #12345 for me.",
        "Process a return for item SKU-999.",
        "Update my shipping address to 123 Main St.",
        "I want a refund for order #67890.",
        "Search the FAQ for return policy.",
        "Report damage on package PKG-456.",
        "Reset my password for user@example.com.",
        "Check if you ship to Canada.",
    ]
    return run_strategy("3. Tool-Use Success", scorer, config, test_inputs=tool_inputs)


# ─────────────────────────────────────────────────────────────────────────
# Strategy 4: Self-Consistency
# ─────────────────────────────────────────────────────────────────────────


def demo_self_consistency(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 4 — Score prompts by output agreement across multiple runs.

    Each prompt is run 3 times per input; consistent outputs score higher.
    Inconsistency usually signals ambiguity in the prompt.
    """
    scorer = SelfConsistencyScorer(num_samples=3)
    return run_strategy("4. Self-Consistency", scorer, config)


# ─────────────────────────────────────────────────────────────────────────
# Strategy 5: Proxy Metrics
# ─────────────────────────────────────────────────────────────────────────


def demo_proxy_metrics(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 5 — Score based on structural/format checks.

    No LLM judge calls needed — pure format checking.  Catches a
    surprising number of failure modes (invalid JSON, empty responses,
    markdown wrappers, missing required fields).
    """
    checks = ProxyMetricsScorer.common_checks() + [
        ProxyCheck(
            "mentions_customer",
            lambda o: any(w in o.lower() for w in ["order", "return", "refund", "ship", "account"]),
            weight=0.5,
        ),
        ProxyCheck(
            "professional_tone",
            lambda o: not any(w in o.lower() for w in ["lol", "idk", "nah", "bruh"]),
            weight=0.3,
        ),
    ]
    scorer = ProxyMetricsScorer(checks=checks)
    return run_strategy("5. Proxy Metrics", scorer, config)


# ─────────────────────────────────────────────────────────────────────────
# Strategy 6: Preference / Contrastive Scoring
# ─────────────────────────────────────────────────────────────────────────


def demo_preference(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 6 — Score by similarity to preferred output examples.

    Provide even 5-10 good/bad pairs and the scorer learns what you want.
    Evolution amplifies even this weak signal.
    """
    pairs = [
        PreferencePair(
            input_text="Where is my order #12345?",
            good_output='{"tool": "lookup_order", "parameters": {"order_id": "12345"}}',
            bad_output="Hmm, I'm not sure. Let me think about this. Your order might be somewhere. Have you checked your email?",
        ),
        PreferencePair(
            input_text="I want to return the shoes I bought last week.",
            good_output='{"tool": "process_return", "parameters": {"item": "shoes", "timeframe": "last week"}}',
            bad_output="Returns? Oh that's complicated. You should probably call our phone number.",
        ),
        PreferencePair(
            input_text="Can I get a refund for order #67890?",
            good_output='{"tool": "process_refund", "parameters": {"order_id": "67890"}}',
            bad_output="I don't know if refunds are possible. Maybe check the website?",
        ),
        PreferencePair(
            input_text="My package arrived damaged, what should I do?",
            good_output='{"tool": "report_damage", "parameters": {"issue": "damaged package"}}',
            bad_output="That sucks. Idk what to tell you lol.",
        ),
        PreferencePair(
            input_text="I forgot my password and can't log in.",
            good_output='{"tool": "reset_password", "parameters": {"issue": "forgotten password"}}',
            bad_output="Have you tried turning it off and on again?",
        ),
    ]
    scorer = PreferenceScorer(pairs)
    inputs = [p.input_text for p in pairs]
    return run_strategy("6. Preference Scoring", scorer, config, test_inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# Strategy 7: Human-in-the-Loop Tournament
# ─────────────────────────────────────────────────────────────────────────


def demo_human_tournament(config: NoEvalConfig) -> dict[str, Any]:
    """Strategy 7 — Human selects the best output per generation.

    Interactive — presents 2-3 outputs and asks the user to pick.
    Only needs ~10-15 judgments for a 3-generation run.
    """
    scorer = HumanTournament()
    # Use fewer inputs to keep the session manageable
    inputs = TEST_INPUTS[:3]
    return run_strategy("7. Human Tournament", scorer, config, test_inputs=inputs)


# ─────────────────────────────────────────────────────────────────────────
# Composite: combine multiple strategies
# ─────────────────────────────────────────────────────────────────────────


def demo_composite(config: NoEvalConfig) -> dict[str, Any]:
    """Combine strategies 1 + 5 + 4 for robust scoring.

    Weighted mix: 50% LLM judge, 30% proxy metrics, 20% consistency.
    This is the recommended approach for production use.
    """
    composite = CompositeScorer([
        (LLMJudge(rubric="Score 0-10 on helpfulness, accuracy, and tone."), 0.5),
        (ProxyMetricsScorer(ProxyMetricsScorer.common_checks()), 0.3),
        (SelfConsistencyScorer(num_samples=3), 0.2),
    ])
    return run_strategy("Composite (Judge + Proxy + Consistency)", composite, config)


# ─────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a comparison table of all strategies."""
    print("\n" + "=" * 72)
    print("  Strategy Comparison — No-Eval Prompt Evolution")
    print("=" * 72)
    print()
    print(f"  {'Strategy':<35s} {'Score':>8s} {'Time':>8s} {'Candidates':>11s}")
    print(f"  {'─' * 35} {'─' * 8} {'─' * 8} {'─' * 11}")

    for r in results:
        print(
            f"  {r['strategy']:<35s} "
            f"{r['best_score']:>7.1f}% "
            f"{r['wall_time']:>7.1f}s "
            f"{r['candidates_tried']:>10d}"
        )

    print()
    if results:
        best = max(results, key=lambda r: r["best_score"])
        print(f"  Best overall: {best['strategy']} ({best['best_score']:.1f}%)")
    print()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-Eval Prompt Evolution — 7 strategies demo"
    )
    parser.add_argument(
        "--strategy", type=int, choices=range(1, 8), default=None,
        help="Run only strategy N (1-7). Default: run all automated (1-6).",
    )
    parser.add_argument(
        "--human", action="store_true",
        help="Run ONLY the human-in-the-loop strategy (interactive).",
    )
    parser.add_argument(
        "--composite", action="store_true",
        help="Run ONLY the composite strategy demo.",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of evolutionary generations (default: 3).",
    )
    args = parser.parse_args()

    print("EvoSim Cookbook Recipe 48 — No-Eval Prompt Evolution")
    print("=" * 62)
    print(f"  Task: Customer-service agent ({len(TEST_INPUTS)} unlabelled inputs)")
    print(f"  Generations: {args.iterations}")
    print()

    config = NoEvalConfig(
        iterations=args.iterations,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.5,
        crossover_rate=0.3,
    )

    results: list[dict[str, Any]] = []

    strategy_map = {
        1: demo_llm_judge,
        2: demo_synthetic_eval,
        3: demo_tool_success,
        4: demo_self_consistency,
        5: demo_proxy_metrics,
        6: demo_preference,
        7: demo_human_tournament,
    }

    if args.human:
        results.append(demo_human_tournament(config))
    elif args.composite:
        results.append(demo_composite(config))
    elif args.strategy:
        fn = strategy_map[args.strategy]
        results.append(fn(config))
    else:
        # Run all automated strategies (1-6)
        for i in range(1, 7):
            results.append(strategy_map[i](config))

    print_summary(results)

    # Save results log
    log_path = "no_eval_strategies_log.json"
    with open(log_path, "w") as f:
        json.dump({"results": results, "task": TASK_DESCRIPTION}, f, indent=2)
    print(f"  Results saved to {log_path}")
    print("\n✓ Recipe 48 complete.")


if __name__ == "__main__":
    main()
