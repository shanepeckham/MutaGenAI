#!/usr/bin/env python3
"""
GSM8K — Evolved vs Default Prompt (GPT-4.1 via Azure AI Foundry)
=================================================================

**Benchmark**: GSM8K (Grade School Math 8K) — Cobbe et al., 2021.
8.5 k grade-school math word problems with natural-language solutions.
The standard metric is **exact-match on the final numerical answer**.

This is the gold-standard reasoning benchmark — cited in virtually
every major LLM paper (GPT-4, Llama, Gemini, Claude, etc.).  Running
MutaGenAI on it demonstrates that evolutionary prompt optimisation
improves **mathematical reasoning**, not just tool selection.

**Design**:

1. Load GSM8K ``test`` split (1,319 problems).
2. Subsample 40 problems (seed 42) for evolution.
3. Evolve prompts using NoEvalPromptEvolver with CompositeScorer.
4. Evaluate the **evolved prompt** on the full 1,319-problem test set.
5. Evaluate the **default prompt** on the same set.
6. Compare accuracy and save detailed log.

Usage::

    uv sync --extra llm
    uv run python examples/experiments/gsm8k/run1_gpt41_gsm8k.py

    # Adjust iterations:
    uv run python examples/experiments/gsm8k/run1_gpt41_gsm8k.py --iterations 6

    # Quick smoke test on 50 eval problems:
    uv run python examples/experiments/gsm8k/run1_gpt41_gsm8k.py --eval-limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root, ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptEvolverConfig,
)
from MutaGenAI.strategies import (
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyCheck,
    ProxyMetricsScorer,
)


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = textwrap.dedent("""\
    You are a math problem solver. Given a grade-school math word
    problem, solve it step by step and provide the final numerical
    answer.

    Rules:
    - Show your reasoning step by step.
    - After your reasoning, write the final answer on its own line
      in the format: #### <number>
    - The final answer must be a single number (integer or decimal).
    - Do NOT include units, dollar signs, or commas in the final
      answer.
    - Example final line: #### 42
""").strip()

# ─────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────


def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test split from HuggingFace.

    Each item has ``question`` and ``answer`` (full solution text).
    The gold numerical answer is extracted from the ``#### <number>``
    marker at the end of the answer field.
    """
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        print("ERROR: Install the 'datasets' library first:")
        print("  pip install datasets")
        sys.exit(1)

    print("  Loading GSM8K test split from HuggingFace …")
    ds = _hf_load("openai/gsm8k", "main", split="test")

    samples: list[dict] = []
    for row in ds:
        gold_answer = extract_gold_answer(row["answer"])
        if gold_answer is not None:
            samples.append({
                "question": row["question"],
                "full_solution": row["answer"],
                "gold_answer": gold_answer,
            })

    print(f"  Loaded {len(samples)} problems with valid gold answers")
    return samples


def extract_gold_answer(solution_text: str) -> str | None:
    """Extract the numerical answer after '####' in the GSM8K solution."""
    m = re.search(r"####\s*(.+)", solution_text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Remove commas from numbers like 1,234
    return raw.replace(",", "")


# ─────────────────────────────────────────────────────────────────────────
# Answer extraction and scoring
# ─────────────────────────────────────────────────────────────────────────


def extract_predicted_answer(output: str) -> str | None:
    """Extract the final numerical answer from model output.

    Looks for the ``#### <number>`` pattern first, then falls back
    to the last number on the last non-empty line.
    """
    if not output:
        return None

    # Primary: #### marker
    m = re.search(r"####\s*(.+)", output)
    if m:
        raw = m.group(1).strip().replace(",", "")
        # Extract just the number part
        num_match = re.match(r"[-+]?\d+\.?\d*", raw)
        return num_match.group(0) if num_match else raw

    # Fallback: last number in the output
    numbers = re.findall(r"[-+]?\d+\.?\d*", output)
    return numbers[-1] if numbers else None


def normalize_answer(ans: str) -> str:
    """Normalize a numerical answer for comparison."""
    if ans is None:
        return ""
    ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
    # Remove trailing .0 → integer
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


def score_gsm8k(predicted: str | None, gold: str) -> bool:
    """Return True if the predicted answer matches the gold answer."""
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(gold)


# ─────────────────────────────────────────────────────────────────────────
# Custom mutations for math reasoning
# ─────────────────────────────────────────────────────────────────────────

GSM8K_MUTATIONS: list[str] = [
    "Break the problem into smaller sub-problems before solving.",
    "Identify the key quantities and their relationships first.",
    "Write an equation or expression before computing the answer.",
    "Double-check your arithmetic at each step.",
    "Re-read the question after solving to verify you answered what was asked.",
    "Convert word descriptions to mathematical expressions explicitly.",
    "Track units throughout your calculation.",
    "If the problem involves rates, set up a clear rate × time = quantity framework.",
    "For multi-step problems, label each intermediate result clearly.",
    "Simplify the problem by considering what information is given vs what is needed.",
    "Use estimation to sanity-check your final answer.",
    "If the problem has multiple parts, solve them in logical order.",
    "Always end with: #### <number>",
    "The final answer must be a plain number with no units or symbols.",
]


# ─────────────────────────────────────────────────────────────────────────
# Proxy checks for GSM8K outputs
# ─────────────────────────────────────────────────────────────────────────

def _has_hash_answer(output: str) -> bool:
    """Check if output contains #### <number> format."""
    return bool(re.search(r"####\s*[-+]?\d+\.?\d*", output))


def _has_step_by_step(output: str) -> bool:
    """Check if output shows multi-step reasoning."""
    # Look for numbered steps, bullet points, or multi-line reasoning
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    return len(lines) >= 3


def _has_numbers(output: str) -> bool:
    """Check if output contains numerical calculations."""
    return len(re.findall(r"\d+", output)) >= 2


def _has_math_operator(output: str) -> bool:
    """Check if output contains arithmetic operators."""
    return bool(re.search(r"[\+\-\×\÷\*\/\=]", output))


def _reasonable_length(output: str) -> bool:
    """Output should be substantial but not excessive."""
    length = len(output.strip())
    return 50 < length < 3000


def _not_refusal(output: str) -> bool:
    """Detect if the model refused to answer."""
    refusals = ["i cannot", "i can't", "i'm sorry", "i don't know"]
    lower = output.lower()
    return not any(r in lower for r in refusals)


def gsm8k_proxy_checks() -> list[ProxyCheck]:
    return [
        ProxyCheck("hash_answer_format", _has_hash_answer, weight=3.0),
        ProxyCheck("step_by_step", _has_step_by_step, weight=1.5),
        ProxyCheck("has_numbers", _has_numbers, weight=1.0),
        ProxyCheck("has_math_ops", _has_math_operator, weight=0.8),
        ProxyCheck("reasonable_length", _reasonable_length, weight=0.5),
        ProxyCheck("not_refusal", _not_refusal, weight=1.0),
        ProxyCheck("not_empty", lambda o: len(o.strip()) > 0, weight=0.5),
    ]


# ─────────────────────────────────────────────────────────────────────────
# Ground-truth evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate_on_gsm8k(
    prompt: str,
    temperature: float,
    top_p: float,
    problems: list[dict],
    client: LLMClient,
    *,
    label: str = "",
) -> dict:
    """Evaluate a system prompt on GSM8K problems.

    Returns dict with accuracy, correct, total, and per-problem details.
    """
    correct = 0
    total = len(problems)
    details: list[dict] = []

    for i, prob in enumerate(problems):
        response = client.complete(
            system_prompt=prompt,
            user_message=prob["question"],
            temperature=temperature,
            top_p=top_p,
        )
        if response is None:
            details.append({
                "idx": i,
                "question": prob["question"][:80],
                "gold": prob["gold_answer"],
                "predicted": None,
                "correct": False,
                "error": "no_response",
            })
            continue

        predicted = extract_predicted_answer(response)
        is_correct = score_gsm8k(predicted, prob["gold_answer"])
        if is_correct:
            correct += 1

        details.append({
            "idx": i,
            "question": prob["question"][:80],
            "gold": prob["gold_answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

        # Progress indicator every 50 problems
        if (i + 1) % 50 == 0:
            running_acc = correct / (i + 1) * 100
            print(f"    [{label}] {i+1}/{total} — running accuracy: "
                  f"{running_acc:.1f}%")

    accuracy = correct / total * 100.0 if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSM8K prompt evolution — GPT-4.1 via Azure AI Foundry"
    )
    parser.add_argument(
        "--iterations", type=int, default=6,
        help="Number of evolutionary generations (default: 6)",
    )
    parser.add_argument(
        "--eval-limit", type=int, default=0,
        help="Limit eval to N problems (0 = full test set, default: 0)",
    )
    args = parser.parse_args()

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  MutaGenAI × GSM8K — Run 1: GPT-4.1 via Azure AI Foundry       ║
    ║  Grade School Math · Evolved vs Default Prompt                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    print(f"  Backend:    Azure AI Foundry (RBAC / Managed Identity)")
    print(f"  Model:      {azure_deployment}")
    print(f"  Endpoint:   {azure_endpoint[:60]}...")
    print(f"  Benchmark:  GSM8K (openai/gsm8k)")
    print()

    # ── Backend setup ──────────────────────────────────────────────────
    azure_cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_use_rbac=True,
        timeout=90.0,
    )
    client = LLMClient(azure_cfg)

    if not client.is_available():
        print("  ✗ Azure AI Foundry not available.")
        print("    Check AZURE_OPENAI_ENDPOINT env var and managed identity.")
        return
    print(f"  ✓ Azure AI Foundry available — deployment: {azure_deployment}")

    # ── Load GSM8K ─────────────────────────────────────────────────────
    all_problems = load_gsm8k_test()
    if not all_problems:
        print("  ✗ No problems loaded.")
        return

    # ── Subsample for evolution ────────────────────────────────────────
    rng = np.random.default_rng(42)
    evo_size = min(40, len(all_problems))
    evo_indices = set(rng.choice(len(all_problems), size=evo_size, replace=False))
    evo_problems = [all_problems[i] for i in sorted(evo_indices)]

    # Holdout = everything not in evolution set
    holdout_problems = [
        all_problems[i] for i in range(len(all_problems))
        if i not in evo_indices
    ]

    # Eval set: holdout (or limited subset for quick testing)
    if args.eval_limit > 0:
        eval_problems = holdout_problems[:args.eval_limit]
    else:
        eval_problems = holdout_problems

    print(f"\n  Evolution subset:   {len(evo_problems)} problems")
    print(f"  Holdout/eval set:   {len(eval_problems)} problems")

    # ── Build test inputs for evolution ────────────────────────────────
    test_inputs = [p["question"] for p in evo_problems]

    # ── Seed templates ─────────────────────────────────────────────────
    seed_templates = [
        TASK_DESCRIPTION,
        (
            "Solve the following math problem step by step.\n"
            "Show all your work.\n"
            "End with: #### <answer>"
        ),
        (
            "You are a careful math tutor. Read the problem slowly, "
            "identify what is being asked, solve step by step, and "
            "give the final answer as:\n#### <number>"
        ),
        (
            "# Math Problem Solver\n\n"
            "## Approach\n"
            "1. Read the problem carefully\n"
            "2. Identify given quantities\n"
            "3. Set up equations\n"
            "4. Solve step by step\n"
            "5. State the answer\n\n"
            "## Output Format\n"
            "End your solution with: #### <number>\n"
            "The number must be plain (no $, no commas, no units)."
        ),
    ]

    # ── NoEvalConfig ───────────────────────────────────────────────────
    noeval_cfg = NoEvalConfig(
        iterations=args.iterations,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.5,
        crossover_rate=0.3,
        migration_interval=2,
        backend=LLMBackend.AZURE_OPENAI,
        problem_type=ProblemType.CLASSIFICATION,
        adaptive_mutations=True,
        llm_mutation_rate=0.3,
        warmup_adaptive=True,
        error_decay=0.5,
        max_tokens=1024,
    )

    # ── Scorer ─────────────────────────────────────────────────────────
    judge_rubric = textwrap.dedent("""\
        Score the math solution 0-10 on these criteria:
        - Correct reasoning: Are the steps logically sound? (0-4)
        - Arithmetic accuracy: Are calculations correct? (0-3)
        - Answer format: Does it end with #### <number>? (0-2)
        - Clarity: Is the solution clearly written? (0-1)
    """)

    composite = CompositeScorer([
        (LLMJudge(rubric=judge_rubric, max_score=10.0), 0.55),
        (ProxyMetricsScorer(gsm8k_proxy_checks()), 0.45),
    ])

    # ── Phase 1: Evolution ─────────────────────────────────────────────
    print(f"\n  Phase 1: Prompt Evolution ({args.iterations} generations)")
    print("  " + "=" * 55)
    print(f"  Evolution set: {len(evo_problems)} problems")
    print(f"  Config: pop=4, islands=2, adaptive=True, warmup=True")
    print(f"  Scorer: LLMJudge(0.55) + ProxyMetrics(0.45)")
    print(f"  Mutations: {len(GSM8K_MUTATIONS)} custom math mutations")

    t0 = time.perf_counter()

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=composite,
        config=noeval_cfg,
        seed_templates=seed_templates,
        verbose=True,
        custom_mutations=GSM8K_MUTATIONS,
    )

    result = evolver.run()
    evo_wall_time = time.perf_counter() - t0

    print(f"\n  No-eval fitness:  {result.best_score:.1f}%")
    print(f"  Temperature:      {result.best_temperature:.4f}")
    print(f"  Top-p:            {result.best_top_p:.4f}")
    print(f"  Evolution time:   {evo_wall_time:.1f}s")

    evolved_prompt = result.best_prompt
    evolved_temp = result.best_temperature
    evolved_top_p = result.best_top_p

    # ── Phase 2: Ground-truth — Evolved Prompt ─────────────────────────
    print(f"\n  Phase 2: Ground-Truth Evaluation — Evolved Prompt")
    print("  " + "=" * 55)
    print(f"  Evaluating on {len(eval_problems)} holdout problems…")

    t1 = time.perf_counter()
    evolved_results = evaluate_on_gsm8k(
        evolved_prompt,
        evolved_temp,
        evolved_top_p,
        eval_problems,
        client,
        label="Evolved",
    )
    eval_evolved_time = time.perf_counter() - t1

    print(f"\n  Evolved Prompt Accuracy: {evolved_results['accuracy']:.2f}%")
    print(f"    ({evolved_results['correct']}/{evolved_results['total']} correct)")
    print(f"    Eval time: {eval_evolved_time:.1f}s")

    # ── Phase 3: Ground-truth — Default Prompt ─────────────────────────
    print(f"\n  Phase 3: Ground-Truth Evaluation — Default Prompt")
    print("  " + "=" * 55)
    print(f"  Evaluating default prompt on {len(eval_problems)} holdout problems…")

    t2 = time.perf_counter()
    default_results = evaluate_on_gsm8k(
        TASK_DESCRIPTION,
        0.1,     # conservative default temperature
        0.95,    # default top_p
        eval_problems,
        client,
        label="Default",
    )
    eval_default_time = time.perf_counter() - t2

    print(f"\n  Default Prompt Accuracy: {default_results['accuracy']:.2f}%")
    print(f"    ({default_results['correct']}/{default_results['total']} correct)")
    print(f"    Eval time: {eval_default_time:.1f}s")

    # ── Phase 4: Comparison ────────────────────────────────────────────
    delta = evolved_results["accuracy"] - default_results["accuracy"]
    print(f"\n{'=' * 70}")
    print("  GSM8K Results Comparison")
    print(f"{'=' * 70}")
    print(f"\n  {'Prompt':<30} {'Accuracy':>10} {'Correct':>10} "
          f"{'Total':>8} {'Time':>10}")
    print(f"  {'─' * 70}")
    print(f"  {'Evolved (MutaGenAI)':<30} "
          f"{evolved_results['accuracy']:9.2f}% "
          f"{evolved_results['correct']:>10} "
          f"{evolved_results['total']:>8} "
          f"{eval_evolved_time:9.1f}s")
    print(f"  {'Default (hand-written)':<30} "
          f"{default_results['accuracy']:9.2f}% "
          f"{default_results['correct']:>10} "
          f"{default_results['total']:>8} "
          f"{eval_default_time:9.1f}s")
    print(f"\n  Delta: {'+' if delta >= 0 else ''}{delta:.2f}%")

    if delta > 0:
        print(f"  ✓ Evolved prompt outperforms default by {delta:.2f}%")
    elif delta < 0:
        print(f"  ✗ Default prompt leads by {abs(delta):.2f}%")
    else:
        print(f"  = Tie")

    # ── Save log ───────────────────────────────────────────────────────
    log_path = Path(_root) / "logs" / "gsm8k_run1_experiment_log.json"
    log = {
        "benchmark": "GSM8K",
        "model": azure_deployment,
        "backend": "Azure AI Foundry (RBAC)",
        "evolution": {
            "iterations": args.iterations,
            "population_size": 4,
            "num_islands": 2,
            "evo_sample_size": len(evo_problems),
            "noeval_fitness": result.best_score,
            "wall_time_s": evo_wall_time,
        },
        "evolved_prompt": evolved_prompt,
        "evolved_temperature": evolved_temp,
        "evolved_top_p": evolved_top_p,
        "evolved_accuracy": evolved_results["accuracy"],
        "evolved_correct": evolved_results["correct"],
        "evolved_total": evolved_results["total"],
        "default_prompt": TASK_DESCRIPTION,
        "default_temperature": 0.1,
        "default_top_p": 0.95,
        "default_accuracy": default_results["accuracy"],
        "default_correct": default_results["correct"],
        "default_total": default_results["total"],
        "delta": delta,
        "eval_problems_count": len(eval_problems),
        "holdout_size": len(holdout_problems),
        "eval_evolved_details": evolved_results["details"],
        "eval_default_details": default_results["details"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Log saved to {log_path}")


if __name__ == "__main__":
    main()
