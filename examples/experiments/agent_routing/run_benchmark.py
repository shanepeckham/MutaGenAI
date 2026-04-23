#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Benchmark: evolved prompt vs static prompt on multi-step agent routing.

Uses the V1rtucious/multi-step-agent-routing HuggingFace dataset.
Samples 100 rows from train + 100 from test, sends each user message
through both prompts via Azure OpenAI gpt-4.1, and compares the
predicted routing against ground truth.

Run:  uv run python examples/experiments/agent_routing/run_benchmark.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

os.environ.setdefault("AZURE_OPENAI_USE_RBAC", "true")

from prompture.prompt_evolver import LLMBackend, LLMClient, PromptEvolverConfig

# ── Configuration ─────────────────────────────────────────
BACKEND = LLMBackend.AZURE_OPENAI
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
HF_DATASET = "V1rtucious/multi-step-agent-routing"
TRAIN_SAMPLE_SIZE = 100
TEST_SAMPLE_SIZE = 100
TEMPERATURE = 0.1
TOP_P = 0.95

STATIC_PROMPT = (
    "You are an intelligent orchestrator. Route the user request to the "
    "correct specialist agents in the appropriate sequence. Available Agents: "
    "authentication_agent, request_validation_agent, authorization_agent, "
    "user_information_retriever_agent, transaction_history_agent, "
    "balance_checking_agent, data_analysis_agent, risk_assessment_agent, "
    "duplicate_detection_agent, fraud_detection_agent, "
    "document_verification_agent, compliance_check_agent, "
    "cross_reference_agent, approval_workflow_agent, leave_approval_agent, "
    "refund_processing_agent, pricing_calculation_agent, "
    "policy_evaluation_agent, notification_agent, email_agent, "
    "report_generation_agent, audit_logging_agent, case_creation_agent, "
    "informational_queries_agent, troubleshooting_agent, "
    "recommendation_agent, intent_and_sentiment_extraction_agent"
)

EVOLVED_PROMPT = (
    "Respond with JSON only. No explanation.\n"
    "You are an intelligent orchestrator. Route the user request to the "
    "correct specialist agents in the appropriate sequence.\n"
    "\n"
    "Available Agents: authentication_agent \u2014 Verifies user identity and "
    "credentials before granting access., request_validation_agent \u2014 Checks "
    "incoming requests for completeness and correctness., authorization_agent "
    "\u2014 Determines if a user has permission to perform a specific action., "
    "user_information_retriever_agent \u2014 Fetches detailed information about a "
    "user., transaction_history_agent \u2014 Retrieves past transaction records "
    "for a user or account., balance_checking_agent \u2014 Provides the current "
    "balance of a user\u2019s account., data_analysis_agent \u2014 Analyzes data to "
    "extract insights or patterns., risk_assessment_agent \u2014 Evaluates the "
    "risk associated with a user or transaction., duplicate_detection_agent "
    "\u2014 Identifies and flags duplicate entries or actions., "
    "fraud_detection_agent \u2014 Detects and flags potentially fraudulent "
    "transactions or account activity., document_verification_agent \u2014 "
    "Confirms the authenticity and validity of submitted documents., "
    "compliance_check_agent \u2014 Ensures actions comply with relevant laws and "
    "regulations., cross_reference_agent \u2014 Matches and verifies data across "
    "multiple sources., approval_workflow_agent \u2014 Manages and tracks the "
    "approval process for requests., leave_approval_agent \u2014 Handles "
    "requests and approvals for employee leave., refund_processing_agent "
    "\u2014 Processes and tracks user refund requests., "
    "pricing_calculation_agent \u2014 Computes pricing based on rules and "
    "conditions., policy_evaluation_agent \u2014 Assesses policies applicable "
    "to a request., notification_agent \u2014 Sends notifications to users., "
    "email_agent \u2014 Sends email communications., report_generation_agent "
    "\u2014 Generates reports from data., audit_logging_agent \u2014 Logs actions "
    "for audit trails., case_creation_agent \u2014 Creates support or "
    "investigation cases., informational_queries_agent \u2014 Answers general "
    "information questions., troubleshooting_agent \u2014 Diagnoses and resolves "
    "issues., recommendation_agent \u2014 Provides recommendations based on "
    "analysis., intent_and_sentiment_extraction_agent \u2014 Extracts user "
    "intent and sentiment from input."
)

STEP_KEYS = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]

# All valid agent names in the system
VALID_AGENTS = {
    "authentication_agent", "request_validation_agent", "authorization_agent",
    "user_information_retriever_agent", "transaction_history_agent",
    "balance_checking_agent", "data_analysis_agent", "risk_assessment_agent",
    "duplicate_detection_agent", "fraud_detection_agent",
    "document_verification_agent", "compliance_check_agent",
    "cross_reference_agent", "approval_workflow_agent", "leave_approval_agent",
    "refund_processing_agent", "pricing_calculation_agent",
    "policy_evaluation_agent", "notification_agent", "email_agent",
    "report_generation_agent", "audit_logging_agent", "case_creation_agent",
    "informational_queries_agent", "troubleshooting_agent",
    "recommendation_agent", "intent_and_sentiment_extraction_agent",
}


# ── Parsing ───────────────────────────────────────────────
def _extract_json(text: str) -> dict | list | None:
    """Try to extract JSON from text, handling markdown fences."""
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Find outermost braces or brackets
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass

    return None


def extract_agent_sequence(text: str) -> list[str]:
    """Extract an ordered list of agent names from LLM output.

    Handles formats:
      - {"sequence": ["agent1", "agent2", ...]}
      - {"routing": ["agent1", ...]}
      - {"step_1": [...], "step_2": [...], ...}
      - {"agents": ["agent1", ...]}
      - ["agent1", "agent2", ...]
      - numbered lists: "1. agent1\\n2. agent2"
      - comma-separated: "agent1, agent2, agent3"
    """
    if not text:
        return []

    data = _extract_json(text)

    if isinstance(data, list):
        # Direct list of agent names
        return [a.strip() for a in data if isinstance(a, str) and a.strip()]

    if isinstance(data, dict):
        # step_N format
        has_steps = any(k in data for k in STEP_KEYS)
        if has_steps:
            agents = []
            for key in STEP_KEYS:
                val = data.get(key)
                if isinstance(val, list):
                    agents.extend([a.strip() for a in val if isinstance(a, str) and a.strip()])
                elif isinstance(val, str) and val.strip():
                    agents.append(val.strip())
            return agents

        # Common wrapper keys: sequence, routing, agents, steps, pipeline, etc.
        for wrapper_key in ["sequence", "routing", "agents", "steps", "pipeline",
                           "agent_sequence", "route", "routing_sequence"]:
            val = data.get(wrapper_key)
            if isinstance(val, list):
                result = []
                for item in val:
                    if isinstance(item, str):
                        result.append(item.strip())
                    elif isinstance(item, dict):
                        # {"step": 1, "agent": "name"} style
                        for k in ["agent", "agent_name", "name"]:
                            if k in item and isinstance(item[k], str):
                                result.append(item[k].strip())
                                break
                return result

        # Try all list values in the dict
        for val in data.values():
            if isinstance(val, list) and val and isinstance(val[0], str):
                return [a.strip() for a in val if isinstance(a, str) and a.strip()]

    # Fallback: find agent names in plain text
    found = []
    for agent in VALID_AGENTS:
        if agent in text:
            found.append(agent)

    if found:
        # Try to preserve ordering from the text
        positions = [(text.index(a), a) for a in found]
        positions.sort()
        return [a for _, a in positions]

    return []


def flatten_ground_truth(gt: dict) -> list[str]:
    """Flatten step_1..step_6 ground truth to an ordered agent list."""
    agents = []
    for key in STEP_KEYS:
        val = gt.get(key)
        if isinstance(val, list):
            # Sort within-step agents for deterministic comparison
            step_agents = sorted([a for a in val if a])
            agents.extend(step_agents)
    return agents


def ground_truth_agent_set(gt: dict) -> set[str]:
    """Get the set of all agents in ground truth (ignoring order)."""
    agents = set()
    for key in STEP_KEYS:
        val = gt.get(key)
        if isinstance(val, list):
            agents.update(a for a in val if a)
    return agents


# ── Scoring ───────────────────────────────────────────────
def score_sample(
    pred_agents: list[str],
    gt: dict,
) -> dict:
    """Score a prediction against ground truth using multiple metrics.

    Returns dict with:
      - agent_set_precision / recall / f1  (set-level, ignoring order)
      - sequence_exact_match  (ordered list matches exactly)
      - sequence_prefix_match (longest common prefix ratio)
    """
    gt_set = ground_truth_agent_set(gt)
    pred_set = set(pred_agents)
    gt_sequence = flatten_ground_truth(gt)

    # Set-level metrics
    if not gt_set and not pred_set:
        precision = recall = f1 = 1.0
    elif not pred_set:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    elif not gt_set:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set)
        recall = tp / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Sequence-level metrics
    sequence_exact = pred_agents == gt_sequence

    # Longest common prefix
    prefix_len = 0
    for a, b in zip(pred_agents, gt_sequence):
        if a == b:
            prefix_len += 1
        else:
            break
    prefix_ratio = prefix_len / len(gt_sequence) if gt_sequence else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sequence_exact_match": sequence_exact,
        "prefix_ratio": prefix_ratio,
        "gt_agents": sorted(gt_set),
        "pred_agents": pred_agents,
        "gt_count": len(gt_set),
        "pred_count": len(pred_set),
    }


# ── Evaluation ────────────────────────────────────────────
def evaluate_split(
    client: LLMClient,
    samples: list[dict],
    system_prompt: str,
    label: str,
) -> list[dict]:
    """Run all samples through the LLM and score against ground truth."""
    results = []
    for i, sample in enumerate(samples):
        # Extract user message from messages list
        user_msg = ""
        for msg in sample["messages"]:
            if msg.get("role") == "user":
                user_msg = msg["content"]
                break

        if not user_msg:
            continue

        gt = sample["ground_truth_routing"]

        # Call LLM
        response = client.complete(
            system_prompt=system_prompt,
            user_message=user_msg,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        pred_agents = extract_agent_sequence(response or "")
        scores = score_sample(pred_agents, gt)

        results.append({
            "index": i,
            "user_message": user_msg,
            "ground_truth_agents": scores["gt_agents"],
            "predicted_agents": pred_agents,
            "raw_response": response,
            "scores": scores,
            "complexity_level": sample.get("complexity_level", ""),
            "routing_pattern": sample.get("routing_pattern", ""),
        })

        status = "✓" if scores["f1"] == 1.0 else ("~" if scores["f1"] > 0.5 else "✗")
        print(
            f"  [{label}] {i + 1:3d}/{len(samples)} {status}  "
            f"F1={scores['f1']:.0%} P={scores['precision']:.0%} R={scores['recall']:.0%}  "
            f"{user_msg[:50]}..."
        )

    return results


def aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-sample results."""
    if not results:
        return {"mean_f1": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "n": 0}

    mean_f1 = sum(r["scores"]["f1"] for r in results) / len(results)
    mean_prec = sum(r["scores"]["precision"] for r in results) / len(results)
    mean_rec = sum(r["scores"]["recall"] for r in results) / len(results)
    seq_exact = sum(1 for r in results if r["scores"]["sequence_exact_match"]) / len(results)
    mean_prefix = sum(r["scores"]["prefix_ratio"] for r in results) / len(results)
    perfect_f1 = sum(1 for r in results if r["scores"]["f1"] == 1.0) / len(results)

    # Breakdown by complexity
    by_complexity: dict[str, list[float]] = {}
    for r in results:
        level = r.get("complexity_level", "unknown")
        by_complexity.setdefault(level, []).append(r["scores"]["f1"])

    complexity_f1 = {
        k: sum(v) / len(v) for k, v in sorted(by_complexity.items())
    }

    # Breakdown by routing pattern
    by_pattern: dict[str, list[float]] = {}
    for r in results:
        pattern = r.get("routing_pattern", "unknown")
        by_pattern.setdefault(pattern, []).append(r["scores"]["f1"])

    pattern_f1 = {
        k: sum(v) / len(v) for k, v in sorted(by_pattern.items())
    }

    return {
        "n": len(results),
        "mean_f1": mean_f1,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "perfect_f1_rate": perfect_f1,
        "sequence_exact_match_rate": seq_exact,
        "mean_prefix_ratio": mean_prefix,
        "by_complexity": complexity_f1,
        "by_routing_pattern": pattern_f1,
    }


# ── Main ──────────────────────────────────────────────────
def main() -> int:
    """Run the benchmark comparison."""
    print()
    print("=" * 70)
    print("  Agent Routing Benchmark: Evolved vs Static Prompt")
    print("=" * 70)
    print()

    # Load dataset
    print("  Loading dataset from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: pip install datasets", file=sys.stderr)
        return 1

    hf_token = os.getenv("HF_TOKEN")
    ds = load_dataset(HF_DATASET, token=hf_token)
    train_data = list(ds["train"])
    test_data = list(ds["test"])
    print(f"  Train: {len(train_data)} rows, Test: {len(test_data)} rows")

    # Sample
    import random
    random.seed(42)
    train_sample = random.sample(train_data, min(TRAIN_SAMPLE_SIZE, len(train_data)))
    test_sample = random.sample(test_data, min(TEST_SAMPLE_SIZE, len(test_data)))
    print(
        f"  Sampled: {len(train_sample)} train, {len(test_sample)} test"
    )

    # Build LLM client
    config = PromptEvolverConfig(backend=BACKEND, azure_deployment=MODEL)
    client = LLMClient(config)

    print(f"  Backend: {BACKEND.value}  Model: {MODEL}")
    print("  Checking LLM backend...", end=" ")
    if client.is_available():
        print("OK")
    else:
        print("UNAVAILABLE — cannot run benchmark")
        return 1

    print()

    # ── Run evaluations ──────────────────────────────────
    all_results: dict[str, dict] = {}

    for prompt_label, system_prompt in [
        ("static", STATIC_PROMPT),
        ("evolved", EVOLVED_PROMPT),
    ]:
        print(f"  {'─' * 68}")
        print(f"  Evaluating: {prompt_label.upper()} prompt")
        print(f"  {'─' * 68}")

        # Train split
        print(f"\n  ── Train split ({len(train_sample)} samples) ──")
        start = time.time()
        train_results = evaluate_split(
            client, train_sample, system_prompt, f"{prompt_label}/train"
        )
        train_elapsed = time.time() - start
        train_metrics = aggregate_metrics(train_results)

        # Test split
        print(f"\n  ── Test split ({len(test_sample)} samples) ──")
        start = time.time()
        test_results = evaluate_split(
            client, test_sample, system_prompt, f"{prompt_label}/test"
        )
        test_elapsed = time.time() - start
        test_metrics = aggregate_metrics(test_results)

        all_results[prompt_label] = {
            "train": {
                "metrics": train_metrics,
                "elapsed": train_elapsed,
                "details": train_results,
            },
            "test": {
                "metrics": test_metrics,
                "elapsed": test_elapsed,
                "details": test_results,
            },
        }

        print(f"\n  {prompt_label.upper()} results:")
        print(
            f"    Train: F1={train_metrics['mean_f1']:.1%}  "
            f"P={train_metrics['mean_precision']:.1%}  "
            f"R={train_metrics['mean_recall']:.1%}  "
            f"SeqEM={train_metrics['sequence_exact_match_rate']:.1%}  "
            f"({train_elapsed:.1f}s)"
        )
        print(
            f"    Test:  F1={test_metrics['mean_f1']:.1%}  "
            f"P={test_metrics['mean_precision']:.1%}  "
            f"R={test_metrics['mean_recall']:.1%}  "
            f"SeqEM={test_metrics['sequence_exact_match_rate']:.1%}  "
            f"({test_elapsed:.1f}s)"
        )
        print()

    # ── Summary comparison ────────────────────────────────
    print("=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"  {'':18s} {'F1':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'SeqEM':>8s}  {'PerfF1':>8s}")
    print(f"  {'':18s} {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for prompt_label in ["static", "evolved"]:
        for split in ["train", "test"]:
            m = all_results[prompt_label][split]["metrics"]
            tag = f"{prompt_label}/{split}"
            print(
                f"  {tag:18s} {m['mean_f1']:>7.1%}  "
                f"{m['mean_precision']:>7.1%}  "
                f"{m['mean_recall']:>7.1%}  "
                f"{m['sequence_exact_match_rate']:>7.1%}  "
                f"{m['perfect_f1_rate']:>7.1%}"
            )
    print()

    # Gain
    for split in ["train", "test"]:
        s_f1 = all_results["static"][split]["metrics"]["mean_f1"]
        e_f1 = all_results["evolved"][split]["metrics"]["mean_f1"]
        s_p = all_results["static"][split]["metrics"]["mean_precision"]
        e_p = all_results["evolved"][split]["metrics"]["mean_precision"]
        s_r = all_results["static"][split]["metrics"]["mean_recall"]
        e_r = all_results["evolved"][split]["metrics"]["mean_recall"]
        print(
            f"  Δ {split:5s}:  F1 {e_f1 - s_f1:+.1%}  "
            f"P {e_p - s_p:+.1%}  R {e_r - s_r:+.1%}"
        )
    print()

    # Breakdown by complexity for evolved/test
    print("  Evolved prompt — Test by complexity (mean F1):")
    for level, rate in all_results["evolved"]["test"]["metrics"].get(
        "by_complexity", {}
    ).items():
        print(f"    {level:12s}  {rate:.1%}")

    print()
    print("  Evolved prompt — Test by routing pattern (mean F1):")
    for pattern, rate in all_results["evolved"]["test"]["metrics"].get(
        "by_routing_pattern", {}
    ).items():
        print(f"    {pattern:25s}  {rate:.1%}")

    # ── Save results ──────────────────────────────────────
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "benchmark_results.json"

    # Strip raw details for the summary file (keep metrics only)
    summary = {}
    for prompt_label in ["static", "evolved"]:
        summary[prompt_label] = {}
        for split in ["train", "test"]:
            entry = all_results[prompt_label][split]
            summary[prompt_label][split] = {
                "metrics": entry["metrics"],
                "elapsed": entry["elapsed"],
            }

    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Summary saved to {out_path}")

    # Full detail log
    detail_path = out_dir / "benchmark_detail.json"
    detail_log = {}
    for prompt_label in ["static", "evolved"]:
        detail_log[prompt_label] = {}
        for split in ["train", "test"]:
            entry = all_results[prompt_label][split]
            detail_log[prompt_label][split] = {
                "metrics": entry["metrics"],
                "elapsed": entry["elapsed"],
                "samples": [
                    {
                        "index": r["index"],
                        "user_message": r["user_message"],
                        "ground_truth_agents": r["ground_truth_agents"],
                        "predicted_agents": r["predicted_agents"],
                        "raw_response": r["raw_response"],
                        "f1": r["scores"]["f1"],
                        "precision": r["scores"]["precision"],
                        "recall": r["scores"]["recall"],
                        "sequence_exact_match": r["scores"]["sequence_exact_match"],
                        "complexity_level": r["complexity_level"],
                        "routing_pattern": r["routing_pattern"],
                    }
                    for r in entry["details"]
                ],
            }

    detail_path.write_text(json.dumps(detail_log, indent=2, default=str))
    print(f"  Details saved to {detail_path}")
    print("  Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
