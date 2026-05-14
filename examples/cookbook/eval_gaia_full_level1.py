#!/usr/bin/env python3
"""Evaluate the evolved GAIA prompt on ALL 146 Level 1 samples.

Loads the best prompt from logs/gaia_evolution_log.json and runs it
against both validation (53) and test (93) splits.

Usage:
    cd /path/to/Prompture
    MUTAGENAI_BACKEND=azure_openai uv run python examples/cookbook/eval_gaia_full_level1.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from MutaGenAI.prompt_evolver import LLMBackend, LLMClient, PromptEvolverConfig  # noqa: E402


# ── Scoring helpers (mirrors cookbook) ─────────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    for p in [".", ",", "!", "?", ";", ":"]:
        s = s.rstrip(p)
    for art in ["the ", "a ", "an "]:
        if s.startswith(art):
            s = s[len(art):]
    return " ".join(s.split())


def score_exact_match(prediction: str, ground_truth: str) -> float:
    norm_pred = _normalize_answer(prediction)
    norm_gold = _normalize_answer(ground_truth)
    if norm_pred == norm_gold:
        return 1.0
    try:
        if abs(float(norm_pred) - float(norm_gold)) < 1e-6:
            return 1.0
    except ValueError:
        pass
    if norm_gold in norm_pred and len(norm_gold) > 2:
        return 0.5
    return 0.0


def extract_final_answer(response: str) -> str:
    for pattern in [
        r"Final\s*Answer:\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
        r"ANSWER:\s*(.+?)(?:\n|$)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs", "gaia_evolution_log.json",
    )
    log = json.load(open(log_path))
    evolved_prompt = log["best_prompt"]
    temp = log["results"]["best_temperature"]
    top_p = log["results"]["best_top_p"]

    print(f"Evolved prompt ({len(evolved_prompt)} chars):")
    print(evolved_prompt[:200])
    print("---")
    print(f"Temperature: {temp}, Top-p: {top_p}\n")

    # ── Load ALL Level 1 from both splits ──
    from datasets import load_dataset

    examples: list[dict] = []
    for split in ["validation", "test"]:
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)
        for row in ds:
            level = str(row.get("Level", ""))
            if level != "1":
                continue
            examples.append({
                "task_id": row.get("task_id", ""),
                "question": row.get("Question", ""),
                "final_answer": row.get("Final answer", ""),
                "split": split,
            })

    n_val = sum(1 for e in examples if e["split"] == "validation")
    n_test = sum(1 for e in examples if e["split"] == "test")
    print(f"Total Level 1 samples: {len(examples)}")
    print(f"  Validation: {n_val}")
    print(f"  Test:       {n_test}")

    # ── LLM client ──
    cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        azure_deployment="gpt-4.1",
        azure_use_rbac=True,
        timeout=120.0,
    )
    client = LLMClient(cfg)

    # ── Evaluate ──
    print(f"\nEvaluating on {len(examples)} samples...\n")
    results: list[dict] = []
    correct = 0.0
    start = time.time()

    for i, ex in enumerate(examples):
        try:
            response = client.complete(
                system_prompt=evolved_prompt,
                user_message=ex["question"],
                temperature=temp,
                top_p=top_p,
            )
            if response is None:
                predicted = ""
                score = 0.0
            else:
                predicted = extract_final_answer(response)
                score = score_exact_match(predicted, ex["final_answer"])
        except Exception as e:
            predicted = ""
            score = 0.0
            response = f"ERROR: {e}"

        correct += score
        status = "\u2713" if score == 1.0 else ("\u00bd" if score == 0.5 else "\u2717")
        gold_trunc = ex["final_answer"][:30]
        pred_trunc = predicted[:30]
        print(
            f"  [{i + 1:3d}/{len(examples)}] {status}  "
            f"gold={gold_trunc:30s}  pred={pred_trunc:30s}  "
            f"({ex['split']})",
        )

        results.append({
            "task_id": ex["task_id"],
            "split": ex["split"],
            "question": ex["question"][:100],
            "gold": ex["final_answer"],
            "predicted": predicted,
            "score": score,
        })

    elapsed = time.time() - start
    total = len(examples)
    accuracy = (correct / total * 100) if total else 0

    print(f"\n{'=' * 60}")
    print(f"RESULTS: All Level 1 ({total} samples)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:       {accuracy:.2f}%  ({correct:.1f}/{total})")
    print(f"  Wall time:      {elapsed:.1f}s")

    for split in ["validation", "test"]:
        s_results = [r for r in results if r["split"] == split]
        s_correct = sum(r["score"] for r in s_results)
        s_total = len(s_results)
        s_acc = (s_correct / s_total * 100) if s_total else 0
        print(f"  {split:12s}:  {s_acc:.2f}%  ({s_correct:.1f}/{s_total})")

    print(f"{'=' * 60}")

    # ── Save ──
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs", "gaia_full_level1_eval.json",
    )
    out = {
        "experiment": "gaia_full_level1_eval",
        "prompt": evolved_prompt,
        "temperature": temp,
        "top_p": top_p,
        "total_samples": total,
        "accuracy_pct": round(accuracy, 2),
        "correct": correct,
        "wall_time_s": round(elapsed, 1),
        "per_split": {},
        "details": results,
    }
    for split in ["validation", "test"]:
        s_results = [r for r in results if r["split"] == split]
        s_correct = sum(r["score"] for r in s_results)
        out["per_split"][split] = {
            "total": len(s_results),
            "correct": s_correct,
            "accuracy_pct": round(s_correct / len(s_results) * 100, 2) if s_results else 0,
        }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
