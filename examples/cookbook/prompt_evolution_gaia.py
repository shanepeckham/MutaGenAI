#!/usr/bin/env python3
"""
Cookbook Recipe — GAIA Agentic Benchmark Prompt Evolution
=========================================================

Evolves system prompts on the **GAIA** (General AI Assistants) benchmark
— the premier multi-step agentic reasoning benchmark with a public
HuggingFace leaderboard.

GAIA (Mialon et al., 2023) tests an AI agent's ability to perform
multi-step reasoning tasks that may require:
- Web browsing and information retrieval
- File reading and comprehension
- Mathematical calculation
- Code execution
- Multi-hop reasoning across sources

The benchmark scores agents on **exact-match accuracy** across 3 levels:
- **Level 1** — 1–5 steps, straightforward reasoning
- **Level 2** — 5–10 steps, complex tool use
- **Level 3** — 10+ steps, expert-level multi-source reasoning

Top systems (GPT-4 + plugins) score only ~15 % on Level 3, while
humans achieve ~92 %, making GAIA one of the hardest agentic
benchmarks.  Even on Level 1, most systems score below 50 %.

Why GAIA for token optimization
-------------------------------
Agent system prompts are typically very verbose (800–1500 tokens),
instructing the model on multi-step planning, tool usage patterns,
and answer formatting.  Token optimization tests whether shorter
prompts maintain agentic reasoning quality.

Algorithm experiments
---------------------
* **Standard** — 3 iterations, pop 4, 2 islands  (balanced)
* **Deep**     — 5 iterations, pop 6, 3 islands  (thorough)

Usage::

    uv sync --extra llm
    uv pip install datasets  # for HuggingFace loading
    uv run python examples/cookbook/prompt_evolution_gaia.py

Environment variables::

    MUTAGENAI_BACKEND         — "ollama" | "azure" | "openai"  (default: ollama)
    MUTAGENAI_MINIMIZE_TOKENS — "1" to enable token optimization
    MUTAGENAI_TOKEN_WEIGHT    — blend weight for efficiency (default: 0.10)
    MUTAGENAI_EFFICIENCY_CAP  — max efficiency ratio (default: 2.0)
    MUTAGENAI_ACCURACY_BAND   — tiebreaker band width (default: 2.0)
    GAIA_EXPERIMENT           — "standard" | "deep"  (default: standard)
    GAIA_LEVEL                — "1" | "2" | "1,2"  (default: 1)
    GAIA_EVAL_SIZE            — samples per evaluation (default: 30)

The script saves an experiment log to ``logs/gaia_evolution_log.json``
for dashboard consumption.

References
----------
* Leaderboard : https://huggingface.co/spaces/gaia-benchmark/leaderboard
* Dataset     : https://huggingface.co/datasets/gaia-benchmark/GAIA
* Paper       : https://arxiv.org/abs/2311.12983
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
    count_prompt_tokens,
)

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

BACKEND = LLMBackend(os.getenv("MUTAGENAI_BACKEND", "ollama"))
MODEL = os.getenv(
    "OLLAMA_MODEL" if BACKEND == LLMBackend.OLLAMA else "AZURE_OPENAI_DEPLOYMENT",
    "qwen3:8b" if BACKEND == LLMBackend.OLLAMA else "gpt-4.1",
)

EXPERIMENT = os.getenv("GAIA_EXPERIMENT", "standard")
GAIA_LEVELS = [int(x) for x in os.getenv("GAIA_LEVEL", "1").split(",")]
EVAL_SAMPLE_SIZE = int(os.getenv("GAIA_EVAL_SIZE", "30"))
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# ── Token minimization ────────────────────────────────────────────────────
MINIMIZE_TOKENS: bool = bool(
    os.getenv("MUTAGENAI_MINIMIZE_TOKENS", "0") not in ("0", "", "false")
)
TOKEN_WEIGHT: float = float(os.getenv("MUTAGENAI_TOKEN_WEIGHT", "0.10"))
EFFICIENCY_CAP: float = float(os.getenv("MUTAGENAI_EFFICIENCY_CAP", "2.0"))
ACCURACY_BAND: float = float(os.getenv("MUTAGENAI_ACCURACY_BAND", "2.0"))


# ─────────────────────────────────────────────────────────────────────────
# Default agent system prompt (baseline to evolve from)
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_PROMPT = """\
You are a general-purpose AI assistant capable of solving complex, \
multi-step tasks. You will be given a question that may require:
- Multi-step reasoning and planning
- Mathematical calculations
- Interpreting structured data (tables, lists)
- Synthesizing information from multiple sources
- Careful reading comprehension

Instructions:
1. Break the problem into clear steps.
2. For each step, show your reasoning.
3. If the question involves calculations, show your work.
4. If information is provided as context, use it carefully.
5. Double-check your final answer before responding.

Output format:
Reasoning: <your step-by-step reasoning>
Final Answer: <your concise final answer>

Important:
- The final answer must be EXACT and CONCISE — a number, name, date, \
or short phrase.
- Do not add units unless they are part of the expected answer format.
- Do not hedge — give a single definitive answer.
- If asked for a number, provide just the number (e.g. "42" not \
"the answer is 42").
- If asked for a name, provide just the name.
- If asked yes/no, answer "Yes" or "No".
"""

# ── Seed templates for evolution ──────────────────────────────────────────

SEED_TEMPLATES = [
    # 1. Default (structured multi-step)
    _DEFAULT_PROMPT,
    # 2. Minimal direct
    """\
Answer the question precisely. Think step by step if needed, then give \
a short exact answer on the last line.

Final Answer: <answer>
""",
    # 3. Planning-first agent
    """\
You are a methodical problem-solving agent. For each question:

Step 1 — PLAN: Identify what information/steps are needed.
Step 2 — EXECUTE: Work through each step systematically.
Step 3 — VERIFY: Check your answer makes sense.
Step 4 — ANSWER: Provide the exact final answer.

Final Answer: <concise answer>
""",
    # 4. Tool-aware agent (simulated)
    """\
You are an AI agent that solves tasks requiring multi-step reasoning. \
When given context or data, use it as if you retrieved it via tools \
(web search, file reading, calculation).

Approach:
- Identify what information is available vs what needs inference
- Chain facts across multiple sources to reach the answer
- Be precise with numbers, dates, and proper nouns

Final Answer: <exact answer>
""",
    # 5. Concise output-strict
    """\
Read the question carefully. Reason internally, then output ONLY:

Final Answer: <answer in fewest possible words>

No explanation. No hedging. Just the answer.
""",
    # 6. Evidence-synthesis agent
    """\
You solve multi-step questions by:
1. Extracting key facts from the provided context
2. Identifying relationships between facts
3. Reasoning step-by-step to a conclusion
4. Stating the exact answer

Be precise. Answer with the minimum words needed.

Final Answer: <answer>
""",
]

# ── Domain mutations for agentic reasoning ────────────────────────────────

DOMAIN_MUTATIONS = [
    "Add an explicit planning phase before reasoning",
    "Add instruction to verify the answer against available evidence",
    "Shorten to remove unnecessary scaffolding while keeping precision",
    "Add instruction to handle numerical questions (show calculation)",
    "Add instruction to handle yes/no questions differently",
    "Replace step-by-step with a single direct-answer instruction",
    "Add instruction to identify question type before answering",
    "Add constraint: answer must match the expected format exactly",
    "Remove output formatting instructions to reduce tokens",
    "Add instruction to extract entities before reasoning",
    "Merge planning and execution into a single compact instruction",
    "Add instruction to handle multi-source synthesis tasks",
    "Add explicit instruction about answer conciseness",
    "Add self-correction step: re-read question after drafting answer",
]


# ─────────────────────────────────────────────────────────────────────────
# GAIA data structures and loading
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class GAIAExample:
    """A single GAIA benchmark example."""

    task_id: str
    question: str
    final_answer: str
    level: int
    file_name: str  # empty if no file needed
    annotator_steps: int  # number of steps annotator needed


def load_gaia_data(
    levels: list[int] | None = None,
    max_examples: int = 200,
    seed: int = 42,
) -> list[GAIAExample]:
    """Load GAIA validation examples from HuggingFace.

    Falls back to bundled sample if datasets library is unavailable.
    """
    if levels is None:
        levels = [1]

    examples: list[GAIAExample] = []

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "gaia-benchmark/GAIA", "2023_all", split="validation",
            trust_remote_code=True,
        )

        for row in ds:
            level = int(row.get("Level", row.get("level", 0)))
            if level not in levels:
                continue

            file_name = row.get("file_name", "") or ""

            # Parse annotator metadata for step count
            metadata = row.get("Annotator Metadata", {}) or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            raw_steps = metadata.get(
                "Steps", metadata.get("Number of steps", 1),
            ) if isinstance(metadata, dict) else 1
            try:
                steps = int(raw_steps)
            except (ValueError, TypeError):
                # Steps field sometimes contains descriptive text
                steps = 1

            examples.append(GAIAExample(
                task_id=str(row.get("task_id", "")),
                question=str(row.get("Question", row.get("question", ""))),
                final_answer=str(row.get("Final answer", row.get("final_answer", ""))),
                level=level,
                file_name=file_name,
                annotator_steps=steps,
            ))

        # Subsample if too many
        if len(examples) > max_examples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(examples), size=max_examples, replace=False)
            examples = [examples[int(i)] for i in indices]

    except (ImportError, Exception) as e:
        print(f"    Warning: Could not load from HuggingFace ({e})")
        print("    Install: uv pip install datasets")
        print("    Falling back to bundled sample...")
        examples = _load_bundled_sample(levels)

    return examples


def _load_bundled_sample(levels: list[int] | None = None) -> list[GAIAExample]:
    """Bundled GAIA-style examples for offline testing.

    These are representative multi-step reasoning questions similar
    to GAIA Level 1 (answerable via reasoning, no tools needed).
    """
    if levels is None:
        levels = [1]

    all_examples = [
        # Arithmetic/reasoning (Level 1 style)
        GAIAExample(
            task_id="bundled_01",
            question="A train travels at 60 mph for 2 hours, then at 80 mph "
                     "for 1.5 hours. What is the total distance traveled in miles?",
            final_answer="240",
            level=1,
            file_name="",
            annotator_steps=2,
        ),
        GAIAExample(
            task_id="bundled_02",
            question="If January 1st, 2024 is a Monday, what day of the week "
                     "is March 1st, 2024?",
            final_answer="Friday",
            level=1,
            file_name="",
            annotator_steps=2,
        ),
        GAIAExample(
            task_id="bundled_03",
            question="A recipe requires 2/3 cup of sugar. If you want to make "
                     "1.5 times the recipe, how many cups of sugar do you need? "
                     "Express as a decimal.",
            final_answer="1.0",
            level=1,
            file_name="",
            annotator_steps=2,
        ),
        # Multi-hop reasoning (Level 1 style)
        GAIAExample(
            task_id="bundled_04",
            question="The Eiffel Tower was completed in 1889. The Empire State "
                     "Building was completed in 1931. How many years apart were "
                     "they completed?",
            final_answer="42",
            level=1,
            file_name="",
            annotator_steps=1,
        ),
        GAIAExample(
            task_id="bundled_05",
            question="In a class of 30 students, 18 play football, 14 play "
                     "basketball, and 6 play both. How many students play "
                     "neither sport?",
            final_answer="4",
            level=1,
            file_name="",
            annotator_steps=3,
        ),
        # Data interpretation (Level 1 style)
        GAIAExample(
            task_id="bundled_06",
            question="A company's revenue was $2.4M in Q1, $3.1M in Q2, "
                     "$2.8M in Q3, and $3.7M in Q4. What was the total "
                     "annual revenue in millions of dollars?",
            final_answer="12.0",
            level=1,
            file_name="",
            annotator_steps=2,
        ),
        GAIAExample(
            task_id="bundled_07",
            question="A sequence follows the pattern: 2, 6, 18, 54, ... "
                     "What is the 6th term?",
            final_answer="486",
            level=1,
            file_name="",
            annotator_steps=2,
        ),
        # Level 2 style (more complex multi-step)
        GAIAExample(
            task_id="bundled_08",
            question="A store offers 20% off all items. An additional 10% "
                     "loyalty discount is applied after the first discount. "
                     "If an item originally costs $150, what is the final "
                     "price in dollars?",
            final_answer="108",
            level=2,
            file_name="",
            annotator_steps=3,
        ),
        GAIAExample(
            task_id="bundled_09",
            question="Three friends split a $87 dinner bill. They add a 20% "
                     "tip before splitting. How much does each person pay "
                     "in dollars, rounded to the nearest cent?",
            final_answer="34.80",
            level=2,
            file_name="",
            annotator_steps=3,
        ),
        GAIAExample(
            task_id="bundled_10",
            question="A car's fuel efficiency is 32 miles per gallon. Gas "
                     "costs $3.50 per gallon. How much does it cost to drive "
                     "240 miles? Give your answer in dollars.",
            final_answer="26.25",
            level=2,
            file_name="",
            annotator_steps=3,
        ),
    ]

    return [ex for ex in all_examples if ex.level in levels]


# ─────────────────────────────────────────────────────────────────────────
# Answer normalization and scoring (GAIA official approach)
# ─────────────────────────────────────────────────────────────────────────


def _normalize_answer(answer: str) -> str:
    """Normalize answer string for GAIA exact-match comparison.

    Based on the GAIA official evaluation:
    - Strip whitespace
    - Lowercase
    - Remove articles (a, an, the)
    - Remove trailing punctuation
    - Normalize number formats
    """
    s = answer.strip().lower()

    # Remove common prefixes from model responses
    for prefix in ("the answer is", "answer:", "final answer:", "result:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Remove trailing punctuation
    s = s.rstrip(".,;:!?")

    # Normalize whitespace
    s = " ".join(s.split())

    # Normalize number formats (remove trailing zeros after decimal)
    number_match = re.match(r"^-?\d+\.?\d*$", s)
    if number_match:
        try:
            num = float(s)
            if num == int(num):
                s = str(int(num))
            else:
                s = f"{num:g}"
        except ValueError:
            pass

    return s


def score_exact_match(prediction: str, ground_truth: str) -> float:
    """Score prediction against ground truth using GAIA exact match.

    Returns 1.0 for correct, 0.0 for incorrect.
    """
    norm_pred = _normalize_answer(prediction)
    norm_gold = _normalize_answer(ground_truth)

    if norm_pred == norm_gold:
        return 1.0

    # Try numeric comparison for number answers
    try:
        pred_num = float(norm_pred)
        gold_num = float(norm_gold)
        if abs(pred_num - gold_num) < 1e-6:
            return 1.0
    except ValueError:
        pass

    # Partial credit: check if gold is contained in prediction
    # (handles cases like "Paris, France" when answer is "Paris")
    if norm_gold in norm_pred and len(norm_gold) > 2:
        return 0.5

    return 0.0


def extract_final_answer(response: str) -> str:
    """Extract the final answer from model response.

    Looks for 'Final Answer:', 'Answer:', or falls back to last line.
    """
    # Try explicit patterns
    for pattern in [
        r"Final\s*Answer:\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
        r"ANSWER:\s*(.+?)(?:\n|$)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: last non-empty line
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


# ─────────────────────────────────────────────────────────────────────────
# Custom evolver for GAIA
# ─────────────────────────────────────────────────────────────────────────


class GAIAEvolver(PromptEvolver):
    """PromptEvolver subclass that scores via GAIA exact-match accuracy."""

    def __init__(
        self,
        config: PromptEvolverConfig,
        examples: list[GAIAExample],
        client: LLMClient,
    ) -> None:
        super().__init__(tools=[], eval_dataset=[], config=config)
        self._examples = examples
        self._gaia_client = client
        self._gaia_rng = np.random.default_rng(42)
        self._client = client
        self._require_tool_schemas = False

    def _evaluate_candidate(self, candidate: PromptCandidate) -> float:
        """Evaluate candidate prompt on a sample of GAIA examples."""
        sample_size = min(EVAL_SAMPLE_SIZE, len(self._examples))
        indices = self._gaia_rng.choice(
            len(self._examples), size=sample_size, replace=False,
        )
        sample = [self._examples[int(i)] for i in indices]

        scores: list[float] = []

        for ex in sample:
            user_msg = ex.question

            try:
                response = self._gaia_client.complete(
                    system_prompt=candidate.template,
                    user_message=user_msg,
                    temperature=candidate.temperature,
                    top_p=candidate.top_p,
                )
                if response is None:
                    scores.append(0.0)
                    continue
                predicted = extract_final_answer(response)
                score = score_exact_match(predicted, ex.final_answer)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        raw_accuracy = (sum(scores) / len(scores)) * 100.0 if scores else 0.0
        return self._apply_token_efficiency(raw_accuracy, candidate)


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────


def evaluate_baseline(
    examples: list[GAIAExample],
    prompt: str,
    client: LLMClient,
    max_samples: int = 30,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate a prompt on GAIA examples, returning (accuracy%, details)."""
    sample = examples[:max_samples]
    details: list[dict[str, Any]] = []

    for ex in sample:
        try:
            response = client.complete(
                system_prompt=prompt,
                user_message=ex.question,
                temperature=0.1,
                top_p=0.95,
            )
            if response is None:
                raise RuntimeError("LLM returned None")
            predicted = extract_final_answer(response)
            score = score_exact_match(predicted, ex.final_answer)
        except Exception as e:
            response = f"ERROR: {e}"
            predicted = ""
            score = 0.0

        details.append({
            "task_id": ex.task_id,
            "question": ex.question[:100] + "..." if len(ex.question) > 100 else ex.question,
            "gold_answer": ex.final_answer,
            "predicted": predicted,
            "score": score,
            "level": ex.level,
            "steps": ex.annotator_steps,
            "needs_file": bool(ex.file_name),
        })

    accuracy = (sum(d["score"] for d in details) / len(details)) * 100.0 if details else 0.0
    return accuracy, details


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  MutaGenAI — GAIA Agentic Benchmark Prompt Evolution")
    print("=" * 70)
    print()
    print(f"    Backend:        {BACKEND.value}")
    print(f"    Model:          {MODEL}")
    print(f"    Experiment:     {EXPERIMENT}")
    print(f"    Levels:         {GAIA_LEVELS}")
    print(f"    Eval samples:   {EVAL_SAMPLE_SIZE}")

    baseline_tokens = count_prompt_tokens(_DEFAULT_PROMPT)
    print(f"    Default tokens: {baseline_tokens}")
    if MINIMIZE_TOKENS:
        print(f"    Token opt:      ON (weight={TOKEN_WEIGHT}, "
              f"cap={EFFICIENCY_CAP}x, band={ACCURACY_BAND})")
    else:
        print("    Token opt:      OFF")
    print()

    # ── Load dataset ──────────────────────────────────────────────────
    print("  Phase 1: Loading GAIA data")
    print("  " + "-" * 60)

    all_examples = load_gaia_data(
        levels=GAIA_LEVELS,
        max_examples=200,
        seed=42,
    )
    print(f"    Loaded {len(all_examples)} examples")

    # ── Train / holdout split (60/40) — holdout is UNSEEN during evolution
    split_rng = np.random.default_rng(42)
    indices = split_rng.permutation(len(all_examples))
    split_point = max(1, int(len(all_examples) * 0.6))
    train_idx = indices[:split_point]
    holdout_idx = indices[split_point:]

    train_examples = [all_examples[int(i)] for i in train_idx]
    holdout_examples = [all_examples[int(i)] for i in holdout_idx]
    print(f"    Train:   {len(train_examples)} (used during evolution)")
    print(f"    Holdout: {len(holdout_examples)} (unseen — used for validation)")

    level_counts: dict[int, int] = {}
    file_count = 0
    for ex in all_examples:
        level_counts[ex.level] = level_counts.get(ex.level, 0) + 1
        if ex.file_name:
            file_count += 1

    for level, count in sorted(level_counts.items()):
        print(f"      Level {level}: {count} examples")
    print(f"      Needing files: {file_count} "
          f"(will score 0 without tool access)")
    print()

    # ── Configure LLM client ─────────────────────────────────────────
    if BACKEND == LLMBackend.OLLAMA:
        base_cfg = PromptEvolverConfig(
            backend=BACKEND,
            ollama_model=MODEL,
            timeout=120.0,
        )
    else:
        base_cfg = PromptEvolverConfig(
            backend=BACKEND,
            azure_deployment=MODEL,
            azure_use_rbac=True,
            timeout=60.0,
        )
    client = LLMClient(base_cfg)

    print("  Checking LLM backend...", end=" ", flush=True)
    if client.is_available():
        print("OK")
    else:
        print("UNAVAILABLE")
        print("    Cannot reach the LLM backend. Exiting.")
        return
    print()

    # ── Baseline evaluation ───────────────────────────────────────────
    print("  Phase 2: Baseline evaluation")
    print("  " + "-" * 60)

    baseline_score, baseline_details = evaluate_baseline(
        holdout_examples, _DEFAULT_PROMPT, client,
        max_samples=EVAL_SAMPLE_SIZE,
    )
    baseline_correct = sum(1 for d in baseline_details if d["score"] >= 1.0)

    print(f"    Accuracy:     {baseline_score:.1f}%")
    print(f"    Correct:      {baseline_correct}/{len(baseline_details)}")

    # Per-level breakdown
    for level in sorted(level_counts):
        level_dets = [d for d in baseline_details if d["level"] == level]
        if level_dets:
            avg = sum(d["score"] for d in level_dets) / len(level_dets) * 100
            print(f"    Level {level}:      {avg:.1f}% ({len(level_dets)} samples)")
    print()

    # ── Evolution ─────────────────────────────────────────────────────
    print("  Phase 3: Prompt evolution")
    print("  " + "-" * 60)

    if EXPERIMENT == "deep":
        config = PromptEvolverConfig(
            iterations=5,
            population_size=6,
            num_islands=3,
            elite_size=2,
            mutation_rate=0.7,
            crossover_rate=0.3,
            adaptive_mutations=True,
            llm_mutation_rate=0.3,
            refine_after_splice=True,
            describe_entities=False,
            minimize_tokens=MINIMIZE_TOKENS,
            token_weight=TOKEN_WEIGHT,
            token_efficiency_cap=EFFICIENCY_CAP,
            token_accuracy_band=ACCURACY_BAND,
            baseline_prompt_tokens=baseline_tokens,
            backend=BACKEND,
            **({"ollama_model": MODEL} if BACKEND == LLMBackend.OLLAMA else {
                "azure_deployment": MODEL,
                "azure_use_rbac": True,
            }),
            timeout=120.0,
        )
    else:  # standard
        config = PromptEvolverConfig(
            iterations=3,
            population_size=4,
            num_islands=2,
            elite_size=2,
            mutation_rate=0.7,
            crossover_rate=0.3,
            adaptive_mutations=True,
            llm_mutation_rate=0.2,
            refine_after_splice=False,
            describe_entities=False,
            minimize_tokens=MINIMIZE_TOKENS,
            token_weight=TOKEN_WEIGHT,
            token_efficiency_cap=EFFICIENCY_CAP,
            token_accuracy_band=ACCURACY_BAND,
            baseline_prompt_tokens=baseline_tokens,
            backend=BACKEND,
            **({"ollama_model": MODEL} if BACKEND == LLMBackend.OLLAMA else {
                "azure_deployment": MODEL,
                "azure_use_rbac": True,
            }),
            timeout=120.0,
        )

    print(f"    Config: {config.iterations} iterations, "
          f"pop {config.population_size}, "
          f"{config.num_islands} islands")
    print()

    # Inject seeds and mutations
    import MutaGenAI.prompt_evolver as _pe
    original_seeds = list(_pe._SEED_TEMPLATES)
    _pe._SEED_TEMPLATES = list(SEED_TEMPLATES)
    original_mutations = list(_pe._DOMAIN_MUTATIONS) if hasattr(_pe, "_DOMAIN_MUTATIONS") else []
    if hasattr(_pe, "_DOMAIN_MUTATIONS"):
        _pe._DOMAIN_MUTATIONS = list(DOMAIN_MUTATIONS)

    t0 = time.perf_counter()
    evolver = GAIAEvolver(config, train_examples, client)
    result = evolver.run()
    wall_time = time.perf_counter() - t0

    # Restore seeds
    _pe._SEED_TEMPLATES = original_seeds
    if hasattr(_pe, "_DOMAIN_MUTATIONS"):
        _pe._DOMAIN_MUTATIONS = original_mutations

    # ── Evaluate evolved prompt ───────────────────────────────────────
    print()
    print("  Phase 4: Evolved prompt evaluation")
    print("  " + "-" * 60)

    best_prompt = result.best_prompt
    evolved_tokens = count_prompt_tokens(best_prompt)

    evolved_score, evolved_details = evaluate_baseline(
        holdout_examples, best_prompt, client,
        max_samples=EVAL_SAMPLE_SIZE,
    )
    evolved_correct = sum(1 for d in evolved_details if d["score"] >= 1.0)

    print(f"    Accuracy:     {evolved_score:.1f}%")
    print(f"    Correct:      {evolved_correct}/{len(evolved_details)}")

    for level in sorted(level_counts):
        level_dets = [d for d in evolved_details if d["level"] == level]
        if level_dets:
            avg = sum(d["score"] for d in level_dets) / len(level_dets) * 100
            print(f"    Level {level}:      {avg:.1f}% ({len(level_dets)} samples)")
    print()

    # ── Results summary ───────────────────────────────────────────────
    print("  " + "=" * 60)
    print("  RESULTS")
    print("  " + "=" * 60)
    print(f"    Default baseline:   {baseline_score:.1f}% accuracy")
    print(f"    Evolved best:       {evolved_score:.1f}% accuracy")
    delta = evolved_score - baseline_score
    print(f"    Delta:              {delta:+.1f} pp")
    print(f"    Wall time:          {wall_time:.1f}s")
    print(f"    Total candidates:   {len(result.all_candidates)}")
    print(f"    Temperature:        {result.best_temperature:.4f}")
    print(f"    Top-p:              {result.best_top_p:.4f}")
    print(f"    Prompt tokens:      {baseline_tokens} -> {evolved_tokens} "
          f"({evolved_tokens - baseline_tokens:+d})")
    if MINIMIZE_TOKENS:
        print(f"    Token weight:       {TOKEN_WEIGHT}")
        print(f"    Efficiency cap:     {EFFICIENCY_CAP}x baseline")
        print(f"    Accuracy band:      {ACCURACY_BAND} (tiebreaker)")
        efficiency_ratio = baseline_tokens / max(evolved_tokens, 1)
        print(f"    Efficiency ratio:   {efficiency_ratio:.2f}x "
              f"({'shorter' if efficiency_ratio > 1 else 'longer'} than baseline)")
        token_saving_pct = (1 - evolved_tokens / max(1, baseline_tokens)) * 100
        print(f"    Token saving:       {token_saving_pct:+.1f}%")
    print()

    # ── Best prompt display ───────────────────────────────────────────
    print("  Best evolved prompt:")
    print("  " + "-" * 60)
    for line in best_prompt.split("\n"):
        print(f"    {line}")
    print("  " + "-" * 60)
    print()

    # ── Export lineage with token stats ───────────────────────────────
    lineage = result.lineage_json()
    for entry in lineage:
        tpl = entry.get("template", "")
        tokens = count_prompt_tokens(tpl)
        entry["prompt_tokens"] = tokens
        entry["efficiency_ratio"] = round(baseline_tokens / max(tokens, 1), 3)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    lineage_path = LOG_DIR / "gaia_lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)
    print(f"  Lineage data saved to {lineage_path}")
    print(f"    ({len(lineage)} candidates)")

    # ── Experiment log ────────────────────────────────────────────────
    experiment_log = {
        "experiment": f"gaia_{EXPERIMENT}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL,
        "backend": BACKEND.value,
        "dataset": "GAIA (gaia-benchmark/GAIA, 2023_all)",
        "reference": "https://huggingface.co/spaces/gaia-benchmark/leaderboard",
        "paper": "https://arxiv.org/abs/2311.12983",
        "config": {
            "iterations": config.iterations,
            "population_size": config.population_size,
            "num_islands": config.num_islands,
            "elite_size": config.elite_size,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "adaptive_mutations": config.adaptive_mutations,
            "llm_mutation_rate": config.llm_mutation_rate,
            "refine_after_splice": config.refine_after_splice,
            "eval_sample_size": EVAL_SAMPLE_SIZE,
            "levels": GAIA_LEVELS,
            "total_examples": len(all_examples),
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
        },
        "results": {
            "baseline_accuracy": round(baseline_score, 2),
            "evolved_accuracy": round(evolved_score, 2),
            "delta_pp": round(delta, 2),
            "baseline_correct": baseline_correct,
            "evolved_correct": evolved_correct,
            "wall_time_s": round(wall_time, 1),
            "total_candidates": len(result.all_candidates),
            "best_temperature": round(result.best_temperature, 4),
            "best_top_p": round(result.best_top_p, 4),
        },
        "best_prompt": best_prompt,
        "history": result.history,
        "lineage_size": len(lineage),
        "token_optimization": {
            "enabled": MINIMIZE_TOKENS,
            "strategy": (
                "baseline_relative_efficiency + lexicographic_tiebreaker"
                if MINIMIZE_TOKENS else "disabled"
            ),
            "token_weight": TOKEN_WEIGHT,
            "efficiency_cap": EFFICIENCY_CAP,
            "accuracy_band": ACCURACY_BAND,
            "baseline_tokens": baseline_tokens,
            "evolved_tokens": evolved_tokens,
            "token_delta": evolved_tokens - baseline_tokens,
            "efficiency_ratio": round(baseline_tokens / max(evolved_tokens, 1), 3),
        },
        "candidate_token_stats": {
            "min_tokens": min(e["prompt_tokens"] for e in lineage) if lineage else 0,
            "max_tokens": max(e["prompt_tokens"] for e in lineage) if lineage else 0,
            "mean_tokens": round(
                sum(e["prompt_tokens"] for e in lineage) / max(len(lineage), 1), 1,
            ),
        },
    }

    log_path = LOG_DIR / "gaia_evolution_log.json"
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"  Experiment log saved to {log_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
