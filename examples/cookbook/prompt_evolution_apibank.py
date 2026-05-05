#!/usr/bin/env python3
"""
Cookbook Recipe 46 — API-Bank Multi-Level Tool-Use Prompt Evolution
===================================================================

Evolves system prompts on the **API-Bank** benchmark (Alibaba DAMO Academy,
EMNLP 2023) — 73 runnable API tools across 314 tool-use dialogues with
753 API calls evaluated at three competency levels:

* **Level 1 — API Calling:** given the API description, generate a
  correct ``[ApiName(key1='value1', ...)]`` call.
* **Level 2 — API Retrieval + Calling:** given multiple candidate APIs,
  select the right one *and* call it correctly.
* **Level 3 — Planning + Retrieval + Calling:** multi-step orchestration
  across APIs with ToolSearcher for discovery.

Published results (Li et al., EMNLP 2023)
------------------------------------------
* GPT-4:     Call 83.8 % · Retrieval 41.2 % · Plan 38.8 %
* GPT-3.5:   Call 82.6 % · Retrieval 35.3 % · Plan 13.1 %
* Lynx (7B): Call 78.4 % · Retrieval 22.5 % · Plan 11.9 %

Algorithm experiments
---------------------
* **Standard** — 3 iterations, pop 4, 2 islands  (balanced)
* **Deep**     — 5 iterations, pop 5, 2 islands  (thorough)

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_apibank.py

The script saves an experiment log to ``apibank_experiment_log.json``
for dashboard consumption.

References
----------
* Repository : https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank
* Dataset    : https://huggingface.co/datasets/liminghao1630/API-Bank
* Paper      : https://arxiv.org/abs/2304.08244 (API-Bank)
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class APIBankCase:
    """A single API-Bank benchmark test case."""

    case_id: str
    instruction: str
    input_text: str
    expected_output: str
    level: int  # 1, 2, or 3
    source_file: str

    @property
    def expected_api_name(self) -> str:
        """Extract the expected API name from the expected output."""
        m = re.search(r"\[(\w+)\(", self.expected_output)
        return m.group(1) if m else ""

    @property
    def expected_params(self) -> dict[str, str]:
        """Extract expected parameters from the expected output."""
        return _parse_api_params(self.expected_output)

    @property
    def category(self) -> str:
        return f"level_{self.level}"


@dataclass
class APIBankExperiment:
    """Result container for one evolution run."""

    category: str
    algorithm: str
    backend: str
    n_cases: int
    baseline_score: float
    evolved_score: float
    best_prompt_template: str
    best_temperature: float
    best_top_p: float
    iterations: int
    wall_time: float
    history: list[tuple[int, float]] = field(default_factory=list)
    prompt_evolution: list[dict[str, Any]] = field(default_factory=list)
    api_name_accuracy: float = 0.0
    param_accuracy: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# API call parsing  (matches official evaluator regex)
# ─────────────────────────────────────────────────────────────────────────

_API_CALL_RE = re.compile(r"\[(\w+)\((.*?)\)\]", re.DOTALL)
_PARAM_RE = re.compile(r"(\w+)\s*=\s*'([^']*)'")


def _parse_api_call(text: str) -> tuple[str, str]:
    """Parse ``[ApiName(key='val', ...)]`` → (api_name, raw_params_str)."""
    m = _API_CALL_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _parse_api_params(text: str) -> dict[str, str]:
    """Extract ``key='value'`` pairs from an API call string."""
    _, param_str = _parse_api_call(text)
    if not param_str:
        return {}
    return dict(_PARAM_RE.findall(param_str))


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading from HuggingFace
# ─────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".apibank_cache"

_HF_REPO = "liminghao1630/API-Bank"
_HF_FILES = {
    "level_1": "test-data/level-1-api.json",
    "level_2": "test-data/level-2-api.json",
}


def _download_apibank_data() -> dict[str, list[dict[str, Any]]]:
    """Download API-Bank test data from HuggingFace.

    Returns a dict keyed by level name with lists of raw records.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / "benchmark_data.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    from huggingface_hub import hf_hub_download

    print("  Downloading API-Bank benchmark from HuggingFace...")
    all_levels: dict[str, list[dict[str, Any]]] = {}

    for level_name, hf_path in _HF_FILES.items():
        try:
            local_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename=hf_path,
                repo_type="dataset",
                cache_dir=str(_CACHE_DIR / "hf_cache"),
            )
            with open(local_path) as f:
                records = json.load(f)
            all_levels[level_name] = records
            print(f"    {level_name}: {len(records)} cases")
        except Exception as exc:
            print(f"    ⚠ Failed to load {level_name}: {exc}")

    if all_levels:
        with open(cache_file, "w") as f:
            json.dump(all_levels, f)

    return all_levels


def load_apibank_dataset(
    max_per_level: int = 30,
    levels: Optional[list[str]] = None,
) -> dict[str, list[APIBankCase]]:
    """Load and group API-Bank benchmark data.

    Parameters
    ----------
    max_per_level : int
        Maximum cases to load per evaluation level.
    levels : list[str] | None
        Which levels to include. Defaults to all available.

    Returns
    -------
    dict mapping level name to list of APIBankCase
    """
    if levels is None:
        levels = list(_HF_FILES.keys())

    raw = _download_apibank_data()
    rng = np.random.default_rng(42)

    by_level: dict[str, list[APIBankCase]] = {}

    for level_name in levels:
        records = raw.get(level_name, [])
        if not records:
            continue

        level_num = int(level_name.split("_")[1])

        # Subsample
        if len(records) > max_per_level:
            indices = rng.choice(len(records), size=max_per_level, replace=False)
            records = [records[int(i)] for i in indices]

        cases = []
        for idx, rec in enumerate(records):
            instruction = rec.get("instruction", "")
            input_text = rec.get("input", "")
            expected = rec.get("expected_output", "")
            source = rec.get("file", "")

            if not input_text or not expected:
                continue

            cases.append(APIBankCase(
                case_id=f"{level_name}_{rec.get('id', idx)}",
                instruction=instruction,
                input_text=input_text,
                expected_output=expected,
                level=level_num,
                source_file=source,
            ))

        if cases:
            by_level[level_name] = cases

    return by_level


# ─────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────


def _normalise(name: str) -> str:
    """Normalise an API name for comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def score_apibank_case(response: str, case: APIBankCase) -> tuple[float, dict[str, Any]]:
    """Score a response against an API-Bank test case.

    Returns (score 0.0–1.0, detail dict).

    Scoring components:
    - API name match    (0.4 weight): did the model pick the right API?
    - Parameter match   (0.4 weight): are param names & values correct?
    - Format compliance (0.2 weight): is the output in [Api(...)] format?
    """
    detail: dict[str, Any] = {
        "api_name_match": False,
        "param_score": 0.0,
        "format_ok": False,
        "error": None,
    }

    # Parse the model response
    pred_name, pred_params_str = _parse_api_call(response)
    pred_params = dict(_PARAM_RE.findall(pred_params_str)) if pred_params_str else {}

    # Parse expected
    exp_name = case.expected_api_name
    exp_params = case.expected_params

    # Format compliance
    format_ok = bool(pred_name)
    detail["format_ok"] = format_ok
    format_score = 1.0 if format_ok else 0.0

    if not format_ok:
        detail["error"] = "NO_API_CALL"
        return 0.0, detail

    # API name match
    name_match = _normalise(pred_name) == _normalise(exp_name)
    detail["api_name_match"] = name_match
    name_score = 1.0 if name_match else 0.0

    if not name_match:
        detail["error"] = "API_NAME_MISMATCH"

    # Parameter match
    if exp_params:
        total_params = len(exp_params)
        matched = 0
        for key, exp_val in exp_params.items():
            pred_val = pred_params.get(key, "")
            if _normalise(pred_val) == _normalise(exp_val):
                matched += 1
            elif key in pred_params:
                # partial credit for having the right key
                matched += 0.3
        param_score = matched / total_params
    else:
        # No params expected — full credit if none predicted either
        param_score = 1.0 if not pred_params else 0.5

    detail["param_score"] = param_score

    return 0.4 * name_score + 0.4 * param_score + 0.2 * format_score, detail


# ─────────────────────────────────────────────────────────────────────────
# Seed prompt templates
# ─────────────────────────────────────────────────────────────────────────

_APIBANK_SEED_TEMPLATES = [
    # T1 — Minimal
    textwrap.dedent("""\
        You are an API-calling assistant. Given a conversation and API \
descriptions, generate the correct API request.

{apibank_instruction}

{apibank_input}
"""),

    # T2 — Step-by-step
    textwrap.dedent("""\
        You are an expert API assistant. Follow these steps:
1. Read the conversation history carefully.
2. Identify which API to call based on the context.
3. Extract the correct parameter values from the conversation.
4. Output ONLY the API request in the format: [ApiName(key1='value1', key2='value2', ...)]

{apibank_instruction}

{apibank_input}
"""),

    # T3 — Constraint-focused
    textwrap.dedent("""\
        You are a precise API-calling agent. Rules:
- Output EXACTLY ONE API call in the format [ApiName(key1='value1', ...)]
- Match API names EXACTLY as described.
- Extract parameter values from the conversation — do NOT invent values.
- Include ALL required parameters.
- Output ONLY the API-Request line, nothing else.

{apibank_instruction}

{apibank_input}
"""),

    # T4 — Reasoning-focused
    textwrap.dedent("""\
        You are an API orchestration expert. Your task:
1. Analyse what the user needs from the conversation.
2. Match the need to the correct API from the descriptions provided.
3. Extract exact parameter values from the dialogue.
4. Generate the API request in format: [ApiName(key1='value1', key2='value2')]

CRITICAL: The parameter values must come from the conversation, not your imagination.
Output ONLY: API-Request: [ApiName(key1='value1', ...)]

{apibank_instruction}

{apibank_input}
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# Mutation operators
# ─────────────────────────────────────────────────────────────────────────

_APIBANK_MUTATIONS: list[str] = [
    # Parameter extraction
    "Add: 'Extract parameter values verbatim from the user's messages — do not paraphrase.'",
    "Append: 'Pay attention to date formats, IDs, and quoted strings in the conversation.'",
    "Insert: 'If the user provides a name, use it exactly as spelled.'",
    # API selection
    "Add: 'Read each API description carefully and match it to the user's intent.'",
    "Inject: 'When multiple APIs are available, choose the one whose description best fits.'",
    "Insert: 'The API name must match exactly — check spelling.'",
    # Format compliance
    "Append: 'Output ONLY [ApiName(key1=\\'value1\\', ...)] — no other text.'",
    "Add: 'Use single quotes around parameter values, not double quotes.'",
    "Insert: 'Do not add extra whitespace inside the API call brackets.'",
    # Multi-step reasoning
    "Add: 'If the conversation shows a prior API result, use that result in your next call.'",
    "Inject: 'For authentication flows, call GetUserToken first with the provided credentials.'",
    "Append: 'Track token values from previous API responses for use in subsequent calls.'",
    # Context understanding
    "Prepend: 'Read the ENTIRE conversation before generating the API call.'",
    "Insert: 'The most recent user message often contains the key information.'",
    "Add: 'Previous API-Request/Response pairs show what has already been done.'",
    # Chain-of-thought
    "Prepend: 'Think step by step about which API to call and with what parameters.'",
    "Insert: 'First identify the API, then list required parameters, then fill values.'",
    "Add: 'Reason about the user intent before outputting the API request.'",
]


def _mutate_apibank_template(
    template: str, rng: np.random.Generator, rate: float = 0.5
) -> str:
    """Apply a random mutation to an API-Bank prompt template."""
    mutation = rng.choice(_APIBANK_MUTATIONS)
    lines = template.strip().split("\n")

    action = mutation.split(":")[0].strip().lower()
    content = mutation.split(":", 1)[1].strip().strip("'\"")

    if "prepend" in action:
        lines.insert(0, content)
    elif "append" in action:
        lines.append(content)
    elif "inject" in action or "insert" in action:
        pos = int(rng.integers(1, max(2, len(lines))))
        lines.insert(pos, content)
    elif "add" in action:
        pos = int(rng.integers(len(lines) // 2, max(len(lines) // 2 + 1, len(lines))))
        lines.insert(pos, content)
    else:
        lines.append(content)

    result = "\n".join(lines)

    # Ensure placeholders survive
    if "{apibank_instruction}" not in result:
        result += "\n\n{apibank_instruction}"
    if "{apibank_input}" not in result:
        result += "\n\n{apibank_input}"

    return result


def _crossover_apibank_templates(
    a: str, b: str, rng: np.random.Generator
) -> str:
    """Crossover two API-Bank prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{apibank_instruction}" not in child:
        child += "\n\n{apibank_instruction}"
    if "{apibank_input}" not in child:
        child += "\n\n{apibank_input}"
    return child


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine
# ─────────────────────────────────────────────────────────────────────────


def run_apibank_evolution(
    cases: list[APIBankCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> APIBankExperiment:
    """Run prompt evolution on API-Bank test cases.

    Returns an APIBankExperiment with full tracking.
    """
    rng = np.random.default_rng(seed)
    category = cases[0].category if cases else "unknown"

    # ── Evaluate a candidate ───────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate, eval_cases: list[APIBankCase]
    ) -> tuple[float, float, float]:
        """Returns (overall_score, api_name_acc, param_acc)."""
        total = 0.0
        name_hits = 0
        param_total = 0.0
        for case in eval_cases:
            sys_prompt = candidate.template.replace(
                "{apibank_instruction}", case.instruction
            ).replace(
                "{apibank_input}", case.input_text
            )

            response = client.complete(
                system_prompt=sys_prompt,
                user_message="Generate API Request:",
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.1))
                continue

            score, detail = score_apibank_case(response, case)
            total += score
            if detail["api_name_match"]:
                name_hits += 1
            param_total += detail["param_score"]

        n = len(eval_cases) if eval_cases else 1
        return (
            (total / n * 100.0),
            (name_hits / n * 100.0),
            (param_total / n * 100.0),
        )

    # ── Subsample for evaluation ───────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(cases):
        eval_indices = rng.choice(
            len(cases), size=config.eval_sample_size, replace=False
        )
        eval_cases = [cases[int(i)] for i in eval_indices]
    else:
        eval_cases = cases

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  API-Bank Level: {category}  |  Algorithm: {algorithm_name}")
        print(
            f"  Cases: {len(eval_cases)}  |  Backend: {config.backend.value}"
        )
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    best_name_acc = 0.0
    best_param_acc = 0.0

    for i, tmpl in enumerate(_APIBANK_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        score, name_acc, param_acc = evaluate(cand, eval_cases)
        cand.score = score
        islands[i % config.num_islands].append(cand)

        if score > best_name_acc:
            best_name_acc = name_acc
            best_param_acc = param_acc

        prompt_trace.append({
            "generation": 0,
            "score": round(score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Baseline best: {baseline_score:.1f}%")

    # ── Evolution loop ─────────────────────────────────────────────────
    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                k = min(3, len(island))
                idxs = rng.choice(len(island), size=k, replace=False)
                parent_a = max(
                    (island[int(i)] for i in idxs), key=lambda c: c.score
                )

                if rng.random() < config.crossover_rate and len(island) > 1:
                    idxs2 = rng.choice(len(island), size=k, replace=False)
                    parent_b = max(
                        (island[int(i)] for i in idxs2), key=lambda c: c.score
                    )
                    child_tmpl = _crossover_apibank_templates(
                        parent_a.template, parent_b.template, rng
                    )
                else:
                    child_tmpl = parent_a.template

                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_apibank_template(
                        child_tmpl, rng, config.mutation_rate
                    )

                temp = parent_a.temperature + float(rng.normal(0, 0.1))
                temp = float(np.clip(temp, *config.temperature_range))
                top_p = parent_a.top_p + float(rng.normal(0, 0.05))
                top_p = float(np.clip(top_p, *config.top_p_range))

                child = PromptCandidate(
                    template=child_tmpl,
                    temperature=temp,
                    top_p=top_p,
                    generation=gen,
                )
                score, name_acc, param_acc = evaluate(child, eval_cases)
                child.score = score
                new_cands.append(child)

                if score > best_overall.score:
                    best_name_acc = name_acc
                    best_param_acc = param_acc

            combined = island + new_cands
            combined.sort(key=lambda c: c.score, reverse=True)
            islands[isl_id] = combined[: config.elite_size]

        # Migration every 3 gens
        if gen % 3 == 0 and config.num_islands > 1:
            for src in range(config.num_islands):
                if not islands[src]:
                    continue
                best_src = max(islands[src], key=lambda c: c.score)
                dest = (src + 1) % config.num_islands
                migrant = PromptCandidate(
                    template=best_src.template,
                    temperature=best_src.temperature,
                    top_p=best_src.top_p,
                    generation=best_src.generation,
                    score=best_src.score,
                )
                islands[dest].append(migrant)

        gen_best = max(
            (c for isl in islands for c in isl), key=lambda c: c.score
        )
        if gen_best.score > best_overall.score:
            best_overall = copy.deepcopy(gen_best)

        history.append((gen, best_overall.score))

        prompt_trace.append({
            "generation": gen,
            "score": round(best_overall.score, 1),
            "template_hash": best_overall.hash,
            "template_preview": best_overall.template[:120].replace("\n", " "),
        })

        if verbose:
            print(
                f"  Gen {gen:2d}/{config.iterations}  "
                f"best={best_overall.score:5.1f}%  "
                f"temp={best_overall.temperature:.3f}  "
                f"top_p={best_overall.top_p:.3f}"
            )

    wall_time = time.perf_counter() - t0

    return APIBankExperiment(
        category=category,
        algorithm=algorithm_name,
        backend=config.backend.value,
        n_cases=len(eval_cases),
        baseline_score=baseline_score,
        evolved_score=best_overall.score,
        best_prompt_template=best_overall.template,
        best_temperature=best_overall.temperature,
        best_top_p=best_overall.top_p,
        iterations=config.iterations,
        wall_time=wall_time,
        history=history,
        prompt_evolution=prompt_trace,
        api_name_accuracy=best_name_acc,
        param_accuracy=best_param_acc,
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_APIBANK_DEFAULT_PROMPT = textwrap.dedent("""\
    You are a helpful assistant. Given the conversation history and API \
descriptions, generate the correct API request in the format \
[ApiName(key1='value1', ...)].

{apibank_instruction}

{apibank_input}
""")


def evaluate_baseline(
    cases: list[APIBankCase],
    level_name: str,
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate the default API-Bank prompt on cases."""
    if verbose:
        print(f"  Evaluating default prompt baseline ({level_name})...")

    rng = np.random.default_rng(42)
    total = 0.0
    for case in cases:
        sys_prompt = _APIBANK_DEFAULT_PROMPT.replace(
            "{apibank_instruction}", case.instruction
        ).replace(
            "{apibank_input}", case.input_text
        )

        response = client.complete(
            system_prompt=sys_prompt,
            user_message="Generate API Request:",
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            total += float(rng.uniform(0, 0.1))
            continue

        score, _ = score_apibank_case(response, case)
        total += score

    result = (total / len(cases) * 100.0) if cases else 0.0
    if verbose:
        print(f"  Default prompt baseline ({level_name}): {result:.1f}%")
    return result


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: APIBankExperiment) -> None:
    """Print the prompt evolution trace for an experiment."""
    print(f"\n{'─' * 60}")
    print(f"  Prompt Evolution — {experiment.category} ({experiment.algorithm})")
    print(f"{'─' * 60}")

    seen_hashes: set[str] = set()
    for entry in experiment.prompt_evolution:
        h = entry["template_hash"]
        marker = " *NEW*" if h not in seen_hashes else ""
        seen_hashes.add(h)
        print(
            f"  Gen {entry['generation']:2d}  "
            f"Score {entry['score']:5.1f}%  "
            f"[{h[:8]}]{marker}"
        )
        if marker:
            print(f"    → {entry['template_preview']}")

    print(
        f"\n  Baseline: {experiment.baseline_score:.1f}%  →  "
        f"Evolved: {experiment.evolved_score:.1f}%  "
        f"(Δ{experiment.evolved_score - experiment.baseline_score:+.1f}%)"
    )
    print(
        f"  API Name Accuracy: {experiment.api_name_accuracy:.1f}%  |  "
        f"Param Accuracy: {experiment.param_accuracy:.1f}%"
    )


def show_results_table(
    experiments: list[APIBankExperiment],
    default_baselines: dict[str, float],
) -> None:
    """Print a summary table of all experiments."""
    print(f"\n{'=' * 90}")
    print("  API-Bank Prompt Evolution Results")
    print(f"{'=' * 90}")
    print(
        f"  {'Level':<14} {'Algorithm':<12} {'Backend':<14} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}"
    )
    print(f"  {'─' * 85}")

    for exp in experiments:
        dflt = default_baselines.get(exp.category, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.category:<14} {exp.algorithm:<12} {exp.backend:<14} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s"
        )

    print(f"  {'─' * 85}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[APIBankExperiment],
    default_baselines: dict[str, float],
    path: str = "apibank_experiment_log.json",
) -> None:
    """Save experiment results to JSON for dashboard consumption."""
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_baselines": default_baselines,
        "experiments": [],
    }
    for exp in experiments:
        entry = {
            "category": exp.category,
            "algorithm": exp.algorithm,
            "backend": exp.backend,
            "n_cases": exp.n_cases,
            "baseline_score": round(exp.baseline_score, 2),
            "evolved_score": round(exp.evolved_score, 2),
            "delta": round(exp.evolved_score - exp.baseline_score, 2),
            "best_temperature": round(exp.best_temperature, 4),
            "best_top_p": round(exp.best_top_p, 4),
            "iterations": exp.iterations,
            "wall_time": round(exp.wall_time, 1),
            "history": exp.history,
            "prompt_evolution": exp.prompt_evolution,
            "best_prompt_template": exp.best_prompt_template,
            "api_name_accuracy": round(exp.api_name_accuracy, 2),
            "param_accuracy": round(exp.param_accuracy, 2),
        }
        log["experiments"].append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Experiment log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main — run all experiments
# ─────────────────────────────────────────────────────────────────────────

ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {
        "iterations": 3,
        "population_size": 4,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.6,
        "crossover_rate": 0.3,
        "eval_sample_size": 10,
    },
    "deep": {
        "iterations": 5,
        "population_size": 5,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.5,
        "crossover_rate": 0.4,
        "eval_sample_size": 12,
    },
}

BENCHMARK_LEVELS = ["level_1", "level_2"]
MAX_PER_LEVEL = 30


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  EvoSim × API-Bank — Multi-Level Tool-Use Prompt Evolution   ║
    ║  73 APIs · 314 dialogues · 753 API calls · 3 levels          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Backend setup ──────────────────────────────────────────────────
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=60.0,
    )
    ollama_client = LLMClient(ollama_cfg)

    if not ollama_client.is_available():
        print("  ⚠ Ollama not available — ensure it is running at localhost:11434")
        print("  Running in mock mode (random scores) for demonstration.\n")

    # ── Load API-Bank data ─────────────────────────────────────────────
    print("  Loading API-Bank benchmark data (downloading from HuggingFace if needed)...")
    try:
        by_level = load_apibank_dataset(
            max_per_level=MAX_PER_LEVEL,
            levels=BENCHMARK_LEVELS,
        )
    except Exception as exc:
        print(f"  ✗ {exc}")
        return

    for level_name in BENCHMARK_LEVELS:
        cases = by_level.get(level_name, [])
        print(f"    {level_name:<14}: {len(cases):3d} cases")

    if not by_level:
        print("  No API-Bank data loaded — exiting.")
        return

    # ── Phase 1: Default prompt baselines ──────────────────────────────
    print("\n  Phase 1: Default Prompt Baselines")
    print("  " + "─" * 50)
    default_baselines: dict[str, float] = {}
    for level_name in BENCHMARK_LEVELS:
        cases = by_level.get(level_name, [])
        if cases:
            score = evaluate_baseline(cases, level_name, ollama_client)
            default_baselines[level_name] = score

    # ── Phase 2: Ollama evolution ──────────────────────────────────────
    print("\n  Phase 2: Prompt Evolution (Ollama)")
    print("  " + "─" * 50)

    all_experiments: list[APIBankExperiment] = []

    # Standard on all levels
    for level_name in BENCHMARK_LEVELS:
        cases = by_level.get(level_name, [])
        if not cases:
            continue

        algo = "standard"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            **algo_params,
        )

        exp = run_apibank_evolution(
            cases=cases,
            client=ollama_client,
            config=cfg,
            algorithm_name=algo,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # Deep on level_2 (hardest — requires retrieval + calling)
    if "level_2" in by_level and by_level["level_2"]:
        algo = "deep"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            **algo_params,
        )

        exp = run_apibank_evolution(
            cases=by_level["level_2"],
            client=ollama_client,
            config=cfg,
            algorithm_name=algo,
            seed=123,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # ── Phase 3: Azure OpenAI (GPT-4.1) ───────────────────────────────
    print("\n  Phase 3: Azure OpenAI (GPT-4.1) Prompt Evolution")
    print("  " + "─" * 50)

    azure_cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        timeout=30.0,
    )
    azure_client = LLMClient(azure_cfg)
    azure_available = azure_client.is_available()

    if not azure_available:
        print(
            "  ⚠ Azure OpenAI not available — set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_DEPLOYMENT env vars."
        )
        print("  Skipping Azure OpenAI experiments.\n")
    else:
        print(
            f"  ✓ Azure OpenAI connected: {azure_cfg.azure_deployment} "
            f"(RBAC={azure_cfg.azure_use_rbac})"
        )

        # Azure baselines
        for level_name in BENCHMARK_LEVELS:
            cases = by_level.get(level_name, [])
            if cases:
                score = evaluate_baseline(cases, level_name, azure_client)
                default_baselines[f"{level_name}_azure"] = score

        # Standard evolution on all levels
        for level_name in BENCHMARK_LEVELS:
            cases = by_level.get(level_name, [])
            if not cases:
                continue

            algo = "standard"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_apibank_evolution(
                cases=cases,
                client=azure_client,
                config=az_cfg,
                algorithm_name=algo,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

        # Deep on level_2
        if "level_2" in by_level and by_level["level_2"]:
            algo = "deep"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_apibank_evolution(
                cases=by_level["level_2"],
                client=azure_client,
                config=az_cfg,
                algorithm_name=algo,
                seed=123,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

    # ── Results ────────────────────────────────────────────────────────
    show_results_table(all_experiments, default_baselines)

    # ── Save experiment log ────────────────────────────────────────────
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "apibank_experiment_log.json"
    )
    save_experiment_log(all_experiments, default_baselines, log_path)

    # ── Print best evolved prompt ──────────────────────────────────────
    if all_experiments:
        best_exp = max(all_experiments, key=lambda e: e.evolved_score)
        print(f"\n{'=' * 60}")
        print(
            f"  Best Evolved Prompt ({best_exp.category} / {best_exp.algorithm})"
        )
        print(f"  Score: {best_exp.evolved_score:.1f}%")
        print(
            f"  API Name Acc: {best_exp.api_name_accuracy:.1f}%  |  "
            f"Param Acc: {best_exp.param_accuracy:.1f}%"
        )
        print(f"{'=' * 60}")
        preview = best_exp.best_prompt_template[:500]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(best_exp.best_prompt_template) > 500:
            print(
                f"    ... ({len(best_exp.best_prompt_template)} chars total)"
            )

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
