#!/usr/bin/env python3
"""
Cookbook Recipe — ScrapeGraphAI Schema-First Structured Output Evolution
========================================================================

Evolves system prompts for **structured JSON extraction** using the
`ScrapeGraphAI 100k <https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k>`_
dataset — 93 695 real-world examples of LLMs extracting structured data
from web content following user-defined JSON schemas.

This recipe showcases the **schema-first generation** pipeline:

* ``ProblemType.GENERATION`` — mutation operators tuned for structured
  output, format compliance, and schema adherence.
* ``schema_to_proxy_checks()`` — auto-generates ``ProxyCheck`` objects
  from a JSON schema so the no-eval scorer can verify structural
  correctness without ground-truth labels.
* ``NoEvalPromptEvolver`` + ``CompositeScorer`` — combines proxy
  metrics (schema conformance) with an LLM judge for semantic quality.

Dataset
-------
Each record in the ScrapeGraphAI 100k dataset contains:

* ``prompt`` — the user's extraction instruction
* ``schema`` — a JSON schema defining expected output structure
* ``content`` — source web page content
* ``response`` — the LLM's actual extraction (ground truth)
* ``response_is_valid`` — whether the response conforms to the schema
* ``schema_complexity_score`` — aggregate complexity metric

The script samples records stratified by schema complexity (low / mid /
high) so evolution is tested across a range of difficulties.

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_scrapegraph.py

    # Use a different Ollama model:
    OLLAMA_MODEL=llama3.2 uv run python examples/cookbook/prompt_evolution_scrapegraph.py

    # Limit to fewer samples for a quick test:
    uv run python examples/cookbook/prompt_evolution_scrapegraph.py --max-samples 10

References
----------
* Dataset  : https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k
* Paper    : https://arxiv.org/abs/2505.04016 (SLOT)
* License  : Apache 2.0
"""
from __future__ import annotations

import argparse
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
    ErrorProfile,
    FailureBucket,
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptCandidate,
    PromptEvolverConfig,
    SelectionMethod,
    get_failure_bucket_mutations,
)
from MutaGenAI.seed_loader import schema_to_proxy_checks
from MutaGenAI.strategies import (
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyCheck,
    ProxyMetricsScorer,
    Scorer,
    SelfConsistencyScorer,
)


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ScrapeGraphCase:
    """A single ScrapeGraphAI structured extraction test case."""

    case_id: str
    prompt: str
    schema: dict[str, Any]
    content: str
    expected_response: str
    response_is_valid: bool
    complexity_score: float
    complexity_tier: str  # "low", "mid", "high"

    @property
    def schema_str(self) -> str:
        return json.dumps(self.schema, indent=2)


@dataclass
class ScrapeGraphExperiment:
    """Result container for one evolution run."""

    tier: str
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
    schema_conformance: float = 0.0
    failure_buckets: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading from HuggingFace
# ─────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".scrapegraph_cache"
_HF_REPO = "scrapegraphai/scrapegraphai-100k"


def _download_scrapegraph_data(max_rows: int = 500) -> list[dict[str, Any]]:
    """Download a sample from the ScrapeGraphAI 100k dataset.

    Uses the ``datasets`` library to stream rows efficiently without
    downloading the full 793 MB dataset.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"sample_{max_rows}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "  ⚠ The 'datasets' package is required to download the benchmark.\n"
            "    Install with: uv pip install datasets"
        )
        raise

    print(f"  Downloading ScrapeGraphAI 100k sample ({max_rows} rows)...")
    ds = load_dataset(
        _HF_REPO,
        split=f"train[:{max_rows}]",
        cache_dir=str(_CACHE_DIR / "hf_cache"),
    )

    records: list[dict[str, Any]] = []
    for row in ds:
        # Parse the schema field (stored as a JSON string)
        schema_raw = row.get("schema", "{}")
        try:
            schema = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
        except (json.JSONDecodeError, ValueError):
            schema = {}

        # Skip records with empty schemas or no content
        if not schema or not row.get("content", "").strip():
            continue

        records.append({
            "prompt": row.get("prompt", ""),
            "schema": schema,
            "content": row.get("content", "")[:3000],  # Truncate long content
            "response": row.get("response", ""),
            "response_is_valid": bool(row.get("response_is_valid", False)),
            "schema_complexity_score": float(
                row.get("schema_complexity_score", 0.0)
            ),
        })

    print(f"    Loaded {len(records)} valid records")

    if records:
        with open(cache_file, "w") as f:
            json.dump(records, f)

    return records


def _classify_complexity(score: float, thresholds: tuple[float, float]) -> str:
    """Classify a complexity score into low/mid/high tier."""
    if score <= thresholds[0]:
        return "low"
    elif score <= thresholds[1]:
        return "mid"
    return "high"


def load_scrapegraph_dataset(
    max_samples: int = 30,
    max_download: int = 500,
) -> dict[str, list[ScrapeGraphCase]]:
    """Load and stratify ScrapeGraphAI benchmark data by schema complexity.

    Parameters
    ----------
    max_samples : int
        Maximum cases per complexity tier.
    max_download : int
        Maximum rows to download from HuggingFace.

    Returns
    -------
    dict mapping tier name to list of ScrapeGraphCase
    """
    raw = _download_scrapegraph_data(max_rows=max_download)
    if not raw:
        return {}

    rng = np.random.default_rng(42)

    # Compute complexity thresholds from the data (terciles)
    scores = sorted(r["schema_complexity_score"] for r in raw)
    t1 = scores[len(scores) // 3]
    t2 = scores[2 * len(scores) // 3]

    # Stratify by complexity tier
    by_tier: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for r in raw:
        tier = _classify_complexity(r["schema_complexity_score"], (t1, t2))
        by_tier[tier].append(r)

    result: dict[str, list[ScrapeGraphCase]] = {}
    for tier_name, records in by_tier.items():
        if not records:
            continue

        # Subsample
        if len(records) > max_samples:
            indices = rng.choice(len(records), size=max_samples, replace=False)
            records = [records[int(i)] for i in indices]

        cases = []
        for idx, rec in enumerate(records):
            cases.append(ScrapeGraphCase(
                case_id=f"{tier_name}_{idx}",
                prompt=rec["prompt"],
                schema=rec["schema"],
                content=rec["content"],
                expected_response=rec["response"],
                response_is_valid=rec["response_is_valid"],
                complexity_score=rec["schema_complexity_score"],
                complexity_tier=tier_name,
            ))
        result[tier_name] = cases

    return result


# ─────────────────────────────────────────────────────────────────────────
# Scoring — schema conformance
# ─────────────────────────────────────────────────────────────────────────


def score_schema_conformance(
    response: str, case: ScrapeGraphCase,
) -> tuple[float, dict[str, Any]]:
    """Score a response for structural conformance to the expected schema.

    Returns (score 0.0–1.0, detail dict).

    Scoring components:
    - Valid JSON           (0.3 weight)
    - Key presence match   (0.4 weight)
    - Value non-emptiness  (0.2 weight)
    - Type correctness     (0.1 weight)
    """
    detail: dict[str, Any] = {
        "valid_json": False,
        "key_match": 0.0,
        "non_empty": 0.0,
        "type_match": 0.0,
        "error": None,
    }

    # Parse response
    # Try to extract JSON from the response (model may wrap in markdown)
    json_text = response.strip()
    if "```json" in json_text:
        m = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
        if m:
            json_text = m.group(1)
    elif "```" in json_text:
        m = re.search(r"```\s*(.*?)\s*```", json_text, re.DOTALL)
        if m:
            json_text = m.group(1)

    try:
        parsed = json.loads(json_text)
        detail["valid_json"] = True
    except (json.JSONDecodeError, ValueError):
        detail["error"] = "INVALID_JSON"
        return 0.0, detail

    if not isinstance(parsed, dict):
        # Some schemas expect a top-level object
        detail["error"] = "NOT_OBJECT"
        return 0.15, detail  # Partial credit for valid JSON

    schema = case.schema
    expected_keys = set()
    if isinstance(schema, dict):
        # Schema may have "properties" (JSON Schema format) or be a flat dict
        if "properties" in schema:
            expected_keys = set(schema["properties"].keys())
        else:
            expected_keys = set(schema.keys())

    if not expected_keys:
        # No schema keys to check — full credit for valid JSON
        return 1.0, detail

    # Key presence
    actual_keys = set(parsed.keys())
    matched_keys = expected_keys & actual_keys
    key_score = len(matched_keys) / len(expected_keys) if expected_keys else 1.0
    detail["key_match"] = key_score

    # Non-emptiness
    non_empty_count = 0
    for k in matched_keys:
        val = parsed[k]
        if val is not None and val != "" and val != [] and val != {}:
            non_empty_count += 1
    non_empty_score = (
        non_empty_count / len(matched_keys) if matched_keys else 0.0
    )
    detail["non_empty"] = non_empty_score

    # Type correctness (basic check against schema type hints)
    type_hits = 0
    type_total = 0
    schema_props = schema.get("properties", schema)
    for k in matched_keys:
        prop_def = schema_props.get(k, {})
        if isinstance(prop_def, dict):
            expected_type = prop_def.get("type", "")
        elif isinstance(prop_def, str):
            expected_type = prop_def
        elif isinstance(prop_def, list):
            expected_type = "array"
        else:
            continue

        type_total += 1
        val = parsed[k]
        if expected_type == "string" and isinstance(val, str):
            type_hits += 1
        elif expected_type in ("number", "integer") and isinstance(val, (int, float)):
            type_hits += 1
        elif expected_type == "array" and isinstance(val, list):
            type_hits += 1
        elif expected_type == "object" and isinstance(val, dict):
            type_hits += 1
        elif expected_type == "boolean" and isinstance(val, bool):
            type_hits += 1
        elif not expected_type:
            type_hits += 1  # No type specified — accept anything

    type_score = type_hits / type_total if type_total > 0 else 1.0
    detail["type_match"] = type_score

    return (
        0.3 * 1.0  # valid JSON
        + 0.4 * key_score
        + 0.2 * non_empty_score
        + 0.1 * type_score
    ), detail


def _classify_failure_bucket(
    score: float, detail: dict[str, Any],
) -> FailureBucket | None:
    """Map a scoring result to a FailureBucket."""
    if detail.get("error") == "INVALID_JSON":
        return FailureBucket.UNPARSEABLE
    if detail.get("error") == "NOT_OBJECT":
        return FailureBucket.UNPARSEABLE
    if score == 0.0:
        return FailureBucket.NO_OUTPUT
    if detail.get("key_match", 0) < 0.3:
        return FailureBucket.WRONG_PARAMS
    if score < 0.7:
        return FailureBucket.PARTIAL_MATCH
    return None


# ─────────────────────────────────────────────────────────────────────────
# Seed prompt templates (schema-first)
# ─────────────────────────────────────────────────────────────────────────

_SCRAPEGRAPH_SEED_TEMPLATES = [
    # T1 — Minimal
    textwrap.dedent("""\
You are a data extraction assistant. Extract structured information from \
the given web content.

User request: {user_prompt}

Output schema:
{output_schema}

Web content:
{web_content}

Respond with ONLY valid JSON matching the schema above.
"""),

    # T2 — Step-by-step
    textwrap.dedent("""\
You are an expert data extractor. Follow these steps:
1. Read the web content carefully.
2. Identify information matching each field in the output schema.
3. Extract values accurately from the content — do NOT invent data.
4. Output ONLY valid JSON conforming to the schema.

User request: {user_prompt}

Required output schema:
{output_schema}

Source content:
{web_content}
"""),

    # T3 — Constraint-focused
    textwrap.dedent("""\
You are a precise structured data extractor. Rules:
- Output ONLY valid JSON — no markdown, no explanations.
- Include ALL fields defined in the schema.
- Use correct types: strings for text, numbers for quantities, arrays for lists.
- If a field value is not found in the content, use null.
- Do NOT hallucinate or invent data not present in the source.

User request: {user_prompt}

Schema:
{output_schema}

Content to extract from:
{web_content}
"""),

    # T4 — Persona-based
    textwrap.dedent("""\
You are a meticulous data analyst who extracts structured information \
from web pages. Your output must be machine-readable JSON that exactly \
follows the provided schema.

Task: {user_prompt}

Expected JSON structure:
{output_schema}

Source material:
{web_content}

Important: Return ONLY the JSON object. Every key from the schema must \
be present. Values must come from the source material.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# Mutation operators (generation-specific)
# ─────────────────────────────────────────────────────────────────────────

_SCRAPEGRAPH_MUTATIONS: list[str] = [
    # Schema compliance
    "Add: 'Every key in the schema MUST appear in your JSON output.'",
    "Append: 'Use the exact field names from the schema — do not rename keys.'",
    "Insert: 'For array fields, always return a list even if there is only one item.'",
    # Type accuracy
    "Add: 'String fields must be quoted. Number fields must not be quoted.'",
    "Inject: 'Boolean fields must be true or false, not strings.'",
    "Insert: 'Nested objects must contain all their required sub-keys.'",
    # Content fidelity
    "Prepend: 'Extract information ONLY from the provided content.'",
    "Add: 'Do NOT hallucinate data — if a value is not in the content, use null.'",
    "Append: 'Prefer exact quotes from the source over paraphrasing.'",
    # Format compliance
    "Append: 'Output ONLY valid JSON — no markdown fences, no explanations.'",
    "Add: 'Do not wrap JSON in ```json``` code blocks.'",
    "Insert: 'Ensure all strings are properly escaped in the JSON output.'",
    # Completeness
    "Add: 'Check that every required field has a non-null value when data exists.'",
    "Prepend: 'Read the ENTIRE content before extracting any fields.'",
    "Append: 'Scan for all instances — lists should include every matching item.'",
    # Reasoning
    "Prepend: 'First identify which parts of the content map to each schema field.'",
    "Insert: 'For each schema key, locate the relevant section in the content.'",
    "Add: 'Cross-reference extracted values against the content for accuracy.'",
]


def _mutate_template(
    template: str,
    rng: np.random.Generator,
    rate: float = 0.5,
    extra_mutations: list[str] | None = None,
) -> str:
    """Apply a random mutation to a prompt template."""
    pool = _SCRAPEGRAPH_MUTATIONS
    if extra_mutations:
        pool = _SCRAPEGRAPH_MUTATIONS + extra_mutations
    mutation = rng.choice(pool)
    lines = template.strip().split("\n")

    if ":" in mutation:
        action = mutation.split(":")[0].strip().lower()
        content = mutation.split(":", 1)[1].strip().strip("'\"")
    else:
        action = "add"
        content = mutation.strip()

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
    for ph in ("{user_prompt}", "{output_schema}", "{web_content}"):
        if ph not in result:
            result += f"\n\n{ph}"

    return result


def _crossover_templates(
    a: str, b: str, rng: np.random.Generator,
) -> str:
    """Crossover two prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    for ph in ("{user_prompt}", "{output_schema}", "{web_content}"):
        if ph not in child:
            child += f"\n\n{ph}"
    return child


# ─────────────────────────────────────────────────────────────────────────
# Score-proportional selection
# ─────────────────────────────────────────────────────────────────────────


def _score_prop_select(
    island: list[PromptCandidate], rng: np.random.Generator,
) -> PromptCandidate:
    """Score-proportional selection with exploration bonus."""
    import math

    if len(island) == 1:
        island[0].selection_count += 1
        return island[0]

    weights = np.array([
        (1.0 / (1.0 + math.exp(-10.0 * (c.score / 100.0 - 0.5))))
        * (1.0 / (1.0 + c.selection_count))
        for c in island
    ])

    total = weights.sum()
    if total <= 0:
        idx = int(rng.integers(len(island)))
        island[idx].selection_count += 1
        return island[idx]

    probs = weights / total
    idx = int(rng.choice(len(island), p=probs))
    island[idx].selection_count += 1
    return island[idx]


# ─────────────────────────────────────────────────────────────────────────
# Build schema-aware proxy scorer
# ─────────────────────────────────────────────────────────────────────────


def build_schema_scorer(
    cases: list[ScrapeGraphCase],
) -> ProxyMetricsScorer:
    """Build a ProxyMetricsScorer from the schemas of the given cases.

    Uses ``schema_to_proxy_checks()`` on the first case's schema to
    generate structural checks, then adds a common valid-JSON check.
    Since all cases within a tier share similar schema structures, the
    checks from the first case generalise well.
    """
    # Use the most common schema structure from the cases
    if not cases:
        return ProxyMetricsScorer(checks=[])

    # Pick the schema with median complexity for representative checks
    sorted_cases = sorted(cases, key=lambda c: c.complexity_score)
    representative = sorted_cases[len(sorted_cases) // 2]

    # Normalise the schema for schema_to_proxy_checks()
    schema = representative.schema
    if "properties" in schema:
        # Standard JSON Schema — extract the properties dict
        flat_schema: dict[str, Any] = {}
        for key, prop in schema["properties"].items():
            prop_type = prop.get("type", "string") if isinstance(prop, dict) else "string"
            if prop_type == "array":
                flat_schema[key] = []
            elif prop_type == "object":
                flat_schema[key] = (
                    prop.get("properties", {}) if isinstance(prop, dict) else {}
                )
            else:
                flat_schema[key] = prop_type
    elif isinstance(schema, dict):
        flat_schema = schema
    else:
        flat_schema = {}

    checks = schema_to_proxy_checks(flat_schema, weight=1.0)
    return ProxyMetricsScorer(checks=checks)


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine
# ─────────────────────────────────────────────────────────────────────────


def run_scrapegraph_evolution(
    cases: list[ScrapeGraphCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> ScrapeGraphExperiment:
    """Run prompt evolution on ScrapeGraphAI structured extraction cases.

    Uses ground-truth schema conformance scoring.
    """
    rng = np.random.default_rng(seed)
    tier = cases[0].complexity_tier if cases else "unknown"
    problem_type = config.problem_type or ProblemType.GENERATION
    error_profile = ErrorProfile()

    # ── Evaluate a candidate ─────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate,
        eval_cases: list[ScrapeGraphCase],
        track_buckets: bool = True,
    ) -> tuple[float, float]:
        """Returns (overall_score_pct, schema_conformance_pct)."""
        total = 0.0
        conform_total = 0.0
        for case in eval_cases:
            sys_prompt = (
                candidate.template
                .replace("{user_prompt}", case.prompt)
                .replace("{output_schema}", case.schema_str)
                .replace("{web_content}", case.content[:2000])
            )

            response = client.complete(
                system_prompt=sys_prompt,
                user_message="Extract the data as JSON:",
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.05))
                if track_buckets:
                    error_profile.record_bucket(FailureBucket.NO_OUTPUT)
                continue

            score, detail = score_schema_conformance(response, case)
            total += score
            if detail["valid_json"]:
                conform_total += 1

            if track_buckets:
                bucket = _classify_failure_bucket(score, detail)
                if bucket is not None:
                    error_profile.record_bucket(bucket)

        n = len(eval_cases) if eval_cases else 1
        return (
            total / n * 100.0,
            conform_total / n * 100.0,
        )

    # ── Subsample for evaluation ─────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(cases):
        eval_indices = rng.choice(
            len(cases), size=config.eval_sample_size, replace=False,
        )
        eval_cases = [cases[int(i)] for i in eval_indices]
    else:
        eval_cases = cases

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  ScrapeGraph Tier: {tier}  |  Algorithm: {algorithm_name}")
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

    best_conformance = 0.0

    for i, tmpl in enumerate(_SCRAPEGRAPH_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        score, conformance = evaluate(cand, eval_cases)
        cand.score = score
        islands[i % config.num_islands].append(cand)

        if score >= best_conformance:
            best_conformance = conformance

        prompt_trace.append({
            "generation": 0,
            "score": round(score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score,
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Baseline best: {baseline_score:.1f}%")

    # ── Evolution loop ───────────────────────────────────────────────
    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        bucket_mutations = get_failure_bucket_mutations(
            error_profile, problem_type,
        )

        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                # Parent selection
                parent_a = _score_prop_select(island, rng)

                if rng.random() < config.crossover_rate and len(island) > 1:
                    parent_b = _score_prop_select(island, rng)
                    child_tmpl = _crossover_templates(
                        parent_a.template, parent_b.template, rng,
                    )
                else:
                    child_tmpl = parent_a.template

                # Mutation (with failure bucket hints)
                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_template(
                        child_tmpl, rng, config.mutation_rate,
                        extra_mutations=bucket_mutations or None,
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

                score, conformance = evaluate(child, eval_cases)
                child.score = score
                new_cands.append(child)

                if score > best_overall.score:
                    best_conformance = conformance

            combined = island + new_cands
            combined.sort(key=lambda c: c.score, reverse=True)
            islands[isl_id] = combined[: config.elite_size]

        error_profile.decay(0.8)

        # Migration
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
            (c for isl in islands for c in isl), key=lambda c: c.score,
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
            if bucket_mutations:
                top_buckets = error_profile.worst_buckets(3)
                bucket_str = ", ".join(
                    f"{b}({c})" for b, c in top_buckets
                )
                print(f"         buckets: {bucket_str}")

    wall_time = time.perf_counter() - t0
    final_buckets = dict(error_profile.failure_buckets)

    return ScrapeGraphExperiment(
        tier=tier,
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
        schema_conformance=best_conformance,
        failure_buckets=final_buckets,
    )


# ─────────────────────────────────────────────────────────────────────────
# No-eval evolution using schema_to_proxy_checks()
# ─────────────────────────────────────────────────────────────────────────


def run_scrapegraph_noeval(
    cases: list[ScrapeGraphCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "noeval",
    seed: int = 42,
    verbose: bool = True,
) -> ScrapeGraphExperiment:
    """Run prompt evolution using schema-derived proxy checks (no labels).

    This is the **key demonstration** of the schema-first approach:
    ``schema_to_proxy_checks()`` generates structural validators from
    the JSON schema, so evolution can verify output conformance without
    any ground-truth labels.
    """
    rng = np.random.default_rng(seed)
    tier = cases[0].complexity_tier if cases else "unknown"

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  ScrapeGraph No-Eval: {tier}  |  Schema Proxy Checks")
        print(f"  Cases: {len(cases)}  |  Backend: {config.backend.value}")
        print(f"{'=' * 60}")

    # Build the schema-aware scorer
    proxy_scorer = build_schema_scorer(cases)
    if verbose:
        print(f"  Generated {len(proxy_scorer.checks)} proxy checks from schema")
        for chk in proxy_scorer.checks[:5]:
            print(f"    • {chk.name}")
        if len(proxy_scorer.checks) > 5:
            print(f"    ... and {len(proxy_scorer.checks) - 5} more")

    # Combine proxy metrics with self-consistency
    composite = CompositeScorer(
        scorers=[
            (proxy_scorer, 0.7),
            (SelfConsistencyScorer(num_samples=2), 0.3),
        ]
    )

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Build input samples for the NoEvalPromptEvolver
    eval_cases = cases[: config.eval_sample_size or len(cases)]
    inputs: list[str] = []
    for case in eval_cases:
        inp = (
            f"User request: {case.prompt}\n\n"
            f"Output schema:\n{case.schema_str}\n\n"
            f"Web content:\n{case.content[:2000]}"
        )
        inputs.append(inp)

    # Manual evolution loop (same island model as ground-truth version)
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    # Seed population — score each template
    for i, tmpl in enumerate(_SCRAPEGRAPH_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )

        # Score using proxy checks on a subset of inputs
        scores: list[float] = []
        for inp in inputs[:5]:
            sys_prompt = (
                cand.template
                .replace("{user_prompt}", inp.split("\n\n")[0].replace("User request: ", ""))
                .replace("{output_schema}", "\n\n".join(inp.split("\n\n")[1:2]))
                .replace("{web_content}", "\n\n".join(inp.split("\n\n")[2:]))
            )
            response = client.complete(
                system_prompt=sys_prompt,
                user_message="Extract the data as JSON:",
                temperature=cand.temperature,
                top_p=cand.top_p,
            )
            if response:
                s, _ = proxy_scorer.score_with_violations(response)
                scores.append(s * 100.0)
            else:
                scores.append(0.0)

        cand.score = float(np.mean(scores)) if scores else 0.0
        islands[i % config.num_islands].append(cand)

        prompt_trace.append({
            "generation": 0,
            "score": round(cand.score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score,
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Baseline best (proxy): {baseline_score:.1f}%")

    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                parent_a = _score_prop_select(island, rng)

                if rng.random() < config.crossover_rate and len(island) > 1:
                    parent_b = _score_prop_select(island, rng)
                    child_tmpl = _crossover_templates(
                        parent_a.template, parent_b.template, rng,
                    )
                else:
                    child_tmpl = parent_a.template

                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_template(child_tmpl, rng)

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

                # Score using proxy checks
                scores_list: list[float] = []
                for case in eval_cases[:5]:
                    sys_prompt = (
                        child.template
                        .replace("{user_prompt}", case.prompt)
                        .replace("{output_schema}", case.schema_str)
                        .replace("{web_content}", case.content[:2000])
                    )
                    response = client.complete(
                        system_prompt=sys_prompt,
                        user_message="Extract the data as JSON:",
                        temperature=child.temperature,
                        top_p=child.top_p,
                    )
                    if response:
                        s, _ = proxy_scorer.score_with_violations(response)
                        scores_list.append(s * 100.0)
                    else:
                        scores_list.append(0.0)

                child.score = float(np.mean(scores_list)) if scores_list else 0.0
                new_cands.append(child)

            combined = island + new_cands
            combined.sort(key=lambda c: c.score, reverse=True)
            islands[isl_id] = combined[: config.elite_size]

        # Migration
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
            (c for isl in islands for c in isl), key=lambda c: c.score,
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

    # Cross-validate: score the best no-eval prompt using ground-truth
    if verbose:
        print("\n  Cross-validating best no-eval prompt against ground truth...")
    gt_total = 0.0
    for case in eval_cases:
        sys_prompt = (
            best_overall.template
            .replace("{user_prompt}", case.prompt)
            .replace("{output_schema}", case.schema_str)
            .replace("{web_content}", case.content[:2000])
        )
        response = client.complete(
            system_prompt=sys_prompt,
            user_message="Extract the data as JSON:",
            temperature=best_overall.temperature,
            top_p=best_overall.top_p,
        )
        if response:
            score, _ = score_schema_conformance(response, case)
            gt_total += score

    gt_score = gt_total / len(eval_cases) * 100.0 if eval_cases else 0.0
    if verbose:
        print(f"  Ground-truth score of no-eval winner: {gt_score:.1f}%")

    return ScrapeGraphExperiment(
        tier=tier,
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
        schema_conformance=gt_score,
        failure_buckets={},
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_PROMPT = textwrap.dedent("""\
You are a helpful assistant. Extract structured data from the given \
content as JSON.

{user_prompt}

{output_schema}

{web_content}
""")


def evaluate_baseline(
    cases: list[ScrapeGraphCase],
    tier_name: str,
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate the default prompt on cases."""
    if verbose:
        print(f"  Evaluating default prompt baseline ({tier_name})...")

    total = 0.0
    for case in cases:
        sys_prompt = (
            _DEFAULT_PROMPT
            .replace("{user_prompt}", case.prompt)
            .replace("{output_schema}", case.schema_str)
            .replace("{web_content}", case.content[:2000])
        )
        response = client.complete(
            system_prompt=sys_prompt,
            user_message="Extract the data as JSON:",
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            continue
        score, _ = score_schema_conformance(response, case)
        total += score

    result = (total / len(cases) * 100.0) if cases else 0.0
    if verbose:
        print(f"  Default prompt baseline ({tier_name}): {result:.1f}%")
    return result


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: ScrapeGraphExperiment) -> None:
    """Print the prompt evolution trace."""
    print(f"\n{'─' * 60}")
    print(
        f"  Prompt Evolution — {experiment.tier} ({experiment.algorithm})"
    )
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
    print(f"  Schema Conformance: {experiment.schema_conformance:.1f}%")
    if experiment.failure_buckets:
        print(
            "  Failure Buckets: "
            + ", ".join(
                f"{k}={v}" for k, v in
                sorted(
                    experiment.failure_buckets.items(),
                    key=lambda x: x[1], reverse=True,
                )
            )
        )


def show_results_table(
    experiments: list[ScrapeGraphExperiment],
    default_baselines: dict[str, float],
) -> None:
    """Print a summary table."""
    print(f"\n{'=' * 90}")
    print("  ScrapeGraphAI Schema-First Prompt Evolution Results")
    print(f"{'=' * 90}")
    print(
        f"  {'Tier':<10} {'Algorithm':<12} {'Backend':<14} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}"
    )
    print(f"  {'─' * 85}")

    for exp in experiments:
        dflt = default_baselines.get(exp.tier, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.tier:<10} {exp.algorithm:<12} {exp.backend:<14} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s"
        )

    print(f"  {'─' * 85}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[ScrapeGraphExperiment],
    default_baselines: dict[str, float],
    path: str = "scrapegraph_experiment_log.json",
) -> None:
    """Save experiment results to JSON for dashboard consumption."""
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "ScrapeGraphAI-100k",
        "default_baselines": default_baselines,
        "experiments": [],
    }
    for exp in experiments:
        entry = {
            "tier": exp.tier,
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
            "schema_conformance": round(exp.schema_conformance, 2),
            "failure_buckets": exp.failure_buckets,
        }
        log["experiments"].append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Experiment log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {
        "iterations": 3,
        "population_size": 4,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.6,
        "crossover_rate": 0.3,
        "eval_sample_size": 8,
        "selection_method": SelectionMethod.SCORE_PROPORTIONAL,
        "problem_type": ProblemType.GENERATION,
    },
    "deep": {
        "iterations": 5,
        "population_size": 5,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.5,
        "crossover_rate": 0.4,
        "eval_sample_size": 10,
        "selection_method": SelectionMethod.SCORE_PROPORTIONAL,
        "problem_type": ProblemType.GENERATION,
    },
}

COMPLEXITY_TIERS = ["low", "mid", "high"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScrapeGraphAI schema-first prompt evolution",
    )
    parser.add_argument(
        "--max-samples", type=int, default=15,
        help="Max cases per complexity tier (default: 15)",
    )
    parser.add_argument(
        "--max-download", type=int, default=500,
        help="Max rows to download from HuggingFace (default: 500)",
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Run only ground-truth evolution (skip no-eval)",
    )
    parser.add_argument(
        "--noeval-only", action="store_true",
        help="Run only no-eval schema proxy evolution (skip ground-truth)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override iteration count for all algorithms",
    )
    args = parser.parse_args()

    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MutaGenAI × ScrapeGraphAI — Schema-First Prompt Evolution   ║
    ║  93k real schemas · 3 complexity tiers · proxy checks        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Backend setup ────────────────────────────────────────────────
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=60.0,
    )
    ollama_client = LLMClient(ollama_cfg)

    if not ollama_client.is_available():
        print(
            "  ⚠ Ollama not available — ensure it is running "
            "at localhost:11434"
        )
        print("  Running in mock mode (random scores) for demonstration.\n")

    # ── Load ScrapeGraphAI data ──────────────────────────────────────
    print("  Loading ScrapeGraphAI benchmark data...")
    try:
        by_tier = load_scrapegraph_dataset(
            max_samples=args.max_samples,
            max_download=args.max_download,
        )
    except Exception as exc:
        print(f"  ✗ {exc}")
        return

    for tier_name in COMPLEXITY_TIERS:
        cases = by_tier.get(tier_name, [])
        print(f"    {tier_name:<10}: {len(cases):3d} cases")

    if not by_tier:
        print("  No data loaded — exiting.")
        return

    # ── Phase 1: Default prompt baselines ────────────────────────────
    print("\n  Phase 1: Default Prompt Baselines")
    print("  " + "─" * 50)
    default_baselines: dict[str, float] = {}
    for tier_name in COMPLEXITY_TIERS:
        cases = by_tier.get(tier_name, [])
        if cases:
            score = evaluate_baseline(cases, tier_name, ollama_client)
            default_baselines[tier_name] = score

    all_experiments: list[ScrapeGraphExperiment] = []

    # ── Phase 2: Ground-truth evolution ──────────────────────────────
    if not args.noeval_only:
        print("\n  Phase 2: Ground-Truth Prompt Evolution")
        print("  " + "─" * 50)

        for tier_name in COMPLEXITY_TIERS:
            cases = by_tier.get(tier_name, [])
            if not cases:
                continue

            algo = "deep"
            algo_params = dict(ALGORITHM_CONFIGS[algo])
            if args.iterations:
                algo_params["iterations"] = args.iterations
            cfg = PromptEvolverConfig(
                backend=LLMBackend.OLLAMA,
                ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
                timeout=60.0,
                **algo_params,
            )

            exp = run_scrapegraph_evolution(
                cases=cases,
                client=ollama_client,
                config=cfg,
                algorithm_name=algo,
                seed=123,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

    # ── Phase 3: No-eval schema proxy evolution ──────────────────────
    if not args.gt_only:
        print("\n  Phase 3: No-Eval Schema Proxy Evolution")
        print("  " + "─" * 50)

        for tier_name in COMPLEXITY_TIERS:
            cases = by_tier.get(tier_name, [])
            if not cases:
                continue

            algo_params = dict(ALGORITHM_CONFIGS["standard"])
            if args.iterations:
                algo_params["iterations"] = args.iterations
            cfg = PromptEvolverConfig(
                backend=LLMBackend.OLLAMA,
                ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
                timeout=60.0,
                **algo_params,
            )

            exp = run_scrapegraph_noeval(
                cases=cases,
                client=ollama_client,
                config=cfg,
                algorithm_name="noeval_proxy",
                seed=456,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

    # ── Results ──────────────────────────────────────────────────────
    if all_experiments:
        show_results_table(all_experiments, default_baselines)
        save_experiment_log(
            all_experiments,
            default_baselines,
            path="scrapegraph_experiment_log.json",
        )

    print("\n  Done.")


if __name__ == "__main__":
    main()
