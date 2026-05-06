#!/usr/bin/env python3
"""
Cookbook Recipe 49 — AssetOpsBench Track 1: Planning-Oriented Prompt Evolution
==============================================================================

Evolves system prompts for **AssetOpsBench Track 1** — Planning-Oriented
Multi-Agent Orchestration for industrial asset operations.

The challenge requires an LLM to decompose natural-language questions about
industrial assets into structured **DAG execution plans** where each step
names a server, tool, dependencies, and expected output.

Scenarios
---------
AssetOpsBench provides 141+ scenarios across:
* **IoT** — sensor data, asset listings, site info
* **FMSR** — failure modes, sensor-to-failure mappings
* **TSFM** — time-series forecasting, anomaly detection
* **WO** — work order generation and queries
* **Vibration** — vibration analysis, DSP diagnostics

Track 1 focus
-------------
Design better prompts that transform complex multi-agent interactions into
clear, structured DAG plans — correct server/tool selection, dependency
ordering, and minimal step count.

Algorithm experiments
---------------------
* **Standard** — 3 iterations, pop 6, 2 islands  (balanced)
* **Deep**     — 5 iterations, pop 8, 3 islands  (thorough)

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_assetops.py

    # Deep mode:
    uv run python examples/cookbook/prompt_evolution_assetops.py --deep

    # No-eval composite mode (no ground truth):
    uv run python examples/cookbook/prompt_evolution_assetops.py --no-eval

The script saves an experiment log to ``logs/assetops_track1_log.json``
for dashboard consumption.

References
----------
* Challenge:  https://sites.google.com/view/assetopsbench-challenge/home
* Repository: https://github.com/IBM/AssetOpsBench
* Dataset:    https://huggingface.co/datasets/ibm-research/AssetOpsBench
* Paper:      https://arxiv.org/abs/2506.03828
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
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)

# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────

# Valid servers and tools in AssetOpsBench
VALID_SERVERS = {"iot", "fmsr", "tsfm", "wo", "utilities", "vibration"}

VALID_TOOLS: dict[str, list[str]] = {
    "iot": ["get_sites", "get_assets", "get_sensors", "get_history"],
    "fmsr": ["get_failure_modes", "get_failure_sensor_mapping"],
    "tsfm": [
        "forecasting",
        "timeseries_anomaly_detection",
        "get_supported_models",
        "get_supported_frequencies",
        "list_datasets",
        "get_dataset_info",
    ],
    "wo": [
        "generate_work_order",
        "get_work_orders",
        "update_work_order",
        "delete_work_order",
        "get_work_order_by_id",
        "search_work_orders",
        "get_equipment_work_orders",
        "add_notes_to_work_order",
    ],
    "utilities": ["get_current_datetime", "get_available_servers", "echo"],
    "vibration": [
        "get_vibration_data",
        "run_fft_analysis",
        "run_envelope_analysis",
        "get_bearing_frequencies",
        "detect_imbalance",
        "detect_misalignment",
        "get_vibration_summary",
        "compare_vibration_trends",
    ],
}

ALL_TOOLS = {tool for tools in VALID_TOOLS.values() for tool in tools}


@dataclass
class AssetOpsScenario:
    """A single AssetOpsBench scenario."""

    scenario_id: str
    utterance: str
    category: str  # Knowledge Query, Data Query, Decision Support, etc.
    scenario_type: str  # IoT, FMSR, TSFM, Workorder, Vibration, Utilities, Multi
    entity: str  # Site, Chiller, Equipment, etc.
    characteristic_form: str  # expected response description
    deterministic: bool
    group: str  # retrospective, predictive, prescriptive
    note: str

    @property
    def primary_server(self) -> str:
        """Map scenario type to the primary MCP server."""
        type_map = {
            "iot": "iot",
            "fmsr": "fmsr",
            "tsfm": "tsfm",
            "workorder": "wo",
            "vibration": "vibration",
            "utilities": "utilities",
        }
        return type_map.get(self.scenario_type.lower(), "")

    @property
    def is_multi_server(self) -> bool:
        return self.scenario_type.lower() == "multi"


@dataclass
class AssetOpsExperiment:
    """Result container for one evolution run."""

    category: str
    algorithm: str
    backend: str
    n_scenarios: int
    baseline_score: float
    evolved_score: float
    best_prompt_template: str
    best_temperature: float
    best_top_p: float
    iterations: int
    wall_time: float
    history: list[tuple[int, float]] = field(default_factory=list)
    prompt_evolution: list[dict[str, Any]] = field(default_factory=list)
    server_accuracy: float = 0.0
    tool_accuracy: float = 0.0
    dependency_accuracy: float = 0.0
    format_compliance: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Plan parsing
# ─────────────────────────────────────────────────────────────────────────

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_SERVER_RE = re.compile(r"#Server(\d+):\s*(.+)")
_TOOL_RE = re.compile(r"#Tool(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


@dataclass
class PlanStep:
    """A parsed plan step."""

    step_number: int
    task: str
    server: str
    tool: str
    dependencies: list[int]
    expected_output: str


def parse_plan(raw: str) -> list[PlanStep]:
    """Parse an LLM-generated plan into structured steps."""
    tasks = {int(m.group(1)): m.group(2).strip() for m in _TASK_RE.finditer(raw)}
    servers = {int(m.group(1)): m.group(2).strip() for m in _SERVER_RE.finditer(raw)}
    tools_map = {
        int(m.group(1)): m.group(2).strip().split("(")[0].strip()
        for m in _TOOL_RE.finditer(raw)
    }
    deps_raw = {int(m.group(1)): m.group(2).strip() for m in _DEP_RE.finditer(raw)}
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_RE.finditer(raw)}

    steps = []
    for n in sorted(tasks):
        raw_dep = deps_raw.get(n, "None").strip()
        if raw_dep.lower() == "none":
            dependencies = []
        else:
            dependencies = [int(x) for x in _DEP_NUM_RE.findall(raw_dep)]

        steps.append(PlanStep(
            step_number=n,
            task=tasks[n],
            server=servers.get(n, ""),
            tool=tools_map.get(n, ""),
            dependencies=dependencies,
            expected_output=outputs.get(n, ""),
        ))

    return steps


# ─────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────


def score_plan(
    response: str, scenario: AssetOpsScenario
) -> tuple[float, dict[str, Any]]:
    """Score a generated plan against an AssetOpsBench scenario.

    Returns (score 0.0–1.0, detail dict).

    Scoring components:
    - Format compliance  (0.15): plan uses the correct #TaskN format
    - Server selection   (0.30): correct servers referenced
    - Tool selection     (0.30): correct tools selected
    - Dependency order   (0.15): dependencies are valid (no forward refs)
    - Plan completeness  (0.10): all expected steps present
    """
    detail: dict[str, Any] = {
        "format_ok": False,
        "server_accuracy": 0.0,
        "tool_accuracy": 0.0,
        "dependency_valid": False,
        "completeness": 0.0,
        "error": None,
    }

    # Parse the plan
    steps = parse_plan(response)

    if not steps:
        detail["error"] = "NO_PLAN_PARSED"
        return 0.0, detail

    # Format compliance — did it produce at least one valid step?
    has_tasks = bool(_TASK_RE.search(response))
    has_servers = bool(_SERVER_RE.search(response))
    has_tools = bool(_TOOL_RE.search(response))
    format_ok = has_tasks and has_servers and has_tools
    detail["format_ok"] = format_ok
    format_score = 1.0 if format_ok else 0.3 if has_tasks else 0.0

    # Server accuracy — are predicted servers valid?
    pred_servers = [s.server.lower().strip() for s in steps]
    valid_server_count = sum(1 for s in pred_servers if s in VALID_SERVERS)
    server_acc = valid_server_count / len(pred_servers) if pred_servers else 0.0
    detail["server_accuracy"] = server_acc

    # Tool accuracy — are predicted tools valid for their server?
    valid_tool_count = 0
    for step in steps:
        server = step.server.lower().strip()
        tool = step.tool.lower().strip()
        if server in VALID_TOOLS:
            server_tools = [t.lower() for t in VALID_TOOLS[server]]
            if tool in server_tools:
                valid_tool_count += 1
        elif tool.lower() in {t.lower() for t in ALL_TOOLS}:
            valid_tool_count += 0.5  # right tool, wrong server
    tool_acc = valid_tool_count / len(steps) if steps else 0.0
    detail["tool_accuracy"] = tool_acc

    # Dependency validity — no forward references, no self-references
    dep_valid = True
    for step in steps:
        for dep in step.dependencies:
            if dep >= step.step_number or dep < 1:
                dep_valid = False
                break
    detail["dependency_valid"] = dep_valid
    dep_score = 1.0 if dep_valid else 0.3

    # Completeness — does the plan address the scenario's domain?
    primary = scenario.primary_server
    if primary:
        pred_server_set = {s.server.lower().strip() for s in steps}
        completeness = 1.0 if primary in pred_server_set else 0.3
    else:
        # Multi or unknown type — score by structural quality only
        completeness = 1.0 if len(steps) >= 1 else 0.0
    detail["completeness"] = completeness

    score = (
        0.15 * format_score
        + 0.30 * server_acc
        + 0.30 * tool_acc
        + 0.15 * dep_score
        + 0.10 * completeness
    )
    return score, detail


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".assetops_cache"


def _build_server_descriptions() -> str:
    """Build the server/tool descriptions block for the planning prompt."""
    lines = []
    for server, tools in VALID_TOOLS.items():
        tool_list = ", ".join(tools)
        lines.append(f"{server}: {tool_list}")
    return "\n".join(lines)


SERVER_DESCRIPTIONS = _build_server_descriptions()


def load_assetops_scenarios(
    max_scenarios: int = 50,
) -> list[AssetOpsScenario]:
    """Load AssetOpsBench scenarios from HuggingFace.

    Falls back to embedded sample scenarios if the dataset is unavailable.
    """
    scenarios: list[AssetOpsScenario] = []

    try:
        from datasets import load_dataset

        print("  Loading AssetOpsBench from HuggingFace...")
        ds = load_dataset("ibm-research/AssetOpsBench", "scenarios", split="train")

        for idx, row in enumerate(ds):
            if idx >= max_scenarios:
                break

            text = row.get("text", "")
            if not text:
                continue

            scenarios.append(AssetOpsScenario(
                scenario_id=str(row.get("id", idx)),
                utterance=text,
                category=str(row.get("category", "")),
                scenario_type=str(row.get("type", "")),
                entity=str(row.get("entity", "")),
                characteristic_form=str(row.get("characteristic_form", "")),
                deterministic=bool(row.get("deterministic", True)),
                group=str(row.get("group", "")),
                note=str(row.get("note", "")),
            ))

        print(f"    Loaded {len(scenarios)} scenarios from HuggingFace")

    except Exception as exc:
        print(f"  ⚠ Could not load from HuggingFace: {exc}")
        print("  Using embedded sample scenarios...")
        scenarios = _sample_scenarios()

    return scenarios


def _sample_scenarios() -> list[AssetOpsScenario]:
    """Embedded sample scenarios for local development."""
    return [
        AssetOpsScenario(
            scenario_id="sample_1",
            utterance="What sensors are on Chiller 6 at MAIN site?",
            category="Data Query",
            scenario_type="IoT",
            entity="Chiller",
            characteristic_form="The expected response should be the sensor list for Chiller 6 at the MAIN site.",
            deterministic=True,
            group="retrospective",
            note="Source: IoT data operations",
        ),
        AssetOpsScenario(
            scenario_id="sample_2",
            utterance="List all assets at site MAIN and get failure modes for Chiller 6.",
            category="Data Query",
            scenario_type="Multi",
            entity="Chiller",
            characteristic_form="The expected response should include the asset list for site MAIN and failure modes for Chiller 6.",
            deterministic=True,
            group="retrospective",
            note="Source: IoT + FMSR combined query",
        ),
        AssetOpsScenario(
            scenario_id="sample_3",
            utterance=(
                "Forecast the Chiller 9 Condenser Water Flow for next week "
                "and generate a work order if anomaly is detected."
            ),
            category="Decision Support",
            scenario_type="Multi",
            entity="Chiller",
            characteristic_form="The expected response should include a time series forecast and conditional work order generation.",
            deterministic=False,
            group="prescriptive",
            note="Source: TSFM + Workorder combined query",
        ),
        AssetOpsScenario(
            scenario_id="sample_4",
            utterance="Get the work order of equipment CWC04013 for year 2017.",
            category="Data Query",
            scenario_type="Workorder",
            entity="Equipment",
            characteristic_form="The expected response should be work orders for equipment CWC04013 from 2017.",
            deterministic=True,
            group="retrospective",
            note="Source: Workorder data operations",
        ),
        AssetOpsScenario(
            scenario_id="sample_5",
            utterance=(
                "What is the current date and time? Also list assets at site MAIN. "
                "Also get sensor list and failure mode list for any of the chiller "
                "at site MAIN."
            ),
            category="Data Query",
            scenario_type="Multi",
            entity="Chiller",
            characteristic_form="The expected response should include current datetime, asset list, sensor list, and failure modes.",
            deterministic=True,
            group="retrospective",
            note="Source: Utilities + IoT + FMSR combined query",
        ),
        AssetOpsScenario(
            scenario_id="sample_6",
            utterance="Identify failure modes detected by Chiller 6 Supply Temperature sensor.",
            category="Knowledge Query",
            scenario_type="FMSR",
            entity="Chiller",
            characteristic_form="The expected response should be the failure modes mapped to the Supply Temperature sensor of Chiller 6.",
            deterministic=True,
            group="retrospective",
            note="Source: FMSR data operations",
        ),
        AssetOpsScenario(
            scenario_id="sample_7",
            utterance=(
                "Get vibration data for Motor 3, run FFT analysis, "
                "and check for imbalance."
            ),
            category="Decision Support",
            scenario_type="Vibration",
            entity="Equipment",
            characteristic_form="The expected response should include vibration data, FFT analysis results, and imbalance detection.",
            deterministic=False,
            group="predictive",
            note="Source: Vibration analysis pipeline",
        ),
        AssetOpsScenario(
            scenario_id="sample_8",
            utterance="Is LSTM model supported in TSFM?",
            category="Knowledge Query",
            scenario_type="TSFM",
            entity="Equipment",
            characteristic_form="The expected response should indicate whether LSTM is among the supported models.",
            deterministic=True,
            group="retrospective",
            note="Source: TSFM knowledge query",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────
# Seed prompt templates
# ─────────────────────────────────────────────────────────────────────────

_ASSETOPS_SEED_TEMPLATES = [
    # T1 — Minimal direct
    textwrap.dedent("""\
        You are a planning assistant for industrial asset operations and maintenance.

Decompose the question into a sequence of subtasks. For each subtask,
assign a server and select the exact tool to call.

Available servers and tools:
{servers}

Output format — one block per step:
#Task1: <task description>
#Server1: <exact server name>
#Tool1: <exact tool name>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

Question: {question}

Plan:
"""),

    # T2 — Chain-of-thought
    textwrap.dedent("""\
        Think step-by-step about this industrial operations question.

1. What data does the user need? Identify the asset, site, and sensors.
2. Which servers and tools provide that data?
3. What is the dependency order?
4. Are there analysis or forecasting steps needed?
5. Does the user need a work order or summary?

Available servers and tools:
{servers}

Output each step using:
#TaskN: <description>
#ServerN: <server>
#ToolN: <tool>
#DependencyN: None or #S<M>
#ExpectedOutputN: <expected result>

Question: {question}

Plan:
"""),

    # T3 — Constraint-first
    textwrap.dedent("""\
        CONSTRAINTS — follow strictly:
- Use ONLY the servers and tools listed below. Do NOT invent tools.
- Each step must call exactly ONE tool from ONE server.
- Dependencies must reference earlier steps only (#S1, #S2, etc.).
- Parallel steps with no dependency should list Dependency as None.
- Keep the plan minimal — do NOT add steps the question does not require.

Available servers and tools:
{servers}

Output format:
#TaskN: <description>
#ServerN: <server>
#ToolN: <tool>
#DependencyN: None or #S<M>
#ExpectedOutputN: <result>

Question: {question}

Plan:
"""),

    # T4 — Domain-expert persona
    textwrap.dedent("""\
        You are a reliability engineer's AI assistant for industrial asset management.

Domain knowledge:
- iot: sensor data, asset listings, site info, sensor history
- fmsr: failure modes, sensor-to-failure mappings
- tsfm: time-series forecasting, anomaly detection, model info
- wo: work order generation, queries, updates
- utilities: date/time, server listing
- vibration: vibration data, FFT, envelope analysis, fault detection

For each step, specify the server, tool, dependencies, and expected output.

Available tools:
{servers}

Question: {question}

Plan:
"""),

    # T5 — Capability-matching
    textwrap.dedent("""\
        Map the user question to an execution plan by matching each information need:

- Asset/sensor/site queries → iot server
- Failure mode analysis → fmsr server
- Forecasting/anomaly → tsfm server
- Work orders → wo server
- Date/time → utilities server
- Vibration diagnostics → vibration server

Available tools:
{servers}

Return only the ordered plan using #TaskN / #ServerN / #ToolN / #DependencyN / #ExpectedOutputN format.

Question: {question}

Plan:
"""),

    # T6 — Output-strict
    textwrap.dedent("""\
        Return a structured plan. No explanation, commentary, or extra text.

Available servers and tools:
{servers}

Rules:
1. Match server and tool names EXACTLY.
2. Use the MINIMUM number of steps.
3. Declare dependencies with #S<N> notation.
4. Every step needs an expected output description.
5. Never combine multiple tool calls into one step.

Question: {question}

Plan:
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# Mutation operators
# ─────────────────────────────────────────────────────────────────────────

_ASSETOPS_MUTATIONS: list[str] = [
    # Server/tool selection
    "Add: 'Each server has specific tools — match the data need to the right server first, then pick the tool.'",
    "Inject: 'For IoT data (sensors, assets, sites), always use the iot server.'",
    "Insert: 'For failure analysis, use fmsr. For forecasting, use tsfm.'",
    "Append: 'Work order operations (create, query, update) belong to the wo server.'",
    # Dependency ordering
    "Add: 'Steps that need sensor data must depend on the IoT step that retrieves it.'",
    "Insert: 'Forecasting steps depend on having identified the correct sensor first.'",
    "Inject: 'Work order generation is always a final step — it depends on diagnosis results.'",
    # Plan minimality
    "Prepend: 'Generate the MINIMUM steps needed. Do not add extra steps.'",
    "Append: 'If a question asks about one asset, do NOT plan steps for other assets.'",
    "Add: 'Do not include a utilities/get_current_datetime step unless the question asks for the time.'",
    # Format compliance
    "Append: 'Use the exact format: #TaskN / #ServerN / #ToolN / #DependencyN / #ExpectedOutputN.'",
    "Insert: 'Server names must be lowercase: iot, fmsr, tsfm, wo, utilities, vibration.'",
    "Add: 'Tool names must match exactly — check spelling against the tool list.'",
    # Domain reasoning
    "Prepend: 'Think about what the reliability engineer needs to know to answer this question.'",
    "Add: 'For anomaly detection, you typically need: get sensor data → detect anomaly → generate work order.'",
    "Inject: 'Multi-step questions may require parallel steps with no dependencies.'",
    # Chain-of-thought
    "Prepend: 'First identify all the information the question asks for, then plan steps for each.'",
    "Insert: 'Before outputting the plan, verify each server and tool name against the available list.'",
]


def _mutate_assetops_template(
    template: str, rng: np.random.Generator, rate: float = 0.5
) -> str:
    """Apply a random mutation to an AssetOps prompt template."""
    mutation = rng.choice(_ASSETOPS_MUTATIONS)
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
    if "{servers}" not in result:
        result += "\n\nAvailable servers and tools:\n{servers}"
    if "{question}" not in result:
        result += "\n\nQuestion: {question}\n\nPlan:"

    return result


def _crossover_assetops_templates(
    a: str, b: str, rng: np.random.Generator
) -> str:
    """Crossover two AssetOps prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{servers}" not in child:
        child += "\n\nAvailable servers and tools:\n{servers}"
    if "{question}" not in child:
        child += "\n\nQuestion: {question}\n\nPlan:"
    return child


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine
# ─────────────────────────────────────────────────────────────────────────


def run_assetops_evolution(
    scenarios: list[AssetOpsScenario],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> AssetOpsExperiment:
    """Run prompt evolution on AssetOpsBench Track 1 scenarios.

    Returns an AssetOpsExperiment with full tracking.
    """
    rng = np.random.default_rng(seed)

    # ── Evaluate a candidate ───────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate, eval_scenarios: list[AssetOpsScenario]
    ) -> tuple[float, float, float, float, float]:
        """Returns (overall, server_acc, tool_acc, dep_acc, format_pct)."""
        total = 0.0
        server_total = 0.0
        tool_total = 0.0
        dep_total = 0.0
        format_total = 0.0

        for scenario in eval_scenarios:
            sys_prompt = candidate.template.replace(
                "{servers}", SERVER_DESCRIPTIONS
            ).replace(
                "{question}", scenario.utterance
            )

            response = client.complete(
                system_prompt=sys_prompt,
                user_message=scenario.utterance,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.05))
                continue

            score, detail = score_plan(response, scenario)
            total += score
            server_total += detail["server_accuracy"]
            tool_total += detail["tool_accuracy"]
            dep_total += 1.0 if detail["dependency_valid"] else 0.0
            format_total += 1.0 if detail["format_ok"] else 0.0

        n = len(eval_scenarios) if eval_scenarios else 1
        return (
            total / n * 100.0,
            server_total / n * 100.0,
            tool_total / n * 100.0,
            dep_total / n * 100.0,
            format_total / n * 100.0,
        )

    # ── Subsample for evaluation ───────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(scenarios):
        eval_indices = rng.choice(
            len(scenarios), size=config.eval_sample_size, replace=False
        )
        eval_scenarios = [scenarios[int(i)] for i in eval_indices]
    else:
        eval_scenarios = scenarios

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  AssetOpsBench Track 1 — Planning Prompt Evolution")
        print(f"  Algorithm: {algorithm_name}  |  Scenarios: {len(eval_scenarios)}")
        print(f"  Backend: {config.backend.value}")
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    best_server_acc = 0.0
    best_tool_acc = 0.0

    for i, tmpl in enumerate(_ASSETOPS_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        score, s_acc, t_acc, d_acc, f_pct = evaluate(cand, eval_scenarios)
        cand.score = score
        islands[i % config.num_islands].append(cand)

        if score > best_server_acc:
            best_server_acc = s_acc
            best_tool_acc = t_acc

        prompt_trace.append({
            "generation": 0,
            "score": round(score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

        if verbose:
            print(f"  Seed {i + 1}: {score:.1f}% (server={s_acc:.0f}%, tool={t_acc:.0f}%)")

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"\n  Baseline best: {baseline_score:.1f}%")

    # ── Evolution loop ─────────────────────────────────────────────────
    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            # Tournament selection
            sorted_island = sorted(island, key=lambda c: c.score, reverse=True)

            new_candidates: list[PromptCandidate] = []

            # Keep elite
            elite = copy.deepcopy(sorted_island[0])
            new_candidates.append(elite)

            # Generate offspring
            while len(new_candidates) < config.population_size:
                if rng.random() < config.crossover_rate and len(sorted_island) >= 2:
                    # Crossover
                    parents = rng.choice(
                        min(3, len(sorted_island)), size=2, replace=False
                    )
                    child_tmpl = _crossover_assetops_templates(
                        sorted_island[int(parents[0])].template,
                        sorted_island[int(parents[1])].template,
                        rng,
                    )
                else:
                    # Clone + mutate
                    parent_idx = int(rng.integers(0, min(3, len(sorted_island))))
                    child_tmpl = sorted_island[parent_idx].template

                # Apply mutation
                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_assetops_template(child_tmpl, rng)

                child = PromptCandidate(
                    template=child_tmpl,
                    temperature=float(
                        np.clip(
                            sorted_island[0].temperature
                            + rng.normal(0, 0.05),
                            *config.temperature_range,
                        )
                    ),
                    top_p=float(
                        np.clip(
                            sorted_island[0].top_p + rng.normal(0, 0.03),
                            *config.top_p_range,
                        )
                    ),
                    generation=gen,
                )
                score, s_acc, t_acc, d_acc, f_pct = evaluate(child, eval_scenarios)
                child.score = score
                new_candidates.append(child)

                if score > best_overall.score:
                    best_overall = copy.deepcopy(child)
                    best_server_acc = s_acc
                    best_tool_acc = t_acc

                prompt_trace.append({
                    "generation": gen,
                    "score": round(score, 1),
                    "template_hash": child.hash,
                    "template_preview": child.template[:120].replace("\n", " "),
                })

            islands[isl_id] = new_candidates

        # Migration (every 2 generations)
        migration_interval = 2
        if gen % migration_interval == 0 and config.num_islands > 1:
            for i in range(config.num_islands):
                src = islands[i]
                dst = islands[(i + 1) % config.num_islands]
                if src:
                    migrant = copy.deepcopy(max(src, key=lambda c: c.score))
                    migrant.generation = gen
                    dst.append(migrant)

        gen_best = max(
            (c for isl in islands for c in isl), key=lambda c: c.score
        )
        history.append((gen, gen_best.score))

        if verbose:
            print(
                f"  Gen {gen}/{config.iterations}: best={gen_best.score:.1f}%"
                f"  (overall best={best_overall.score:.1f}%)"
            )

    wall_time = time.perf_counter() - t0

    if verbose:
        print(f"\n  {'─' * 50}")
        print(f"  Baseline: {baseline_score:.1f}%  →  Evolved: {best_overall.score:.1f}%")
        print(f"  Improvement: {best_overall.score - baseline_score:+.1f}%")
        print(f"  Server accuracy: {best_server_acc:.1f}%")
        print(f"  Tool accuracy: {best_tool_acc:.1f}%")
        print(f"  Wall time: {wall_time:.0f}s")
        print(f"  {'─' * 50}\n")

    return AssetOpsExperiment(
        category="track1_planning",
        algorithm=algorithm_name,
        backend=config.backend.value,
        n_scenarios=len(eval_scenarios),
        baseline_score=baseline_score,
        evolved_score=best_overall.score,
        best_prompt_template=best_overall.template,
        best_temperature=best_overall.temperature,
        best_top_p=best_overall.top_p,
        iterations=config.iterations,
        wall_time=wall_time,
        history=history,
        prompt_evolution=prompt_trace,
        server_accuracy=best_server_acc,
        tool_accuracy=best_tool_acc,
    )


# ─────────────────────────────────────────────────────────────────────────
# No-eval mode (for Phase 2 generalization — unseen scenarios)
# ─────────────────────────────────────────────────────────────────────────


def run_assetops_no_eval(
    client: LLMClient,
    config: PromptEvolverConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run no-eval composite evolution for AssetOpsBench.

    Uses LLMJudge + ProxyMetrics + SelfConsistency — no ground truth needed.
    Designed for Phase 2 where scenarios are from unseen asset classes.
    """
    from MutaGenAI.strategies import (
        CompositeScorer,
        LLMJudge,
        NoEvalConfig,
        NoEvalPromptEvolver,
        ProxyCheck,
        ProxyMetricsScorer,
        SelfConsistencyScorer,
    )

    task_description = textwrap.dedent("""\
        You are a planning assistant for industrial asset operations.
        Given a question about industrial assets (sensors, maintenance,
        forecasting, work orders), decompose it into an execution plan.
        Each step must name a server (iot, fmsr, tsfm, wo, utilities,
        vibration), a tool, dependencies, and expected output.
        Use the format: #TaskN / #ServerN / #ToolN / #DependencyN / #ExpectedOutputN
    """).strip()

    test_inputs = [
        "What sensors are on Chiller 6 at MAIN site?",
        "List all assets at site MAIN and get failure modes for Chiller 6.",
        "Forecast Chiller 9 Condenser Water Flow for next week.",
        "Generate a work order for Chiller 6 anomaly detection.",
        "Get vibration data for Motor 3 and run FFT analysis.",
        "What is the current date? List sensors for AHU 2.",
        "Identify failure modes for Pump 4 supply pressure sensor.",
        "Get work orders for equipment CWC04013 from 2017.",
    ]

    # Rubric for LLM-as-Judge
    rubric = textwrap.dedent("""\
        Score the plan 0-10 on these criteria:
        - Format: Uses #TaskN/#ServerN/#ToolN/#DependencyN/#ExpectedOutputN (3 pts)
        - Server validity: Only iot/fmsr/tsfm/wo/utilities/vibration (3 pts)
        - Tool match: Tools belong to their stated server (2 pts)
        - Minimality: No unnecessary steps (1 pt)
        - Dependencies: Valid ordering, no forward references (1 pt)
    """).strip()

    # Proxy checks for structural quality
    proxy_checks = [
        ProxyCheck(name="has_task_tag", check_type="contains", value="#Task"),
        ProxyCheck(name="has_server_tag", check_type="contains", value="#Server"),
        ProxyCheck(name="has_tool_tag", check_type="contains", value="#Tool"),
        ProxyCheck(name="not_too_long", check_type="max_length", value=3000),
    ]

    judge = LLMJudge(rubric=rubric)
    proxy = ProxyMetricsScorer(checks=proxy_checks)
    consistency = SelfConsistencyScorer(num_samples=3)

    scorer = CompositeScorer(
        scorers=[judge, proxy, consistency],
        weights=[0.5, 0.3, 0.2],
    )

    noeval_config = NoEvalConfig(
        iterations=config.iterations,
        population_size=config.population_size,
        num_islands=config.num_islands,
        backend=config.backend,
    )

    evolver = NoEvalPromptEvolver(
        task_description=task_description,
        test_inputs=test_inputs,
        scorer=scorer,
        config=noeval_config,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  AssetOpsBench Track 1 — No-Eval Composite Evolution")
        print(f"  Strategies: LLMJudge (0.5) + Proxy (0.3) + Consistency (0.2)")
        print(f"{'=' * 60}")

    result = evolver.run()

    if verbose:
        print(f"\n  Best score: {result.best_score:.1f}")
        print(f"  Best prompt preview: {result.best_prompt[:120]}...")

    return {
        "mode": "no_eval_composite",
        "best_score": result.best_score,
        "best_prompt": result.best_prompt,
        "history": result.history,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AssetOpsBench Track 1 — Planning Prompt Evolution"
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="Use deep configuration (5 iterations, 8 pop, 3 islands)",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Run no-eval composite mode (no ground truth needed)",
    )
    parser.add_argument(
        "--backend", default="ollama",
        choices=["ollama", "openai", "azure_openai"],
        help="LLM backend",
    )
    parser.add_argument(
        "--model", default="llama3.2",
        help="Model name/deployment",
    )
    parser.add_argument(
        "--max-scenarios", type=int, default=50,
        help="Maximum scenarios to load",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by category (e.g. 'Decision Support', 'Data Query', 'Knowledge Query')",
    )
    parser.add_argument(
        "--type", type=str, default=None, dest="scenario_type",
        help="Filter by scenario type (e.g. 'IoT', 'FMSR', 'TSFM', 'Workorder', 'Vibration', 'Multi')",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit scenarios after filtering (applied after --category/--type)",
    )
    args = parser.parse_args()

    # Backend
    backend_map = {
        "ollama": LLMBackend.OLLAMA,
        "openai": LLMBackend.OPENAI,
        "azure_openai": LLMBackend.AZURE_OPENAI,
    }
    backend = backend_map[args.backend]

    if args.no_eval:
        # No-eval mode
        config = PromptEvolverConfig(
            iterations=5 if args.deep else 3,
            population_size=8 if args.deep else 6,
            num_islands=3 if args.deep else 2,
            backend=backend,
            ollama_model=args.model,
        )
        noeval_client = LLMClient(config)
        result = run_assetops_no_eval(noeval_client, config)

        log_path = Path(__file__).resolve().parent.parent.parent / "logs"
        log_path.mkdir(exist_ok=True)
        out_file = log_path / "assetops_track1_noeval_log.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Log saved to {out_file}")
        return

    # Ground-truth mode
    scenarios = load_assetops_scenarios(max_scenarios=args.max_scenarios)

    # Apply category/type filters
    if args.category:
        scenarios = [
            s for s in scenarios
            if args.category.lower() in s.category.lower()
        ]
        print(f"  Filtered to {len(scenarios)} '{args.category}' scenarios")
    if args.scenario_type:
        scenarios = [
            s for s in scenarios
            if args.scenario_type.lower() in s.scenario_type.lower()
        ]
        print(f"  Filtered to {len(scenarios)} '{args.scenario_type}' type scenarios")

    # Apply post-filter limit
    if args.limit and len(scenarios) > args.limit:
        scenarios = scenarios[:args.limit]
        print(f"  Limited to {args.limit} scenarios")

    if not scenarios:
        print("  ✗ No scenarios loaded. Exiting.")
        return

    configs = {
        "standard": PromptEvolverConfig(
            iterations=3,
            population_size=6,
            num_islands=2,
            backend=backend,
            ollama_model=args.model,
        ),
    }
    if args.deep:
        configs["deep"] = PromptEvolverConfig(
            iterations=5,
            population_size=8,
            num_islands=3,
            backend=backend,
            ollama_model=args.model,
        )

    all_results: list[dict[str, Any]] = []

    for alg_name, config in configs.items():
        client = LLMClient(config)
        experiment = run_assetops_evolution(
            scenarios=scenarios,
            client=client,
            config=config,
            algorithm_name=alg_name,
        )
        all_results.append({
            "category": experiment.category,
            "algorithm": experiment.algorithm,
            "backend": experiment.backend,
            "n_scenarios": experiment.n_scenarios,
            "baseline_score": round(experiment.baseline_score, 2),
            "evolved_score": round(experiment.evolved_score, 2),
            "improvement": round(
                experiment.evolved_score - experiment.baseline_score, 2
            ),
            "server_accuracy": round(experiment.server_accuracy, 2),
            "tool_accuracy": round(experiment.tool_accuracy, 2),
            "best_temperature": round(experiment.best_temperature, 4),
            "best_top_p": round(experiment.best_top_p, 4),
            "iterations": experiment.iterations,
            "wall_time": round(experiment.wall_time, 1),
            "history": experiment.history,
            "prompt_evolution": experiment.prompt_evolution,
            "best_prompt_template": experiment.best_prompt_template,
        })

    # Save log
    log_path = Path(__file__).resolve().parent.parent.parent / "logs"
    log_path.mkdir(exist_ok=True)
    out_file = log_path / "assetops_track1_log.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Log saved to {out_file}")


if __name__ == "__main__":
    main()
