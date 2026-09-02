#!/usr/bin/env python3
"""Compare existing xLAM default and evolved prompts across seeded splits.

The experiment reuses the xLAM cookbook dataset loader, scorer, default prompt,
and a previously evolved prompt from ``logs/xlam_experiment_log.json``. It
performs no optimization calls; every reported call is a target-model
evaluation.

Example:
    OLLAMA_MODEL=llama3.2 uv run python \
        examples/experiments/xlam/run2_research_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
COOKBOOK_DIR = ROOT / "examples" / "cookbook"
sys.path.insert(0, str(COOKBOOK_DIR))

from MutaGenAI import (  # noqa: E402
    CandidateEvaluation,
    ExperimentReport,
    ExperimentRun,
    split_dataset,
)
from MutaGenAI.prompt_evolver import (  # noqa: E402
    LLMBackend,
    LLMClient,
    PromptEvolverConfig,
    count_prompt_tokens,
)
from prompt_evolution_xlam import (  # noqa: E402
    XLAMCase,
    _XLAM_DEFAULT_PROMPT,
    _format_xlam_tools,
    load_xlam_dataset,
    score_xlam_case,
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
DEFAULT_SEEDS = (17, 42, 91)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare existing xLAM prompts with research-grade reporting"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        help="Local Ollama model (default: OLLAMA_MODEL or llama3.2)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=12,
        help="Number of mixed-category cases per seeded run (default: 12)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Independent split seeds (default: 17 42 91)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "logs" / "xlam_experiment_log.json",
        help="Existing xLAM experiment log containing an evolved prompt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "xlam_first_research_report.json",
        help="Destination for the JSON report",
    )
    return parser


def load_evolved_prompt(log_path: Path) -> str:
    """Load the highest-scoring Ollama prompt from an existing xLAM log."""
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    experiments = [
        experiment
        for experiment in payload.get("experiments", [])
        if experiment.get("backend") == "ollama"
        and experiment.get("best_prompt_template")
    ]
    if not experiments:
        raise ValueError(f"no evolved Ollama prompt found in {log_path}")
    winner = max(experiments, key=lambda experiment: experiment["evolved_score"])
    return str(winner["best_prompt_template"])


def select_mixed_cases(
    by_category: dict[str, list[XLAMCase]],
    case_count: int,
) -> list[XLAMCase]:
    """Select cases round-robin so each available category is represented."""
    selected: list[XLAMCase] = []
    categories = sorted(by_category)
    case_index = 0
    while len(selected) < case_count:
        added = False
        for category in categories:
            cases = by_category[category]
            if case_index < len(cases):
                selected.append(cases[case_index])
                added = True
                if len(selected) == case_count:
                    break
        if not added:
            break
        case_index += 1
    return selected


def evaluate_prompt(
    prompt_template: str,
    cases: tuple[XLAMCase, ...],
    client: LLMClient,
) -> tuple[float, int, int]:
    """Evaluate one prompt and return score plus approximate token usage."""
    scores: list[float] = []
    input_tokens = 0
    output_tokens = 0
    for case in cases:
        tools = _format_xlam_tools(case.tools)
        system_prompt = prompt_template.replace("{xlam_functions}", tools)
        response = client.complete(
            system_prompt=system_prompt,
            user_message=case.query,
            temperature=0.1,
            top_p=0.95,
        ) or ""
        scores.append(score_xlam_case(response, case) * 100.0)
        input_tokens += count_prompt_tokens(system_prompt)
        input_tokens += count_prompt_tokens(case.query)
        output_tokens += count_prompt_tokens(response)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, input_tokens, output_tokens


def build_run(
    seed: int,
    cases: list[XLAMCase],
    prompts: dict[str, str],
    client: LLMClient,
) -> ExperimentRun:
    """Evaluate paired candidates on one deterministic three-way split."""
    splits = split_dataset(cases, seed=seed)
    candidates: list[CandidateEvaluation] = []
    input_tokens = 0
    output_tokens = 0
    target_calls = 0

    for candidate_name, prompt in prompts.items():
        scores: dict[str, float] = {}
        for split_name, split_cases in splits.as_dict().items():
            score, split_input_tokens, split_output_tokens = evaluate_prompt(
                prompt, split_cases, client
            )
            scores[split_name] = score
            input_tokens += split_input_tokens
            output_tokens += split_output_tokens
            target_calls += len(split_cases)
        candidates.append(CandidateEvaluation(candidate_name, prompt, scores))

    selected = max(candidates, key=lambda candidate: candidate.scores["development"])
    return ExperimentRun(
        seed=seed,
        variant="existing_xlam_setup",
        selected_candidate=selected.name,
        candidates=tuple(candidates),
        target_calls=target_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def serialize_report(
    report: ExperimentReport,
    comparison: Any,
    *,
    model: str,
    case_count: int,
) -> dict[str, Any]:
    """Build a JSON-compatible experiment result."""
    payload = report.to_dict()
    payload["experiment"] = {
        "benchmark": "xlam-function-calling-60k",
        "model": model,
        "cases_per_seed": case_count,
        "comparison": "evolved_minus_default",
    }
    payload["paired_holdout_comparison"] = asdict(comparison)
    payload["prompt_length_curve"] = [
        asdict(point) for point in report.prompt_length_curve(split="holdout")
    ]
    return payload


def run(args: argparse.Namespace) -> int:
    """Run the seeded xLAM comparison and persist its report."""
    if args.cases < 3:
        raise ValueError("--cases must be at least 3")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be unique")

    config = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=args.model,
        timeout=60.0,
    )
    client = LLMClient(config)
    if not client.is_available():
        raise RuntimeError("Ollama is not available at localhost:11434")

    by_category = load_xlam_dataset(max_per_category=args.cases)
    cases = select_mixed_cases(by_category, args.cases)
    if len(cases) < args.cases:
        raise RuntimeError(f"only {len(cases)} xLAM cases were available")

    prompts = {
        "default": _XLAM_DEFAULT_PROMPT,
        "evolved": load_evolved_prompt(args.log),
    }
    report = ExperimentReport.run_seeded(
        args.seeds,
        lambda seed: build_run(seed, cases, prompts, client),
    )
    comparison = report.compare_candidates("evolved", "default", split="holdout")
    totals = report.resource_totals()

    default_summary = ExperimentReport(
        ExperimentRun(
            seed=run.seed,
            variant="default",
            selected_candidate="default",
            candidates=(next(c for c in run.candidates if c.name == "default"),),
        )
        for run in report.runs
    ).summarize(split="holdout")
    evolved_summary = ExperimentReport(
        ExperimentRun(
            seed=run.seed,
            variant="evolved",
            selected_candidate="evolved",
            candidates=(next(c for c in run.candidates if c.name == "evolved"),),
        )
        for run in report.runs
    ).summarize(split="holdout")

    print(f"xLAM first research experiment ({args.model})")
    print(f"Seeds: {', '.join(map(str, args.seeds))}; cases/seed: {args.cases}")
    print(
        f"Default holdout: {default_summary.mean:.2f}% "
        f"(95% CI {default_summary.confidence_interval.lower:.2f} to "
        f"{default_summary.confidence_interval.upper:.2f})"
    )
    print(
        f"Evolved holdout: {evolved_summary.mean:.2f}% "
        f"(95% CI {evolved_summary.confidence_interval.lower:.2f} to "
        f"{evolved_summary.confidence_interval.upper:.2f})"
    )
    print(
        f"Paired delta: {comparison.differences.mean:+.2f} pp "
        f"({comparison.wins} wins, {comparison.ties} ties, "
        f"{comparison.losses} losses)"
    )
    print(
        f"Usage: {totals.target_calls} target calls, "
        f"{totals.input_tokens} input tokens, {totals.output_tokens} output tokens"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            serialize_report(
                report,
                comparison,
                model=args.model,
                case_count=args.cases,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report: {args.output}")
    return EXIT_SUCCESS


def main() -> int:
    """Run the CLI with top-level error handling."""
    try:
        return run(create_parser().parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())