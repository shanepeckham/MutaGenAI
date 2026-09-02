#!/usr/bin/env python3
"""Compare historical and current xLAM optimizers under matched budgets.

Both arms receive identical model, seeds, seed prompts, train/development/
holdout cases, evolutionary shape, and resource ceilings. The old arm uses
the historical wizard Composite scorer and generic mutations. The new arm
uses targeted tool scoring, adaptive critic feedback, warmup, refinement,
and decayed error tracking.

Example:
    OLLAMA_MODEL=llama3.2 uv run python \
        examples/experiments/xlam/run3_old_vs_new_optimizer.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from MutaGenAI import (  # noqa: E402
    CandidateEvaluation,
    ExperimentReport,
    ExperimentRun,
    split_dataset,
)
from MutaGenAI.prompt_evolver import (  # noqa: E402
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)
from MutaGenAI.strategies import (  # noqa: E402
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyMetricsScorer,
    ToolSuccessScorer,
)
from prompt_evolution_xlam import XLAMCase, load_xlam_dataset  # noqa: E402
from prompt_evolution_xlam_no_eval import (  # noqa: E402
    SEED_TEMPLATES,
    TASK_DESCRIPTION,
    build_scorer as build_old_scorer,
    evaluate_on_gt,
)
from run1_adaptive_ollama import (  # noqa: E402
    XLAM_FUNCTION_MUTATIONS,
    _KNOWN_XLAM_TOOLS,
    _LABEL_MAP,
    _parse_xlam_tool_call,
    _xlam_tool_executor,
    extract_function_category,
    xlam_proxy_checks,
)
from run2_research_report import select_mixed_cases  # noqa: E402

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
VARIANTS = ("old", "new")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run matched old-versus-new xLAM optimizer experiments"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model (default: OLLAMA_MODEL or llama3.2)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=9,
        help="Mixed-category cases per seed (default: 9)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[17, 42, 91],
        help="Independent optimizer and split seeds (default: 17 42 91)",
    )
    parser.add_argument(
        "--max-optimizer-calls",
        type=int,
        default=80,
        help="Optimizer-call ceiling for each arm and seed (default: 80)",
    )
    parser.add_argument(
        "--max-target-calls",
        type=int,
        default=180,
        help="Target-call ceiling for each arm and seed (default: 180)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8_000,
        help="Optimizer output-token ceiling per arm and seed (default: 8000)",
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=900.0,
        help="Optimizer wall-time ceiling per arm and seed (default: 900 seconds)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "xlam_old_vs_new_optimizer_report.json",
        help="JSON report destination",
    )
    return parser


def build_new_scorer() -> CompositeScorer:
    """Build the current xLAM-targeted Composite scorer."""
    judge = LLMJudge(
        rubric=(
            "Score 0-10: exact function selection (0-3), parameters copied "
            "from the query (0-3), [func(param=value)] format (0-2), and no "
            "extra prose (0-2)."
        )
    )
    return CompositeScorer(
        [
            (judge, 0.45),
            (
                ToolSuccessScorer(
                    tool_executor=_xlam_tool_executor,
                    parse_fn=_parse_xlam_tool_call,
                ),
                0.30,
            ),
            (ProxyMetricsScorer(xlam_proxy_checks()), 0.25),
        ]
    )


def make_config(args: argparse.Namespace, variant: str) -> NoEvalConfig:
    """Create one arm while keeping all shared controls identical."""
    shared: dict[str, Any] = {
        "iterations": 1,
        "population_size": 4,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.5,
        "crossover_rate": 0.3,
        "migration_interval": 3,
        "backend": LLMBackend.OLLAMA,
        "max_tokens": 128,
        "max_optimizer_calls": args.max_optimizer_calls,
        "max_target_calls": args.max_target_calls,
        "max_output_tokens": args.max_output_tokens,
        "max_wall_time": args.max_wall_time,
    }
    if variant == "old":
        return NoEvalConfig(**shared)
    return NoEvalConfig(
        **shared,
        adaptive_mutations=True,
        llm_mutation_rate=0.3,
        warmup_adaptive=True,
        error_decay=0.5,
        refine_after_splice=True,
    )


def configure_xlam_context(cases: tuple[XLAMCase, ...]) -> list[str]:
    """Build label-free optimizer inputs and targeted-scoring context."""
    from prompt_evolution_xlam import _format_xlam_tools

    _KNOWN_XLAM_TOOLS.clear()
    _LABEL_MAP.clear()
    inputs: list[str] = []
    for case in cases:
        tool_text = _format_xlam_tools(case.tools)
        test_input = f"Available functions:\n{tool_text}\n\nUser query: {case.query}"
        inputs.append(test_input)
        _LABEL_MAP[test_input] = case.expected_function_names[0]
        for tool in case.tools:
            _KNOWN_XLAM_TOOLS.add(tool.get("name", ""))
    return inputs


def unique_top_candidates(
    candidates: list[PromptCandidate],
    limit: int = 2,
) -> list[PromptCandidate]:
    """Return the highest-scoring candidates with unique rendered prompts."""
    selected: list[PromptCandidate] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        prompt = candidate.render_prompt()
        if prompt in seen:
            continue
        seen.add(prompt)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def evaluate_candidate(
    candidate: PromptCandidate,
    cases: tuple[XLAMCase, ...],
    client: LLMClient,
) -> dict[str, float]:
    """Evaluate one candidate with xLAM ground truth."""
    return evaluate_on_gt(candidate.render_prompt(), list(cases), client)


def run_arm(
    args: argparse.Namespace,
    *,
    seed: int,
    variant: str,
    train_cases: tuple[XLAMCase, ...],
    development_cases: tuple[XLAMCase, ...],
    holdout_cases: tuple[XLAMCase, ...],
) -> tuple[ExperimentRun, dict[str, Any]]:
    """Optimize one variant and evaluate its development-selected prompt."""
    test_inputs = configure_xlam_context(train_cases)
    config = make_config(args, variant)
    scorer = build_old_scorer(None) if variant == "old" else build_new_scorer()
    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=scorer,
        config=config,
        seed_templates=SEED_TEMPLATES,
        seed=seed,
        verbose=False,
        extract_category=(extract_function_category if variant == "new" else None),
        custom_mutations=(XLAM_FUNCTION_MUTATIONS if variant == "new" else None),
    )
    result = evolver.run()

    evaluation_client = LLMClient(
        PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=args.model,
            max_tokens=128,
            timeout=60.0,
        )
    )
    finalists = unique_top_candidates(result.all_candidates)
    development_results = [
        evaluate_candidate(candidate, development_cases, evaluation_client)
        for candidate in finalists
    ]
    winner_index = max(
        range(len(finalists)),
        key=lambda index: development_results[index]["overall"],
    )
    winner = finalists[winner_index]
    train_result = evaluate_candidate(winner, train_cases, evaluation_client)
    holdout_result = evaluate_candidate(winner, holdout_cases, evaluation_client)
    split_scores = {
        "train": train_result["overall"] * 100.0,
        "development": development_results[winner_index]["overall"] * 100.0,
        "holdout": holdout_result["overall"] * 100.0,
    }
    optimizer_usage = result.budget_usage
    candidate = CandidateEvaluation("selected", winner.render_prompt(), split_scores)
    run = ExperimentRun(
        seed=seed,
        variant=variant,
        selected_candidate="selected",
        candidates=(candidate,),
        optimizer_calls=optimizer_usage.optimizer_calls,
        target_calls=optimizer_usage.target_calls + evaluation_client.target_calls,
        input_tokens=(
            optimizer_usage.input_tokens + evaluation_client.total_input_tokens
        ),
        output_tokens=(
            optimizer_usage.output_tokens + evaluation_client.total_output_tokens
        ),
    )
    details = {
        "optimizer_fitness": result.best_score,
        "iterations_run": result.iterations_run,
        "stop_reason": optimizer_usage.stop_reason,
        "optimizer_calls": optimizer_usage.optimizer_calls,
        "optimizer_target_calls": optimizer_usage.target_calls,
        "evaluation_target_calls": evaluation_client.target_calls,
        "wall_time": result.wall_time,
        "critic_artifacts": len(result.critic_artifacts),
        "name_accuracy": holdout_result["name_accuracy"] * 100.0,
        "parameter_accuracy": holdout_result["param_accuracy"] * 100.0,
    }
    return run, details


def run(args: argparse.Namespace) -> int:
    """Execute every paired seed and write the research report."""
    if args.cases < 5:
        raise ValueError("--cases must be at least 5")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be unique")
    os.environ["OLLAMA_MODEL"] = args.model
    availability_client = LLMClient(
        PromptEvolverConfig(backend=LLMBackend.OLLAMA, ollama_model=args.model)
    )
    if not availability_client.is_available():
        raise RuntimeError(f"Ollama model is unavailable: {args.model}")

    by_category = load_xlam_dataset(max_per_category=args.cases)
    cases = select_mixed_cases(by_category, args.cases)
    if len(cases) != args.cases:
        raise RuntimeError(f"requested {args.cases} cases, found {len(cases)}")

    runs: list[ExperimentRun] = []
    run_details: list[dict[str, Any]] = []
    for seed in args.seeds:
        splits = split_dataset(cases, seed=seed)
        for variant in VARIANTS:
            print(f"Running seed={seed} variant={variant}...", flush=True)
            experiment_run, details = run_arm(
                args,
                seed=seed,
                variant=variant,
                train_cases=splits.train,
                development_cases=splits.development,
                holdout_cases=splits.holdout,
            )
            runs.append(experiment_run)
            run_details.append({"seed": seed, "variant": variant, **details})
            print(
                f"  holdout={experiment_run.selected.scores['holdout']:.2f}% "
                f"calls={experiment_run.total_calls} "
                f"stop={details['stop_reason']}",
                flush=True,
            )

    report = ExperimentReport(runs)
    comparison = report.compare_variants("new", "old", split="holdout")
    payload = report.to_dict()
    payload.update(
        {
            "experiment": {
                "benchmark": "xlam-function-calling-60k",
                "model": args.model,
                "cases_per_seed": args.cases,
                "seeds": args.seeds,
                "split": {"train": 0.6, "development": 0.2, "holdout": 0.2},
                "shared_budget_per_arm_seed": {
                    "max_optimizer_calls": args.max_optimizer_calls,
                    "max_target_calls": args.max_target_calls,
                    "max_output_tokens": args.max_output_tokens,
                    "max_wall_time": args.max_wall_time,
                    "max_tokens_per_call": 128,
                },
            },
            "configuration_changes": {
                "old": ["historical_composite", "generic_mutations"],
                "new": [
                    "targeted_composite",
                    "tool_success_scorer",
                    "targeted_mutations",
                    "adaptive_mutations",
                    "structured_critic_feedback",
                    "warmup_adaptive",
                    "error_decay_0.5",
                    "refine_after_splice",
                ],
            },
            "run_details": run_details,
            "paired_holdout_comparison": asdict(comparison),
            "variant_summaries": {
                variant: {
                    "holdout": asdict(report.summarize(split="holdout", variant=variant)),
                    "resources": asdict(report.resource_totals(variant=variant)),
                }
                for variant in VARIANTS
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    old = report.summarize(split="holdout", variant="old")
    new = report.summarize(split="holdout", variant="new")
    print("\nMatched optimizer comparison")
    print(f"Old holdout mean: {old.mean:.2f}%")
    print(f"New holdout mean: {new.mean:.2f}%")
    print(
        f"Paired delta: {comparison.differences.mean:+.2f} pp "
        f"(95% CI {comparison.differences.confidence_interval.lower:+.2f} to "
        f"{comparison.differences.confidence_interval.upper:+.2f})"
    )
    print(
        f"Outcomes: {comparison.wins} wins, {comparison.ties} ties, "
        f"{comparison.losses} losses"
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