"""Tests for research-grade experiment reporting."""

from __future__ import annotations

import pytest

from MutaGenAI import ExperimentReport as PublicExperimentReport
from MutaGenAI.reporting import (
    CandidateEvaluation,
    ExperimentReport,
    ExperimentRun,
    split_dataset,
    summarize_metric,
)


def _run(
    seed: int,
    variant: str,
    evolved_score: float,
    baseline_score: float = 70.0,
) -> ExperimentRun:
    return ExperimentRun(
        seed=seed,
        variant=variant,
        selected_candidate="evolved",
        candidates=(
            CandidateEvaluation(
                name="baseline",
                prompt="Short baseline prompt.",
                scores={
                    "train": baseline_score + 2,
                    "development": baseline_score + 1,
                    "holdout": baseline_score,
                },
            ),
            CandidateEvaluation(
                name="evolved",
                prompt="A longer evolved prompt with explicit output instructions.",
                scores={
                    "train": evolved_score + 2,
                    "development": evolved_score + 1,
                    "holdout": evolved_score,
                },
            ),
        ),
        optimizer_calls=seed,
        target_calls=seed * 2,
        input_tokens=seed * 100,
        output_tokens=seed * 20,
    )


def test_given_items_when_split_then_partitions_are_reproducible_and_disjoint():
    # Arrange
    items = list(range(20))

    # Act
    first = split_dataset(items, seed=7)
    second = split_dataset(items, seed=7)

    # Assert
    assert first == second
    assert set(first.train).isdisjoint(first.development)
    assert set(first.train).isdisjoint(first.holdout)
    assert set(first.development).isdisjoint(first.holdout)
    assert set(first.train + first.development + first.holdout) == set(items)


def test_given_public_package_when_imported_then_report_is_exported():
    # Assert
    assert PublicExperimentReport is ExperimentReport


def test_given_seeded_scores_when_summarized_then_reports_mean_variance_and_ci():
    # Act
    summary = summarize_metric([70.0, 80.0, 90.0], seed=42)

    # Assert
    assert summary.count == 3
    assert summary.mean == 80.0
    assert summary.variance == 100.0
    assert summary.confidence_interval.lower <= summary.mean
    assert summary.confidence_interval.upper >= summary.mean


def test_given_seed_runner_when_run_seeded_then_executes_every_unique_seed():
    # Arrange
    observed_seeds = []

    def runner(seed: int) -> ExperimentRun:
        observed_seeds.append(seed)
        return _run(seed, "full", 80.0 + seed)

    # Act
    report = ExperimentReport.run_seeded([1, 2, 3], runner)

    # Assert
    assert observed_seeds == [1, 2, 3]
    assert len(report.runs) == 3


def test_given_multiple_runs_when_totaled_then_calls_and_tokens_are_summed():
    # Arrange
    report = ExperimentReport([_run(1, "full", 80), _run(2, "full", 82)])

    # Act
    resources = report.resource_totals()

    # Assert
    assert resources.optimizer_calls == 3
    assert resources.target_calls == 6
    assert resources.total_calls == 9
    assert resources.input_tokens == 300
    assert resources.output_tokens == 60
    assert resources.total_tokens == 360


def test_given_component_variants_when_ablated_then_comparison_is_seed_paired():
    # Arrange
    report = ExperimentReport(
        [
            _run(1, "full", 84),
            _run(2, "full", 86),
            _run(1, "without_critic", 80),
            _run(2, "without_critic", 81),
        ]
    )

    # Act
    comparison = report.ablations("full")["without_critic"]

    # Assert
    assert comparison.differences.values == (4.0, 5.0)
    assert comparison.wins == 2
    assert comparison.ties == 0
    assert comparison.losses == 0


def test_given_candidates_in_same_runs_when_compared_then_pairs_by_run():
    # Arrange
    report = ExperimentReport(
        [
            _run(1, "full", 75, baseline_score=70),
            _run(2, "full", 70, baseline_score=70),
            _run(3, "full", 65, baseline_score=70),
        ]
    )

    # Act
    comparison = report.compare_candidates("evolved", "baseline")

    # Assert
    assert comparison.differences.values == (5.0, 0.0, -5.0)
    assert (comparison.wins, comparison.ties, comparison.losses) == (1, 1, 1)


def test_given_candidate_lengths_when_curved_then_scores_are_binned_by_tokens():
    # Arrange
    report = ExperimentReport([_run(1, "full", 80), _run(2, "full", 90)])

    # Act
    curve = report.prompt_length_curve(bin_width=5)

    # Assert
    assert len(curve) >= 2
    assert sum(point.performance.count for point in curve) == 4
    assert all(point.minimum_tokens <= point.maximum_tokens for point in curve)


def test_given_report_when_serialized_then_includes_runs_splits_and_resources():
    # Arrange
    report = ExperimentReport([_run(1, "full", 80)])

    # Act
    payload = report.to_dict()

    # Assert
    assert payload["variants"] == ["full"]
    assert payload["splits"]["holdout"]["mean"] == 80.0
    assert payload["resources"]["optimizer_calls"] == 1
    assert payload["resources"]["total_calls"] == 3
    assert payload["resources"]["total_tokens"] == 120


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_fraction": 0.0},
        {"development_fraction": 0.0},
        {"train_fraction": 0.8, "development_fraction": 0.2},
    ],
)
def test_given_invalid_split_fractions_when_split_then_raises(kwargs):
    # Act and Assert
    with pytest.raises(ValueError):
        split_dataset(range(10), **kwargs)