"""Reproducible, benchmark-agnostic reporting for prompt experiments.

The reporting layer stores one record per seeded run and computes aggregate
statistics without making model calls. Scores use the same scale within an
experiment, typically 0 to 100.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

from MutaGenAI.prompt_evolver import PromptEvolverResult, count_prompt_tokens


SPLIT_NAMES = ("train", "development", "holdout")
Item = TypeVar("Item")


@dataclass(frozen=True)
class DatasetSplits(Generic[Item]):
    """Disjoint train, development, and holdout partitions."""

    train: tuple[Item, ...]
    development: tuple[Item, ...]
    holdout: tuple[Item, ...]

    def as_dict(self) -> dict[str, tuple[Item, ...]]:
        """Return partitions keyed by their canonical split names."""
        return {
            "train": self.train,
            "development": self.development,
            "holdout": self.holdout,
        }


def split_dataset(
    items: Sequence[Item],
    *,
    train_fraction: float = 0.6,
    development_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetSplits[Item]:
    """Create a deterministic train/development/holdout partition.

    Args:
        items: Dataset items to partition.
        train_fraction: Fraction reserved for prompt optimization.
        development_fraction: Fraction used for candidate selection.
        seed: Shuffle seed recorded by the calling experiment.

    Returns:
        Immutable, disjoint dataset partitions. The remaining fraction is
        assigned to holdout.

    Raises:
        ValueError: If fractions are invalid or fewer than three items exist.
    """
    if len(items) < 3:
        raise ValueError("at least three items are required for three-way splitting")
    if not 0 < train_fraction < 1 or not 0 < development_fraction < 1:
        raise ValueError("train and development fractions must be between 0 and 1")
    if train_fraction + development_fraction >= 1:
        raise ValueError("train and development fractions must leave a holdout set")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * train_fraction))
    development_count = max(1, int(len(shuffled) * development_fraction))
    if train_count + development_count >= len(shuffled):
        development_count = 1
        train_count = len(shuffled) - 2

    return DatasetSplits(
        train=tuple(shuffled[:train_count]),
        development=tuple(
            shuffled[train_count : train_count + development_count]
        ),
        holdout=tuple(shuffled[train_count + development_count :]),
    )


@dataclass(frozen=True)
class ConfidenceInterval:
    """Two-sided confidence interval."""

    lower: float
    upper: float
    level: float = 0.95


@dataclass(frozen=True)
class MetricSummary:
    """Descriptive statistics across independent seeded runs."""

    count: int
    mean: float
    variance: float
    confidence_interval: ConfidenceInterval
    values: tuple[float, ...]


def summarize_metric(
    values: Iterable[float],
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
) -> MetricSummary:
    """Summarize values with sample variance and a bootstrap mean interval."""
    observations = tuple(float(value) for value in values)
    if not observations:
        raise ValueError("at least one observation is required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")

    mean = statistics.fmean(observations)
    variance = statistics.variance(observations) if len(observations) > 1 else 0.0
    if len(observations) == 1:
        interval = ConfidenceInterval(mean, mean, confidence_level)
    else:
        rng = random.Random(seed)
        bootstrap_means = sorted(
            statistics.fmean(rng.choices(observations, k=len(observations)))
            for _ in range(bootstrap_resamples)
        )
        tail = (1.0 - confidence_level) / 2.0
        lower_index = max(0, math.floor(tail * (bootstrap_resamples - 1)))
        upper_index = min(
            bootstrap_resamples - 1,
            math.ceil((1.0 - tail) * (bootstrap_resamples - 1)),
        )
        interval = ConfidenceInterval(
            bootstrap_means[lower_index],
            bootstrap_means[upper_index],
            confidence_level,
        )
    return MetricSummary(len(observations), mean, variance, interval, observations)


@dataclass(frozen=True)
class CandidateEvaluation:
    """One prompt candidate evaluated on explicit dataset splits."""

    name: str
    prompt: str
    scores: dict[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.scores) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")

    @property
    def prompt_tokens(self) -> int:
        """Return the prompt's approximate token count."""
        return count_prompt_tokens(self.prompt)


@dataclass(frozen=True)
class ExperimentRun:
    """Results and resource usage for one independent seeded run."""

    seed: int
    variant: str
    selected_candidate: str
    candidates: tuple[CandidateEvaluation, ...]
    optimizer_calls: int = 0
    target_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        names = [candidate.name for candidate in self.candidates]
        if not names:
            raise ValueError("a run must contain at least one candidate")
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique within a run")
        if self.selected_candidate not in names:
            raise ValueError("selected_candidate must identify a run candidate")

    @property
    def selected(self) -> CandidateEvaluation:
        """Return the candidate selected on development data."""
        return next(
            candidate
            for candidate in self.candidates
            if candidate.name == self.selected_candidate
        )

    @property
    def total_calls(self) -> int:
        return self.optimizer_calls + self.target_calls

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_result(
        cls,
        result: PromptEvolverResult,
        *,
        seed: int,
        variant: str,
        split_scores: dict[str, float],
        candidate_name: str = "evolved",
    ) -> ExperimentRun:
        """Build a run record from an evolution result and external split scores."""
        usage = result.budget_usage
        return cls(
            seed=seed,
            variant=variant,
            selected_candidate=candidate_name,
            candidates=(
                CandidateEvaluation(candidate_name, result.best_prompt, split_scores),
            ),
            optimizer_calls=usage.optimizer_calls,
            target_calls=usage.target_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


@dataclass(frozen=True)
class ResourceTotals:
    """Aggregate LLM resource consumption."""

    optimizer_calls: int
    target_calls: int
    input_tokens: int
    output_tokens: int

    @property
    def total_calls(self) -> int:
        return self.optimizer_calls + self.target_calls

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class PairedComparison:
    """Seed-matched score difference between two candidates or variants."""

    left: str
    right: str
    split: str
    differences: MetricSummary
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True)
class PromptLengthPoint:
    """Aggregate performance for one prompt-length bin."""

    minimum_tokens: int
    maximum_tokens: int
    performance: MetricSummary


class ExperimentReport:
    """Aggregate and compare research experiment runs."""

    def __init__(self, runs: Iterable[ExperimentRun]) -> None:
        self.runs = tuple(runs)
        if not self.runs:
            raise ValueError("at least one experiment run is required")
        identities = [(run.variant, run.seed) for run in self.runs]
        if len(identities) != len(set(identities)):
            raise ValueError("each variant and seed pair must be unique")

    @classmethod
    def run_seeded(
        cls,
        seeds: Iterable[int],
        runner: Callable[[int], ExperimentRun],
    ) -> ExperimentReport:
        """Execute the same experiment independently for every seed."""
        seed_values = tuple(seeds)
        if len(seed_values) != len(set(seed_values)):
            raise ValueError("seeds must be unique")
        return cls(runner(seed) for seed in seed_values)

    @property
    def variants(self) -> tuple[str, ...]:
        """Return sorted component variants represented by the report."""
        return tuple(sorted({run.variant for run in self.runs}))

    def summarize(
        self,
        *,
        split: str = "holdout",
        variant: str | None = None,
    ) -> MetricSummary:
        """Summarize selected-candidate performance across seeded runs."""
        self._validate_split(split)
        values = [
            run.selected.scores[split]
            for run in self.runs
            if (variant is None or run.variant == variant)
            and split in run.selected.scores
        ]
        return summarize_metric(values)

    def resource_totals(self, *, variant: str | None = None) -> ResourceTotals:
        """Sum calls and tokens across all matching seeded runs."""
        matching = [
            run for run in self.runs if variant is None or run.variant == variant
        ]
        return ResourceTotals(
            optimizer_calls=sum(run.optimizer_calls for run in matching),
            target_calls=sum(run.target_calls for run in matching),
            input_tokens=sum(run.input_tokens for run in matching),
            output_tokens=sum(run.output_tokens for run in matching),
        )

    def compare_variants(
        self,
        left: str,
        right: str,
        *,
        split: str = "holdout",
    ) -> PairedComparison:
        """Compare two component variants using only their shared seeds."""
        self._validate_split(split)
        left_by_seed = {
            run.seed: run.selected.scores[split]
            for run in self.runs
            if run.variant == left and split in run.selected.scores
        }
        right_by_seed = {
            run.seed: run.selected.scores[split]
            for run in self.runs
            if run.variant == right and split in run.selected.scores
        }
        common_seeds = sorted(set(left_by_seed) & set(right_by_seed))
        if not common_seeds:
            raise ValueError("paired comparisons require at least one shared seed")
        differences = tuple(
            left_by_seed[seed] - right_by_seed[seed] for seed in common_seeds
        )
        return PairedComparison(
            left=left,
            right=right,
            split=split,
            differences=summarize_metric(differences),
            wins=sum(difference > 0 for difference in differences),
            ties=sum(difference == 0 for difference in differences),
            losses=sum(difference < 0 for difference in differences),
        )

    def compare_candidates(
        self,
        left: str,
        right: str,
        *,
        split: str = "holdout",
        variant: str | None = None,
    ) -> PairedComparison:
        """Compare candidates evaluated within the same seeded runs."""
        self._validate_split(split)
        differences = []
        for run in self.runs:
            if variant is not None and run.variant != variant:
                continue
            by_name = {candidate.name: candidate for candidate in run.candidates}
            if left not in by_name or right not in by_name:
                continue
            if split not in by_name[left].scores or split not in by_name[right].scores:
                continue
            differences.append(
                by_name[left].scores[split] - by_name[right].scores[split]
            )
        if not differences:
            raise ValueError("candidates were not paired in any matching run")
        return PairedComparison(
            left=left,
            right=right,
            split=split,
            differences=summarize_metric(differences),
            wins=sum(difference > 0 for difference in differences),
            ties=sum(difference == 0 for difference in differences),
            losses=sum(difference < 0 for difference in differences),
        )

    def ablations(
        self,
        reference: str,
        *,
        split: str = "holdout",
    ) -> dict[str, PairedComparison]:
        """Return reference-minus-ablation deltas for every other variant."""
        return {
            variant: self.compare_variants(reference, variant, split=split)
            for variant in self.variants
            if variant != reference
        }

    def prompt_length_curve(
        self,
        *,
        split: str = "holdout",
        bin_width: int = 25,
    ) -> tuple[PromptLengthPoint, ...]:
        """Aggregate candidate performance by prompt-token-length bins."""
        self._validate_split(split)
        if bin_width < 1:
            raise ValueError("bin_width must be positive")
        bins: dict[int, list[float]] = {}
        for run in self.runs:
            for candidate in run.candidates:
                if split not in candidate.scores:
                    continue
                lower = (candidate.prompt_tokens // bin_width) * bin_width
                bins.setdefault(lower, []).append(candidate.scores[split])
        return tuple(
            PromptLengthPoint(
                minimum_tokens=lower,
                maximum_tokens=lower + bin_width - 1,
                performance=summarize_metric(values),
            )
            for lower, values in sorted(bins.items())
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report with core aggregate metrics."""
        resources = self.resource_totals()
        return {
            "runs": [asdict(run) for run in self.runs],
            "variants": list(self.variants),
            "resources": {
                **asdict(resources),
                "total_calls": resources.total_calls,
                "total_tokens": resources.total_tokens,
            },
            "splits": {
                split: asdict(self.summarize(split=split))
                for split in SPLIT_NAMES
                if any(split in run.selected.scores for run in self.runs)
            },
        }

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")