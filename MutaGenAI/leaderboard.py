"""Leaderboard of best-known evolved prompts, shipped as seed templates.

Rather than starting evolution from generic seeds, you can start from the
strongest prompt MutaGenAI has discovered for a given benchmark.  Each
entry lives in ``seed_templates/leaderboard/<benchmark>.json`` and records
the winning prompt together with its score, the model it was evolved on,
and the source experiment log.

Typical usage::

    from MutaGenAI import leaderboard_seeds, NoEvalPromptEvolver, NoEvalConfig

    evolver = NoEvalPromptEvolver(
        task_description="Route requests to the right agent.",
        test_inputs=[...],
        scorer=my_scorer,
        seed_templates=leaderboard_seeds("agent_routing"),
    )

Or inspect what is available::

    from MutaGenAI import list_leaderboard, leaderboard_table
    print(list_leaderboard())
    for row in leaderboard_table():
        print(row["benchmark"], row["score"], row["model"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LeaderboardEntry",
    "list_leaderboard",
    "load_leaderboard",
    "leaderboard_seeds",
    "leaderboard_table",
    "best_prompt",
]

_LEADERBOARD_DIR = (
    Path(__file__).resolve().parent.parent / "seed_templates" / "leaderboard"
)


@dataclass
class LeaderboardEntry:
    """A best-known evolved prompt for a benchmark.

    Attributes:
        benchmark:    Short benchmark identifier (the file stem).
        task:         Human-readable description of the task.
        problem_type: ``tool_routing`` / ``classification`` / ``generation``.
        model:        Model the prompt was evolved on.
        score:        Best score achieved (units given by ``metric``).
        metric:       Name of the score metric (e.g. ``accuracy``, ``f1``).
        baseline:     Baseline score for comparison (``None`` if unknown).
        source:       Path to the experiment log the prompt came from.
        prompt:       The winning prompt text.
        seeds:        Seed prompts to start evolution from.  Defaults to
                      ``[prompt]`` when not given explicitly.
        notes:        Optional free-text notes about the prompt.
    """

    benchmark: str
    task: str
    problem_type: str
    model: str
    score: float
    metric: str
    prompt: str
    baseline: float | None = None
    source: str = ""
    seeds: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.seeds:
            self.seeds = [self.prompt]


def list_leaderboard(*, directory: Path | None = None) -> list[str]:
    """Return the benchmark names with a leaderboard entry."""
    base = directory or _LEADERBOARD_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.json"))


def load_leaderboard(
    benchmark: str, *, directory: Path | None = None
) -> LeaderboardEntry:
    """Load the :class:`LeaderboardEntry` for ``benchmark``.

    Raises
    ------
    FileNotFoundError
        If no leaderboard entry exists for ``benchmark``.
    ValueError
        If the entry file is missing required fields.
    """
    base = directory or _LEADERBOARD_DIR
    path = base / f"{benchmark}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No leaderboard entry for {benchmark!r}: {path}\n"
            f"Available: {list_leaderboard(directory=base)}"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for required in ("task", "problem_type", "model", "score", "metric", "prompt"):
        if required not in data:
            raise ValueError(
                f"Leaderboard entry {path.name} is missing required "
                f"field {required!r}."
            )
    return LeaderboardEntry(
        benchmark=data.get("benchmark", benchmark),
        task=data["task"],
        problem_type=data["problem_type"],
        model=data["model"],
        score=float(data["score"]),
        metric=data["metric"],
        prompt=data["prompt"],
        baseline=(
            float(data["baseline"]) if data.get("baseline") is not None else None
        ),
        source=data.get("source", ""),
        seeds=list(data.get("seeds", [])),
        notes=data.get("notes", ""),
    )


def leaderboard_seeds(
    benchmark: str, *, directory: Path | None = None
) -> list[str]:
    """Return the seed prompts for ``benchmark`` (best prompt by default)."""
    return load_leaderboard(benchmark, directory=directory).seeds


def best_prompt(benchmark: str, *, directory: Path | None = None) -> str:
    """Return just the winning prompt text for ``benchmark``."""
    return load_leaderboard(benchmark, directory=directory).prompt


def leaderboard_table(*, directory: Path | None = None) -> list[dict[str, Any]]:
    """Return a summary row per benchmark, sorted by score (best first).

    Each row has ``benchmark``, ``task``, ``model``, ``score``, ``metric``,
    ``baseline``, and ``delta`` (score − baseline, or ``None``).
    """
    rows: list[dict[str, Any]] = []
    for name in list_leaderboard(directory=directory):
        entry = load_leaderboard(name, directory=directory)
        delta = (
            round(entry.score - entry.baseline, 2)
            if entry.baseline is not None
            else None
        )
        rows.append(
            {
                "benchmark": entry.benchmark,
                "task": entry.task,
                "model": entry.model,
                "score": entry.score,
                "metric": entry.metric,
                "baseline": entry.baseline,
                "delta": delta,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows
