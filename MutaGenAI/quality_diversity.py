"""Quality-diversity analysis for evolved prompts: Pareto fronts + MAP-Elites.

Standard evolution returns a single winner.  But for prompt engineering the
*frontier* is often more useful than the peak: a slightly less accurate but
far shorter prompt, or a stylistically different prompt that is more robust,
may be the better deployment choice.  This module derives two complementary
views from the candidates an evolution run already produced — no extra LLM
calls required:

- **Pareto front** — the set of non-dominated prompts across multiple
  objectives (by default: maximise accuracy, minimise token length).  This
  directly surfaces the accuracy/cost trade-off that the token-optimisation
  features target.
- **MAP-Elites archive** — a quality-diversity grid that keeps the best
  prompt in each behavioural cell.  Cells are defined by token-length band
  and *style archetype*, so the archive illuminates which kinds of prompts
  work, not just the single highest scorer.

Both operate on any iterable of :class:`~MutaGenAI.prompt_evolver.PromptCandidate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from MutaGenAI.prompt_evolver import PromptCandidate, count_prompt_tokens

__all__ = [
    "Objective",
    "DEFAULT_OBJECTIVES",
    "style_archetype",
    "pareto_front",
    "MapElitesArchive",
    "build_map_elites",
]


# ---------------------------------------------------------------------------
# Style archetype detection
# ---------------------------------------------------------------------------

#: The fixed set of style archetypes a prompt can be classified into.
STYLE_ARCHETYPES = (
    "minimal",
    "persona",
    "example_driven",
    "structured",
    "verbose",
)

_PERSONA_MARKERS = ("you are ", "you're ", "act as ", "as an expert", "your role")
_EXAMPLE_MARKERS = ("example", "e.g.", "for instance", "input:", "output:", "->")
_STRUCTURED_MARKERS = (
    "json", "schema", "format", "step 1", "1.", "2.", "- ", "* ", "rules:",
)


def style_archetype(template: str) -> str:
    """Classify a prompt template into one of :data:`STYLE_ARCHETYPES`.

    The classification is a deterministic heuristic based on structural
    cues, evaluated in priority order:

    1. ``persona`` — opens by assigning the model a role/identity.
    2. ``example_driven`` — contains worked input/output examples.
    3. ``structured`` — uses JSON/schema/format keywords or list markers.
    4. ``minimal`` — very short (≤ 3 non-empty lines and < 40 tokens).
    5. ``verbose`` — everything else (long, prose-heavy prompts).

    Returns one of the strings in :data:`STYLE_ARCHETYPES`.
    """
    text = template.strip()
    lowered = text.lower()
    head = lowered[:120]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    tokens = count_prompt_tokens(text)

    if any(marker in head for marker in _PERSONA_MARKERS):
        return "persona"
    if any(marker in lowered for marker in _EXAMPLE_MARKERS):
        return "example_driven"
    if any(marker in lowered for marker in _STRUCTURED_MARKERS):
        return "structured"
    if len(lines) <= 3 and tokens < 40:
        return "minimal"
    return "verbose"


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Objective:
    """A single optimisation objective for Pareto analysis.

    Attributes:
        name:     Human-readable objective name (e.g. ``"score"``).
        key:      Callable mapping a candidate to a numeric value.
        maximize: ``True`` to prefer larger values, ``False`` for smaller.
    """

    name: str
    key: Callable[[PromptCandidate], float]
    maximize: bool = True

    def value(self, candidate: PromptCandidate) -> float:
        """Return this objective's (sign-normalised) value for a candidate.

        Values are normalised so that *larger is always better*, which lets
        domination checks treat every objective uniformly.
        """
        raw = float(self.key(candidate))
        return raw if self.maximize else -raw


#: Default objectives: maximise fitness score, minimise prompt token length.
DEFAULT_OBJECTIVES: tuple[Objective, ...] = (
    Objective("score", lambda c: c.score, maximize=True),
    Objective(
        "tokens", lambda c: float(count_prompt_tokens(c.template)), maximize=False
    ),
)


def _dominates(
    a: PromptCandidate,
    b: PromptCandidate,
    objectives: Sequence[Objective],
) -> bool:
    """Return ``True`` if ``a`` Pareto-dominates ``b``.

    ``a`` dominates ``b`` when it is at least as good on every objective and
    strictly better on at least one.
    """
    at_least_as_good = True
    strictly_better = False
    for obj in objectives:
        va, vb = obj.value(a), obj.value(b)
        if va < vb:
            at_least_as_good = False
            break
        if va > vb:
            strictly_better = True
    return at_least_as_good and strictly_better


def pareto_front(
    candidates: Iterable[PromptCandidate],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
) -> list[PromptCandidate]:
    """Return the non-dominated prompts across ``objectives``.

    Duplicate templates are collapsed (the first occurrence is kept) so the
    front contains distinct prompts.  The result is sorted by the first
    objective, best first.

    Parameters
    ----------
    candidates :
        Any iterable of evaluated candidates.
    objectives :
        Objectives to trade off.  Defaults to :data:`DEFAULT_OBJECTIVES`
        (maximise score, minimise tokens).
    """
    # Collapse duplicate templates, keeping the highest first-objective value.
    unique: dict[str, PromptCandidate] = {}
    primary = objectives[0] if objectives else None
    for cand in candidates:
        existing = unique.get(cand.template)
        if existing is None:
            unique[cand.template] = cand
        elif primary is not None and primary.value(cand) > primary.value(existing):
            unique[cand.template] = cand
    pool = list(unique.values())

    front = [
        cand
        for cand in pool
        if not any(
            other is not cand and _dominates(other, cand, objectives)
            for other in pool
        )
    ]
    if primary is not None:
        front.sort(key=primary.value, reverse=True)
    return front


# ---------------------------------------------------------------------------
# MAP-Elites archive
# ---------------------------------------------------------------------------


@dataclass
class MapElitesArchive:
    """A quality-diversity archive keeping the best prompt per behaviour cell.

    Behaviour is described by ``(token_band, style_archetype)``.  Adding a
    candidate keeps it only if its cell is empty or it beats the current
    occupant's fitness (``score``).

    Attributes:
        token_bin_size: Width of each token-length band.  Default ``50``.
        cells:          Mapping of ``(token_band, style)`` → best candidate.
    """

    token_bin_size: int = 50
    cells: dict[tuple[int, str], PromptCandidate] = field(default_factory=dict)

    def descriptor(self, candidate: PromptCandidate) -> tuple[int, str]:
        """Return the behavioural descriptor cell for a candidate."""
        tokens = count_prompt_tokens(candidate.template)
        band = tokens // max(1, self.token_bin_size)
        return (band, style_archetype(candidate.template))

    def add(self, candidate: PromptCandidate) -> bool:
        """Insert a candidate.  Returns ``True`` if it became a cell elite."""
        key = self.descriptor(candidate)
        current = self.cells.get(key)
        if current is None or candidate.score > current.score:
            self.cells[key] = candidate
            return True
        return False

    def add_all(self, candidates: Iterable[PromptCandidate]) -> int:
        """Add many candidates; return how many became cell elites."""
        return sum(1 for c in candidates if self.add(c))

    def elites(self) -> list[PromptCandidate]:
        """Return all cell elites, sorted by fitness (best first)."""
        return sorted(self.cells.values(), key=lambda c: c.score, reverse=True)

    @property
    def coverage(self) -> int:
        """Number of occupied behaviour cells."""
        return len(self.cells)

    def best(self) -> Optional[PromptCandidate]:
        """Return the highest-scoring elite, or ``None`` when empty."""
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: c.score)

    def to_json(self) -> list[dict]:
        """Serialise the archive to a list of JSON-friendly cell records."""
        records = []
        for (band, style), cand in self.cells.items():
            records.append(
                {
                    "token_band": band,
                    "token_band_range": [
                        band * self.token_bin_size,
                        (band + 1) * self.token_bin_size,
                    ],
                    "style": style,
                    "score": round(cand.score, 2),
                    "tokens": count_prompt_tokens(cand.template),
                    "temperature": round(cand.temperature, 4),
                    "top_p": round(cand.top_p, 4),
                    "template": cand.template,
                }
            )
        records.sort(key=lambda r: r["score"], reverse=True)
        return records


def build_map_elites(
    candidates: Iterable[PromptCandidate],
    token_bin_size: int = 50,
) -> MapElitesArchive:
    """Build a :class:`MapElitesArchive` from candidates in one call."""
    archive = MapElitesArchive(token_bin_size=token_bin_size)
    archive.add_all(candidates)
    return archive
