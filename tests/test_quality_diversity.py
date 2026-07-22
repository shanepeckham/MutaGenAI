"""Tests for MutaGenAI.quality_diversity — Pareto fronts + MAP-Elites."""
from __future__ import annotations

from MutaGenAI.prompt_evolver import PromptCandidate, count_prompt_tokens
from MutaGenAI.quality_diversity import (
    DEFAULT_OBJECTIVES,
    MapElitesArchive,
    Objective,
    build_map_elites,
    pareto_front,
    style_archetype,
)


def _cand(template: str, score: float) -> PromptCandidate:
    return PromptCandidate(template=template, score=score)


class TestStyleArchetype:
    def test_persona(self):
        assert style_archetype("You are a helpful routing assistant.") == "persona"

    def test_example_driven(self):
        assert style_archetype(
            "Route the query.\nExample: input: hi -> output: greet"
        ) == "example_driven"

    def test_structured(self):
        assert style_archetype("Return JSON matching the schema.") == "structured"

    def test_minimal(self):
        assert style_archetype("Pick a tool.") == "minimal"

    def test_verbose(self):
        text = "\n".join(
            f"Consider carefully the nuances of request number {i} here."
            for i in range(8)
        )
        assert style_archetype(text) == "verbose"

    def test_returns_known_archetype(self):
        from MutaGenAI.quality_diversity import STYLE_ARCHETYPES
        assert style_archetype("anything at all here") in STYLE_ARCHETYPES


class TestParetoFront:
    def test_single_candidate(self):
        c = _cand("Pick a tool.", 80.0)
        front = pareto_front([c])
        assert front == [c]

    def test_dominated_candidate_excluded(self):
        # b is worse on score AND longer → dominated by a.
        a = _cand("short", 90.0)
        b = _cand("a much much much longer prompt here", 50.0)
        front = pareto_front([a, b])
        assert a in front
        assert b not in front

    def test_tradeoff_both_on_front(self):
        # Short-but-weaker vs long-but-stronger → both non-dominated.
        short_weak = _cand("x", 70.0)
        long_strong = _cand("a considerably longer and richer prompt", 95.0)
        front = pareto_front([short_weak, long_strong])
        assert len(front) == 2
        assert any(c is short_weak for c in front)
        assert any(c is long_strong for c in front)

    def test_duplicates_collapsed_keep_best(self):
        lo = _cand("same template", 40.0)
        hi = _cand("same template", 88.0)
        front = pareto_front([lo, hi])
        assert front == [hi]

    def test_sorted_best_first(self):
        a = _cand("aaaa", 60.0)
        b = _cand("bb", 90.0)
        front = pareto_front([a, b])
        assert front[0].score >= front[-1].score

    def test_custom_objectives(self):
        # Maximise score only → front is just the single best.
        objs = (Objective("score", lambda c: c.score, maximize=True),)
        a = _cand("a", 30.0)
        b = _cand("bbbb", 80.0)
        front = pareto_front([a, b], objectives=objs)
        assert front == [b]

    def test_empty_input(self):
        assert pareto_front([]) == []


class TestMapElitesArchive:
    def test_keeps_best_per_cell(self):
        arc = MapElitesArchive(token_bin_size=50)
        weak = _cand("Pick a tool now.", 40.0)
        strong = _cand("Pick a tool now.", 90.0)  # same cell
        assert arc.add(weak) is True
        assert arc.add(strong) is True  # replaces weak (higher score)
        assert arc.add(_cand("Pick a tool now.", 10.0)) is False
        assert arc.coverage == 1
        assert arc.best().score == 90.0

    def test_distinct_cells_for_different_styles(self):
        arc = MapElitesArchive()
        arc.add(_cand("You are an expert router.", 50.0))   # persona
        arc.add(_cand("Return JSON output now.", 60.0))     # structured
        assert arc.coverage == 2

    def test_elites_sorted(self):
        arc = build_map_elites([
            _cand("You are a router.", 70.0),
            _cand("Return JSON now.", 95.0),
        ])
        elites = arc.elites()
        assert [e.score for e in elites] == sorted(
            [e.score for e in elites], reverse=True
        )

    def test_best_none_when_empty(self):
        assert MapElitesArchive().best() is None

    def test_to_json_serialisable(self):
        import json
        arc = build_map_elites([
            _cand("You are a router.", 70.0),
            _cand("Return JSON now matching schema.", 95.0),
        ])
        records = arc.to_json()
        json.dumps(records)  # must not raise
        assert all("style" in r and "token_band" in r for r in records)
        assert records[0]["score"] >= records[-1]["score"]

    def test_descriptor_token_band(self):
        arc = MapElitesArchive(token_bin_size=10)
        c = _cand("word " * 30, 50.0)  # ~30+ tokens
        band, style = arc.descriptor(c)
        assert band == count_prompt_tokens(c.template) // 10


def test_default_objectives_shape():
    assert [o.name for o in DEFAULT_OBJECTIVES] == ["score", "tokens"]
    assert DEFAULT_OBJECTIVES[0].maximize is True
    assert DEFAULT_OBJECTIVES[1].maximize is False
