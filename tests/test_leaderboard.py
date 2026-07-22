"""Tests for MutaGenAI.leaderboard — best-known prompts as seed templates."""
from __future__ import annotations

import json

import pytest

from MutaGenAI.leaderboard import (
    LeaderboardEntry,
    best_prompt,
    leaderboard_seeds,
    leaderboard_table,
    list_leaderboard,
    load_leaderboard,
)


class TestShippedEntries:
    def test_entries_exist(self):
        names = list_leaderboard()
        assert "toolbench_g1" in names
        assert "agent_routing" in names
        assert len(names) >= 5

    def test_every_entry_loads_and_validates(self):
        for name in list_leaderboard():
            entry = load_leaderboard(name)
            assert isinstance(entry, LeaderboardEntry)
            assert entry.prompt.strip()
            assert entry.problem_type in (
                "tool_routing", "classification", "generation"
            )
            assert entry.seeds and entry.seeds[0] == entry.prompt
            assert entry.score >= 0

    def test_best_prompt_matches_entry(self):
        entry = load_leaderboard("toolbench_g1")
        assert best_prompt("toolbench_g1") == entry.prompt
        assert "{toolbench_apis}" in entry.prompt  # placeholder preserved

    def test_seeds_default_to_prompt(self):
        seeds = leaderboard_seeds("gaia")
        assert seeds == [load_leaderboard("gaia").prompt]

    def test_table_sorted_by_score(self):
        rows = leaderboard_table()
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)
        assert all("delta" in r for r in rows)


class TestErrors:
    def test_missing_benchmark_raises(self):
        with pytest.raises(FileNotFoundError):
            load_leaderboard("does_not_exist")


class TestCustomDirectory:
    def test_load_from_custom_dir(self, tmp_path):
        entry = {
            "benchmark": "demo",
            "task": "Demo task",
            "problem_type": "classification",
            "model": "test-model",
            "score": 99.0,
            "metric": "accuracy",
            "baseline": 50.0,
            "prompt": "Classify precisely.",
        }
        (tmp_path / "demo.json").write_text(json.dumps(entry))
        loaded = load_leaderboard("demo", directory=tmp_path)
        assert loaded.score == 99.0
        assert list_leaderboard(directory=tmp_path) == ["demo"]
        rows = leaderboard_table(directory=tmp_path)
        assert rows[0]["delta"] == 49.0

    def test_missing_required_field_raises(self, tmp_path):
        (tmp_path / "bad.json").write_text(json.dumps({"task": "x"}))
        with pytest.raises(ValueError):
            load_leaderboard("bad", directory=tmp_path)

    def test_explicit_seeds_used(self, tmp_path):
        entry = {
            "benchmark": "multi",
            "task": "t", "problem_type": "generation", "model": "m",
            "score": 1.0, "metric": "accuracy", "prompt": "P0",
            "seeds": ["P0", "P1", "P2"],
        }
        (tmp_path / "multi.json").write_text(json.dumps(entry))
        assert leaderboard_seeds("multi", directory=tmp_path) == ["P0", "P1", "P2"]
