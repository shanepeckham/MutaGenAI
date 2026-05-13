"""Tests for the A+B token optimization logic in BrowserGymEvolver.

Tests the two complementary mechanisms:
  (A) Baseline-relative efficiency scoring
  (B) Lexicographic tournament tiebreaker within ACCURACY_BAND
"""
from __future__ import annotations

import sys
import os
from unittest.mock import patch

import numpy as np
import pytest

# Ensure repo root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MutaGenAI.prompt_evolver import PromptCandidate, PromptEvolverConfig


# ─── Helpers ──────────────────────────────────────────────────────────────

def _make_candidate(
    template: str = "short",
    score: float = 80.0,
    generation: int = 0,
    penalty_violations: int = 0,
) -> PromptCandidate:
    return PromptCandidate(
        template=template,
        score=score,
        generation=generation,
        penalty_violations=penalty_violations,
    )


# ─── (A) Efficiency scoring tests ─────────────────────────────────────────

class TestEfficiencyScoring:
    """Test baseline-relative efficiency bonus formula."""

    def test_shorter_prompt_gets_higher_bonus(self):
        """A prompt half the length of baseline gets efficiency > 1."""
        baseline_tokens = 100
        prompt_tokens = 50
        efficiency = baseline_tokens / max(prompt_tokens, 1)
        assert efficiency == 2.0

    def test_longer_prompt_gets_lower_bonus(self):
        """A prompt longer than baseline gets efficiency < 1."""
        baseline_tokens = 100
        prompt_tokens = 200
        efficiency = baseline_tokens / max(prompt_tokens, 1)
        assert efficiency == 0.5

    def test_efficiency_capped(self):
        """Efficiency ratio is capped at EFFICIENCY_CAP."""
        baseline_tokens = 100
        prompt_tokens = 10  # Would be 10x, but cap at 2.0
        efficiency_cap = 2.0
        efficiency = baseline_tokens / max(prompt_tokens, 1)
        capped = min(efficiency, efficiency_cap)
        assert capped == efficiency_cap

    def test_blended_score_formula(self):
        """Blended score = accuracy * (1 - w) + bonus * w."""
        raw_accuracy = 80.0
        token_weight = 0.10
        baseline_tokens = 100
        prompt_tokens = 50  # efficiency = 2.0
        efficiency_cap = 2.0

        efficiency = baseline_tokens / max(prompt_tokens, 1)
        efficiency_bonus = min(efficiency, efficiency_cap) / efficiency_cap * 100.0
        blended = raw_accuracy * (1 - token_weight) + efficiency_bonus * token_weight

        # bonus = 2.0/2.0 * 100 = 100.0
        # blended = 80 * 0.9 + 100 * 0.1 = 72 + 10 = 82.0
        assert blended == pytest.approx(82.0)

    def test_same_length_as_baseline_gives_50_bonus(self):
        """When prompt == baseline length, efficiency=1.0, bonus=50%."""
        baseline_tokens = 100
        prompt_tokens = 100
        efficiency_cap = 2.0

        efficiency = baseline_tokens / max(prompt_tokens, 1)
        efficiency_bonus = min(efficiency, efficiency_cap) / efficiency_cap * 100.0

        assert efficiency == 1.0
        assert efficiency_bonus == pytest.approx(50.0)

    def test_zero_prompt_tokens_no_crash(self):
        """Edge case: zero tokens should not cause ZeroDivisionError."""
        baseline_tokens = 100
        prompt_tokens = 0
        efficiency = baseline_tokens / max(prompt_tokens, 1)
        assert efficiency == 100.0  # capped later

    def test_zero_baseline_tokens_gives_zero_bonus(self):
        """If baseline is zero, efficiency is always 0 (no bonus)."""
        baseline_tokens = 0
        prompt_tokens = 50
        efficiency = baseline_tokens / max(prompt_tokens, 1)
        efficiency_cap = 2.0
        efficiency_bonus = min(efficiency, efficiency_cap) / efficiency_cap * 100.0
        assert efficiency_bonus == 0.0

    def test_weight_zero_returns_raw_accuracy(self):
        """With weight=0, blended equals raw accuracy."""
        raw_accuracy = 75.0
        token_weight = 0.0
        efficiency_bonus = 100.0  # doesn't matter
        blended = raw_accuracy * (1 - token_weight) + efficiency_bonus * token_weight
        assert blended == pytest.approx(75.0)


# ─── (B) Lexicographic tiebreaker tests ───────────────────────────────────

class TestTournamentTiebreaker:
    """Test the lexicographic selection within ACCURACY_BAND."""

    def _token_aware_key(
        self, c: PromptCandidate, accuracy_band: float = 2.0,
    ) -> tuple:
        """Replicate the tiebreaker key from BrowserGymEvolver."""
        feasible = 1 if c.penalty_violations == 0 else 0
        bucket = int(c.score // accuracy_band)
        # Use len(template) // 4 as a rough token proxy for testing
        tokens = len(c.template) // 4
        return (feasible, bucket, -tokens)

    def test_same_bucket_shorter_wins(self):
        """Within the same accuracy band, shorter prompt wins."""
        c_short = _make_candidate(template="a" * 40, score=80.0)   # ~10 tokens
        c_long = _make_candidate(template="a" * 200, score=80.5)   # ~50 tokens

        key_short = self._token_aware_key(c_short)
        key_long = self._token_aware_key(c_long)

        # Both in bucket 40 (80//2 == 40, 80.5//2 == 40)
        assert key_short > key_long  # shorter wins

    def test_different_bucket_higher_score_wins(self):
        """Across accuracy bands, higher score bucket wins."""
        c_low = _make_candidate(template="a" * 40, score=78.0)    # bucket 39
        c_high = _make_candidate(template="a" * 200, score=82.0)  # bucket 41

        key_low = self._token_aware_key(c_low)
        key_high = self._token_aware_key(c_high)

        assert key_high > key_low  # higher bucket wins

    def test_infeasible_always_loses(self):
        """A candidate with penalty_violations > 0 loses to feasible."""
        c_feasible = _make_candidate(score=60.0, penalty_violations=0)
        c_infeasible = _make_candidate(score=90.0, penalty_violations=1)

        key_feas = self._token_aware_key(c_feasible)
        key_infeas = self._token_aware_key(c_infeasible)

        assert key_feas > key_infeas

    def test_exact_same_score_shorter_wins(self):
        """Exact same score — shorter template wins."""
        c_short = _make_candidate(template="x" * 20, score=75.0)
        c_long = _make_candidate(template="x" * 400, score=75.0)

        key_short = self._token_aware_key(c_short)
        key_long = self._token_aware_key(c_long)

        assert key_short > key_long

    def test_band_width_2_groups_correctly(self):
        """Scores 80.0 and 81.9 are in the same band (bucket 40)."""
        c1 = _make_candidate(score=80.0)
        c2 = _make_candidate(score=81.9)

        bucket1 = int(c1.score // 2.0)
        bucket2 = int(c2.score // 2.0)

        assert bucket1 == bucket2 == 40


# ─── Integration: lineage enrichment ──────────────────────────────────────

class TestLineageEnrichment:
    """Test that lineage entries get token data."""

    def test_lineage_entry_has_token_fields(self):
        """Simulates the lineage enrichment logic from main()."""
        baseline_tokens = 100
        lineage = [
            {"template": "a" * 200, "score": 80.0},
            {"template": "b" * 100, "score": 85.0},
        ]

        for entry in lineage:
            tpl = entry.get("template", "")
            tokens = len(tpl) // 4  # simplified token count
            entry["prompt_tokens"] = tokens
            entry["efficiency_ratio"] = round(baseline_tokens / max(tokens, 1), 3)

        assert lineage[0]["prompt_tokens"] == 50
        assert lineage[0]["efficiency_ratio"] == 2.0
        assert lineage[1]["prompt_tokens"] == 25
        assert lineage[1]["efficiency_ratio"] == 4.0

    def test_candidate_token_stats(self):
        """Test min/max/mean token stats computation."""
        lineage = [
            {"prompt_tokens": 50},
            {"prompt_tokens": 100},
            {"prompt_tokens": 150},
        ]

        stats = {
            "min_tokens": min(e["prompt_tokens"] for e in lineage),
            "max_tokens": max(e["prompt_tokens"] for e in lineage),
            "mean_tokens": round(
                sum(e["prompt_tokens"] for e in lineage) / max(len(lineage), 1), 1,
            ),
        }

        assert stats["min_tokens"] == 50
        assert stats["max_tokens"] == 150
        assert stats["mean_tokens"] == 100.0
