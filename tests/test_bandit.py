"""Tests for MutaGenAI.bandit — multi-armed operator selection."""
from __future__ import annotations

import numpy as np
import pytest

from MutaGenAI.bandit import OperatorBandit, UCB, THOMPSON


class TestConstruction:
    def test_requires_arms(self):
        with pytest.raises(ValueError):
            OperatorBandit([])

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            OperatorBandit(["a"], method="banana")

    def test_dedupes_arms_preserving_order(self):
        b = OperatorBandit(["mutation", "crossover", "mutation"])
        assert b.arms == ["mutation", "crossover"]

    def test_initial_state(self):
        b = OperatorBandit(["a", "b"])
        assert b.total_pulls == 0
        assert b.best_arm() is None
        assert b.counts == {"a": 0, "b": 0}


class TestUCB:
    def test_tries_each_arm_before_repeating(self):
        b = OperatorBandit(["a", "b", "c"], method=UCB)
        first = b.select()
        b.update(first, 0.0)
        second = b.select()
        b.update(second, 0.0)
        third = b.select()
        b.update(third, 0.0)
        assert {first, second, third} == {"a", "b", "c"}

    def test_exploits_high_reward_arm(self):
        b = OperatorBandit(["good", "bad"], method=UCB, c=0.1)
        # Seed both arms.
        b.update("good", 1.0)
        b.update("bad", 0.0)
        # With low exploration, the high-mean arm should dominate.
        picks = [b.select() for _ in range(20)]
        # Re-pull according to selection to keep means stable.
        for p in picks:
            b.update(p, 1.0 if p == "good" else 0.0)
        assert picks.count("good") > picks.count("bad")

    def test_allowed_restricts_arms(self):
        b = OperatorBandit(["a", "b", "c"], method=UCB)
        for _ in range(10):
            arm = b.select(allowed={"a", "b"})
            assert arm in {"a", "b"}
            b.update(arm, 0.5)

    def test_allowed_empty_falls_back_to_all(self):
        b = OperatorBandit(["a", "b"], method=UCB)
        # Unknown allowed names → fall back to full arm set, never crash.
        arm = b.select(allowed={"zzz"})
        assert arm in {"a", "b"}


class TestThompson:
    def test_deterministic_with_seeded_rng(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        b1 = OperatorBandit(["a", "b", "c"], method=THOMPSON, rng=rng1)
        b2 = OperatorBandit(["a", "b", "c"], method=THOMPSON, rng=rng2)
        seq1, seq2 = [], []
        for _ in range(15):
            s1 = b1.select()
            b1.update(s1, 0.5)
            seq1.append(s1)
            s2 = b2.select()
            b2.update(s2, 0.5)
            seq2.append(s2)
        assert seq1 == seq2

    def test_favours_better_arm_over_time(self):
        rng = np.random.default_rng(7)
        b = OperatorBandit(["good", "bad"], method=THOMPSON, rng=rng)
        for _ in range(80):
            arm = b.select()
            b.update(arm, 1.0 if arm == "good" else 0.0)
        assert b.counts["good"] > b.counts["bad"]
        assert b.best_arm() == "good"


class TestUpdate:
    def test_reward_clamped_to_unit_interval(self):
        b = OperatorBandit(["a"])
        b.update("a", 5.0)
        b.update("a", -3.0)
        assert b.means["a"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2

    def test_unknown_arm_raises(self):
        b = OperatorBandit(["a"])
        with pytest.raises(KeyError):
            b.update("zzz", 0.5)

    def test_mean_tracks_running_average(self):
        b = OperatorBandit(["a"])
        b.update("a", 1.0)
        b.update("a", 0.0)
        b.update("a", 1.0)
        assert b.means["a"] == pytest.approx(2 / 3)
        assert b.total_pulls == 3

    def test_stats_shape(self):
        b = OperatorBandit(["a", "b"])
        b.update("a", 1.0)
        stats = b.stats()
        assert set(stats) == {"a", "b"}
        assert stats["a"]["count"] == 1
        assert stats["a"]["mean_reward"] == 1.0
        assert stats["a"]["share"] == 1.0
        assert stats["b"]["share"] == 0.0
