"""Multi-armed bandit operator selection for evolutionary prompt search.

Instead of applying genetic operators (mutation, crossover, LLM rewrite)
at fixed probabilities, the evolver can treat *which operator to apply*
as a multi-armed bandit problem.  Each breeding event pulls an arm (an
operator), the resulting child is evaluated, and the fitness improvement
becomes the reward.  Over time the bandit concentrates effort on the
operators that actually help the current population — adapting the
operator mix per run and per problem.

Two classic policies are provided:

- **UCB1** — deterministic; balances exploitation (high mean reward) with
  exploration (uncertainty) via an upper-confidence bound.
- **Thompson sampling** — Bayesian; draws a sample from each arm's Beta
  posterior and pulls the arm with the highest draw.

This module has **no internal MutaGenAI dependencies** so it can be reused
or tested in isolation.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

__all__ = ["OperatorBandit"]

#: Supported bandit policies.
UCB = "ucb"
THOMPSON = "thompson"


class OperatorBandit:
    """A multi-armed bandit over a fixed set of operator names (arms).

    Parameters
    ----------
    arms :
        The operator names the bandit chooses between (e.g.
        ``["mutation", "crossover", "llm_mutation"]``).  Must be non-empty.
    method :
        ``"ucb"`` (default) for UCB1 or ``"thompson"`` for Thompson
        sampling over Beta posteriors.
    c :
        UCB exploration constant.  Higher values explore more.  Ignored by
        Thompson sampling.  Default ``1.4`` (≈ √2).
    rng :
        NumPy random generator used by Thompson sampling.  Provide a
        dedicated generator (separate from the evolver's main stream) to
        keep evolution deterministic.  Defaults to a fresh generator.

    Notes
    -----
    Rewards are expected in ``[0, 1]`` and are clamped to that range in
    :meth:`update`.  Both policies start by trying every arm once (UCB
    gives untried arms infinite priority; Thompson's uniform Beta(1, 1)
    prior yields broad initial exploration).
    """

    def __init__(
        self,
        arms: Iterable[str],
        method: str = UCB,
        c: float = 1.4,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.arms: list[str] = list(dict.fromkeys(arms))  # de-dup, keep order
        if not self.arms:
            raise ValueError("OperatorBandit requires at least one arm")
        if method not in (UCB, THOMPSON):
            raise ValueError(
                f"Unknown bandit method {method!r}; use {UCB!r} or {THOMPSON!r}"
            )
        self.method = method
        self.c = float(c)
        self._rng = rng if rng is not None else np.random.default_rng()

        self.counts: dict[str, int] = {a: 0 for a in self.arms}
        self.means: dict[str, float] = {a: 0.0 for a in self.arms}
        self.total_reward: dict[str, float] = {a: 0.0 for a in self.arms}
        # Beta posterior parameters for Thompson sampling.
        self._alpha: dict[str, float] = {a: 1.0 for a in self.arms}
        self._beta: dict[str, float] = {a: 1.0 for a in self.arms}
        self._total_pulls: int = 0

    # -- selection -----------------------------------------------------------

    def select(self, allowed: Optional[Iterable[str]] = None) -> str:
        """Choose an arm to pull, optionally restricted to ``allowed``.

        ``allowed`` lets the caller hide arms that are not applicable this
        round (e.g. crossover when an island has a single member).  Unknown
        names in ``allowed`` are ignored.  Returns the selected arm name.
        """
        candidates = self._allowed_arms(allowed)
        if self.method == THOMPSON:
            return self._select_thompson(candidates)
        return self._select_ucb(candidates)

    def _allowed_arms(self, allowed: Optional[Iterable[str]]) -> list[str]:
        if allowed is None:
            return list(self.arms)
        allowed_set = set(allowed)
        filtered = [a for a in self.arms if a in allowed_set]
        return filtered or list(self.arms)

    def _select_ucb(self, candidates: list[str]) -> str:
        # Always try an untried arm first (infinite UCB).
        untried = [a for a in candidates if self.counts[a] == 0]
        if untried:
            return untried[0]
        total = sum(self.counts[a] for a in candidates)
        log_total = math.log(total) if total > 0 else 0.0
        best_arm = candidates[0]
        best_score = -math.inf
        for arm in candidates:
            n = self.counts[arm]
            bonus = self.c * math.sqrt(log_total / n) if n > 0 else math.inf
            score = self.means[arm] + bonus
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def _select_thompson(self, candidates: list[str]) -> str:
        best_arm = candidates[0]
        best_sample = -math.inf
        for arm in candidates:
            sample = float(self._rng.beta(self._alpha[arm], self._beta[arm]))
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        return best_arm

    # -- update --------------------------------------------------------------

    def update(self, arm: str, reward: float) -> None:
        """Record ``reward`` (clamped to ``[0, 1]``) for ``arm``."""
        if arm not in self.counts:
            raise KeyError(f"Unknown arm {arm!r}")
        r = float(min(1.0, max(0.0, reward)))
        self.counts[arm] += 1
        self.total_reward[arm] += r
        self.means[arm] = self.total_reward[arm] / self.counts[arm]
        self._alpha[arm] += r
        self._beta[arm] += 1.0 - r
        self._total_pulls += 1

    # -- introspection -------------------------------------------------------

    @property
    def total_pulls(self) -> int:
        """Total number of recorded rewards across all arms."""
        return self._total_pulls

    def best_arm(self) -> Optional[str]:
        """Return the arm with the highest mean reward, or ``None`` if no
        arm has been pulled yet."""
        pulled = [a for a in self.arms if self.counts[a] > 0]
        if not pulled:
            return None
        return max(pulled, key=lambda a: self.means[a])

    def stats(self) -> dict[str, dict[str, float]]:
        """Return per-arm ``{count, mean_reward, share}`` statistics."""
        total = self._total_pulls or 1
        return {
            arm: {
                "count": self.counts[arm],
                "mean_reward": round(self.means[arm], 4),
                "share": round(self.counts[arm] / total, 4),
            }
            for arm in self.arms
        }
