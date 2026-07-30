"""Red-team run report — metrics, coverage, and safe persistence.

Reports capture *measurements* (attack-success rate, per-behavior verdicts,
before/after hardening deltas) and the best evolved prompt.  By default the
raw model outputs and successful attack strings are **not** embedded in the
report body; they are sensitive and, when persistence is enabled, are written
only under the git-ignored ``redteam_runs/`` directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class BehaviorResult:
    """Per-behavior verdict.  Stores flags/scores, not raw harmful output."""

    behavior: str
    jailbroken: bool
    refused: bool
    score: float


@dataclass
class RedTeamReport:
    """Outcome of a red-team run.

    Attributes
    ----------
    mode : str
        ``"harden"`` or ``"attack"``.
    target : str
        Target identifier (``backend:model``).
    operator : str
        Who ran the assessment (from the scope).
    timestamp : str
        UTC ISO-8601 run time.
    behaviors_tested : int
        Number of behaviors evaluated.
    asr : float
        Attack-success rate in ``[0, 1]`` for the final/best configuration.
    asr_before, asr_after : float or None
        Hardening mode only: baseline vs. hardened attack-success rate.
    best_prompt : str
        The evolved artifact — a hardened system prompt (harden mode) or the
        best attack scaffold (attack mode).
    history : list
        Per-generation ``(gen, best_score)`` trace from evolution.
    per_behavior : list[BehaviorResult]
        Verdicts per behavior for the best configuration.
    metadata : dict
        Free-form run metadata (config, seeds count, wall time, etc.).
    """

    mode: str
    target: str
    operator: str
    timestamp: str
    behaviors_tested: int
    asr: float
    asr_before: Optional[float] = None
    asr_after: Optional[float] = None
    best_prompt: str = ""
    history: list = field(default_factory=list)
    per_behavior: list[BehaviorResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def asr_reduction(self) -> Optional[float]:
        """Absolute reduction in attack-success rate from hardening."""
        if self.asr_before is None or self.asr_after is None:
            return None
        return self.asr_before - self.asr_after

    def summary(self) -> str:
        lines = [
            "Red-Team Report",
            "=" * 50,
            f"  Mode:            {self.mode}",
            f"  Target:          {self.target}",
            f"  Operator:        {self.operator}",
            f"  Timestamp:       {self.timestamp}",
            f"  Behaviors:       {self.behaviors_tested}",
        ]
        if self.mode == "harden":
            before = "n/a" if self.asr_before is None else f"{self.asr_before:.1%}"
            after = "n/a" if self.asr_after is None else f"{self.asr_after:.1%}"
            red = self.asr_reduction
            red_s = "n/a" if red is None else f"{red:+.1%}"
            lines += [
                f"  ASR before:      {before}",
                f"  ASR after:       {after}",
                f"  ASR reduction:   {red_s}",
            ]
        else:
            lines.append(f"  Attack success:  {self.asr:.1%}")
        jb = sum(1 for r in self.per_behavior if r.jailbroken)
        lines.append(f"  Jailbroken:      {jb}/{len(self.per_behavior)}")
        lines += ["", "Best evolved prompt/scaffold:", "-" * 50, self.best_prompt]
        return "\n".join(lines)

    def to_dict(self, include_behaviors: bool = True) -> dict:
        data = asdict(self)
        if not include_behaviors:
            data["per_behavior"] = []
        return data

    def to_json(
        self,
        output_dir: str = "redteam_runs",
        *,
        include_behaviors: bool = True,
    ) -> Path:
        """Write the report as JSON under *output_dir* (git-ignored)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "").replace("-", "")
        safe_target = self.target.replace(":", "_").replace("/", "_")
        path = out / f"redteam_{self.mode}_{safe_target}_{stamp}.json"
        path.write_text(
            json.dumps(self.to_dict(include_behaviors), indent=2),
            encoding="utf-8",
        )
        return path
