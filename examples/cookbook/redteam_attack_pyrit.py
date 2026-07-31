"""Red-team example — ATTACK mode paired with Microsoft PyRIT.

Authorized red teaming only. This script evolves an *attack scaffold* (a
user-turn wrapper containing a ``{goal}`` placeholder) to maximize jailbreak
success against a fixed local target, mapping where the model is weak. The
scaffolds are diversified using PyRIT prompt converters — this library never
authors attacks itself.

Prerequisites
-------------
1. ``pip install 'MutaGenAI[llm,redteam]'``  (installs PyRIT).
2. Local Ollama target:  ``ollama pull qwen2.5``.
3. Your own behavior set + your own base scaffolds (from PyRIT/HarmBench).

Run
---
    python examples/cookbook/redteam_attack_pyrit.py behaviors.txt scaffolds.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from MutaGenAI.redteam import (
    RedTeamConfig,
    RedTeamHarness,
    RedTeamMode,
    RedTeamScope,
    TargetConfig,
    TargetModel,
)
from MutaGenAI.redteam.pyrit_bridge import (
    expand_seeds_with_converters,
    load_behaviors,
    pyrit_available,
)


def main(behaviors_path: str, scaffolds_path: str) -> None:
    scope = RedTeamScope(
        operator="your-name-or-team",
        targets=["ollama:qwen2.5"],
        authorized=True,
        acknowledged_policy=True,
        notes="authorized robustness assessment",
    )

    target = TargetModel(
        TargetConfig(
            model="qwen2.5",
            system_prompt="You are a helpful, harmless, and honest assistant.",
        )
    )
    if not target.is_available():
        print("Ollama not reachable — start it and pull the model.")
        return

    behaviors = load_behaviors(source="file", path=behaviors_path, limit=25)

    # Base attack scaffolds are caller-supplied; each must contain {goal}.
    base_scaffolds = [
        s for s in Path(scaffolds_path).read_text().splitlines() if s.strip()
    ]

    # Diversify the initial population with PyRIT converters (optional).
    if pyrit_available():
        seed_templates = expand_seeds_with_converters(
            base_scaffolds,
            ["Base64Converter", "ROT13Converter", "CaesarConverter"],
        )
    else:
        print("PyRIT not installed — using base scaffolds only.")
        seed_templates = base_scaffolds

    harness = RedTeamHarness(
        scope,
        target,
        RedTeamConfig(
            mode=RedTeamMode.ATTACK,
            iterations=12,
            population_size=8,
            num_islands=3,
            use_safety_judge=True,
        ),
        behaviors=behaviors,
        seed_templates=seed_templates,
    )
    report = harness.run()

    print(report.summary())
    print(
        f"\nJailbroken {report.metadata['coverage_jailbroken']} / "
        f"{report.behaviors_tested} behaviors. "
        f"Report: {report.metadata.get('report_path')}"
    )
    print(
        "\nFeed these weaknesses into HARDEN mode to close them, then re-run "
        "to confirm the attack-success rate drops."
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
