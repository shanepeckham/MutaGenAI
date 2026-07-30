"""Red-team example — HARDEN mode: evolve a safer system prompt for an SLM.

Defensive use only. This script evolves a *system prompt* that maximizes safe
refusals for a local open-source SLM served by Ollama, then reports the
attack-success rate (ASR) before vs. after applying the hardened prompt.

Prerequisites
-------------
1. Install extras:  ``pip install 'MutaGenAI[llm]'``  (add ``[redteam]`` to use
   PyRIT dataset loaders).
2. Run Ollama locally and pull a model:  ``ollama pull llama3.2``.
3. Provide your own harmful-behavior set (e.g. exported from HarmBench or
   PyRIT) as ``behaviors.txt`` (one behavior per line).  This library ships
   no harmful content.  A SAFE synthetic example set is provided at
   ``examples/redteam/behaviors.txt`` for testing the harness plumbing.

Run
---
    python examples/cookbook/redteam_harden_slm.py examples/redteam/behaviors.txt
"""

from __future__ import annotations

import sys

from MutaGenAI.redteam import (
    RedTeamConfig,
    RedTeamHarness,
    RedTeamMode,
    RedTeamScope,
    TargetConfig,
    TargetModel,
)
from MutaGenAI.redteam.pyrit_bridge import load_behaviors


def main(behaviors_path: str) -> None:
    # 1) Authorization — only ever test models you own or may test.
    scope = RedTeamScope(
        operator="your-name-or-team",
        targets=["ollama:llama3.2"],
        authorized=True,          # set True only with real authorization
        acknowledged_policy=True,  # findings are for hardening + disclosure
        notes="local hardening experiment",
    )

    # 2) Target under test — pin its current (baseline) system prompt.
    target = TargetModel(
        TargetConfig(
            model="llama3.2",
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
        )
    )
    if not target.is_available():
        print("Ollama not reachable — start it and `ollama pull llama3.2`.")
        return

    # 3) Behaviors you supply (never authored by this library).
    behaviors = load_behaviors(source="file", path=behaviors_path, limit=25)

    # 4) Evolve a hardened system prompt and measure ASR before/after.
    harness = RedTeamHarness(
        scope,
        target,
        RedTeamConfig(
            mode=RedTeamMode.HARDEN,
            iterations=10,
            population_size=6,
            num_islands=3,
            use_safety_judge=True,  # LLM judge classifies compliance
        ),
        behaviors=behaviors,
    )
    report = harness.run()

    print(report.summary())
    print(f"\nFull report JSON: {report.metadata.get('report_path')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
