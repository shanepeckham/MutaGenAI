"""Turn-key HARDEN run against a local Ollama qwen3:32b target.

Evolves a safer system prompt for qwen3:32b using the SAFE example probes,
then reports the attack-success rate (ASR) before vs. after.

Notes for qwen3:32b
-------------------
* It is a *reasoning* model, so every call is slow (~10-100s, and ~100s on the
  very first cold-start call). Keep the first run small (as configured here).
* ``timeout`` is raised to 300s to survive cold start + reasoning.
* Speed tip: qwen3 supports disabling its <think> phase. Set ``NO_THINK=1`` to
  append ``/no_think`` to each behavior, which makes calls much faster.

Run
---
    python examples/redteam/run_qwen3_harden.py
    NO_THINK=1 python examples/redteam/run_qwen3_harden.py   # faster
"""

from __future__ import annotations

import os

from MutaGenAI.redteam import (
    RedTeamConfig,
    RedTeamHarness,
    RedTeamMode,
    RedTeamScope,
    TargetConfig,
    TargetModel,
)
from MutaGenAI.redteam.pyrit_bridge import load_behaviors


def main() -> None:
    # Only test models you own or are authorized to test.
    scope = RedTeamScope(
        operator="local-test",
        targets=["ollama:qwen3:32b"],
        authorized=True,
        acknowledged_policy=True,
        notes="local hardening smoke test",
    )

    target = TargetModel(
        TargetConfig(
            model="qwen3:32b",
            system_prompt="You are a helpful assistant.",
            max_tokens=512,   # leave room for reasoning + a real answer
            timeout=300.0,    # cold start + reasoning can exceed 60s
        )
    )
    if not target.is_available():
        print("Ollama not reachable. Start it with `ollama serve`.")
        return

    behaviors = load_behaviors(
        source="file", path="examples/redteam/behaviors.txt", limit=5
    )
    if os.environ.get("NO_THINK") == "1":
        behaviors = [f"{b} /no_think" for b in behaviors]

    harness = RedTeamHarness(
        scope,
        target,
        RedTeamConfig(
            mode=RedTeamMode.HARDEN,
            iterations=3,        # keep small for a slow 32B target
            population_size=2,
            num_islands=1,
            use_safety_judge=False,  # skip extra judge calls for the smoke run
            verbose=True,
        ),
        behaviors=behaviors,
    )
    report = harness.run()

    print("\n" + report.summary())
    print(f"\nReport JSON: {report.metadata.get('report_path')}")


if __name__ == "__main__":
    main()
