#!/usr/bin/env python3
"""Webview wizard sidecar — generates an evolution script from JSON state.

Called by the VS Code extension's WizardPanel with a JSON file path
as the sole argument.  Prints the generated script path to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the workspace root (cwd) is importable
sys.path.insert(0, str(Path.cwd()))

from MutaGenAI.wizard import WizardState, Penalty, _generate_script  # noqa: E402


def _map_state(raw: dict) -> WizardState:
    """Convert the webview JSON blob into a WizardState dataclass."""
    state = WizardState()
    state.problem_type = raw.get("problemType", "tool_routing")
    state.task_description = raw.get("taskDescription", "")
    state.has_ground_truth = raw.get("groundTruth", "no")
    state.eval_file = raw.get("evalFile", "")

    # Test inputs
    state.test_input_file = raw.get("testInputFile", "")
    state.test_inputs = raw.get("testInputs", [])

    # Strategies
    state.strategies = raw.get("strategies", ["composite"])
    state.llm_judge_rubric = raw.get("rubric", "")

    # Proxy checks
    state.proxy_checks = raw.get("proxyChecks", [])

    # Penalties — the webview may or may not include these
    for p in raw.get("penalties", []):
        state.penalties.append(Penalty(
            name=p.get("name", ""),
            description=p.get("description", ""),
            condition=p.get("condition", ""),
            threshold=p.get("threshold"),
            pattern=p.get("pattern"),
            weight=p.get("weight", -2.0),
        ))

    # Domain mutations
    state.domain_mutations = raw.get("mutations", [])
    state.has_domain_mutations = bool(state.domain_mutations)

    # Human eval
    state.human_eval = raw.get("humanEval", "final")

    # Seeds
    state.seed_templates = raw.get("seeds", [])
    state.has_seed_templates = bool(state.seed_templates)

    # Backend / model
    backend_raw = raw.get("backend", "ollama")
    # Normalise azure_openai → azure (wizard internal convention)
    if backend_raw == "azure_openai":
        backend_raw = "azure"
    state.backend = backend_raw
    state.model = raw.get("model", "llama3.2")

    # Config
    preset = raw.get("preset", "standard")
    state.config_preset = preset
    if preset == "standard":
        state.iterations = 5
        state.population_size = 6
        state.num_islands = 2
    elif preset == "deep":
        state.iterations = 10
        state.population_size = 8
        state.num_islands = 3
    else:
        state.iterations = int(raw.get("iterations", 5))
        state.population_size = int(raw.get("populationSize", 6))
        state.num_islands = int(raw.get("numIslands", 2))

    return state


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: wizard_gen.py <state.json>", file=sys.stderr)
        sys.exit(1)

    state_path = Path(sys.argv[1])
    if not state_path.exists():
        print(f"File not found: {state_path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(state_path.read_text())
    state = _map_state(raw)
    script = _generate_script(state)

    # Write to the working directory (workspace root)
    out = Path.cwd() / "evolve_prompt_wizard.py"
    out.write_text(script)
    # Print the path so the extension can open it
    print(str(out))


if __name__ == "__main__":
    main()
