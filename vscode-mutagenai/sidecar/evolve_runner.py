#!/usr/bin/env python3
"""MutaGenAI VS Code sidecar — wraps the evolution engine with JSON-line streaming.

Protocol
--------
- Receives a single JSON config object on **stdin** (one line).
- Emits JSON-line events on **stdout** as evolution progresses.
- Reads **stdin** for control messages (``{"type": "stop"}``).
- Exits when evolution completes or a stop signal is received.

Event types
-----------
``seed``        A seed candidate was evaluated.
``generation``  A generation completed.
``candidate``   A new candidate was bred and scored.
``migration``   Island migration occurred.
``done``        Evolution finished — final results attached.
``error``       A fatal error occurred.
``log``         Informational log line.
"""

from __future__ import annotations

import json
import logging
import os
import select
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Discover the MutaGenAI package by walking up from this script's location.
# This works regardless of whether the sidecar runs from the installed
# extension directory or from the source tree during development.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _ancestor in [_HERE] + list(_HERE.parents):
    if (_ancestor / "MutaGenAI" / "__init__.py").exists():
        _PROJECT_ROOT = _ancestor
        if str(_ancestor) not in sys.path:
            sys.path.insert(0, str(_ancestor))
        break

# Also honour --workspace <path> from the extension
for _i, _arg in enumerate(sys.argv):
    if _arg == "--workspace" and _i + 1 < len(sys.argv):
        _ws = sys.argv[_i + 1]
        if _ws not in sys.path:
            sys.path.insert(0, _ws)
        if _PROJECT_ROOT is None and (Path(_ws) / "MutaGenAI" / "__init__.py").exists():
            _PROJECT_ROOT = Path(_ws)
        break

# Fallback: add cwd
_CWD = os.getcwd()
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)
if _PROJECT_ROOT is None and (Path(_CWD) / "MutaGenAI" / "__init__.py").exists():
    _PROJECT_ROOT = Path(_CWD)

# ---------------------------------------------------------------------------
# Re-exec under the project venv if we're running with the wrong Python.
# This handles the case where VS Code spawns us with system Python because
# workspaceFolders was empty and no .venv was found at extension install time.
# ---------------------------------------------------------------------------
if _PROJECT_ROOT is not None and os.environ.get("_MUTAGENAI_REEXEC") != "1":
    _venv_python = _PROJECT_ROOT / ".venv" / "bin" / "python"
    if _venv_python.exists() and str(_venv_python) != sys.executable:
        os.environ["_MUTAGENAI_REEXEC"] = "1"
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

# Write debug info to a temp file for troubleshooting
_debug_path = Path(os.environ.get("TMPDIR", "/tmp")) / "mutagenai_sidecar_debug.txt"
with open(_debug_path, "w") as _df:
    _df.write(f"python={sys.executable}\n")
    _df.write(f"cwd={os.getcwd()}\n")
    _df.write(f"argv={sys.argv}\n")
    _df.write(f"sys.path={sys.path}\n")
    _df.write(f"__file__={__file__}\n")
    _df.write(f"_PROJECT_ROOT={_PROJECT_ROOT}\n")
    try:
        import MutaGenAI as _probe
        _df.write(f"MutaGenAI.__file__={_probe.__file__}\n")
    except ImportError as _ie:
        _df.write(f"MutaGenAI IMPORT FAILED: {_ie}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_stop_event = threading.Event()


def _emit(event: dict) -> None:
    """Write a JSON-line event to stdout and flush immediately."""
    line = json.dumps(event, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _listen_for_stop() -> None:
    """Background thread: reads stdin for ``{"type": "stop"}``."""
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
                if msg.get("type") == "stop":
                    _stop_event.set()
                    return
            except json.JSONDecodeError:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Patched run — supports both ground-truth and no-eval modes
# ---------------------------------------------------------------------------

def _run_evolution(cfg: dict) -> None:
    """Import MutaGenAI, build evolver from *cfg*, and run with streaming."""

    mode = cfg.get("mode", "ground_truth")
    if mode == "noeval":
        return _run_noeval_evolution(cfg)
    return _run_ground_truth_evolution(cfg)


# ---------------------------------------------------------------------------
# No-eval evolution (wizard "Run Now with Dashboard")
# ---------------------------------------------------------------------------

def _run_noeval_evolution(cfg: dict) -> None:
    """Run NoEvalPromptEvolver with streaming events."""

    from MutaGenAI.prompt_evolver import (
        LLMBackend,
        LLMClient,
        ProblemType,
        PromptCandidate,
        PromptEvolverConfig,
        _feasibility_key,
    )
    from MutaGenAI.strategies import (
        CompositeScorer,
        LLMJudge,
        NoEvalConfig,
        NoEvalPromptEvolver,
        ProxyCheck,
        ProxyMetricsScorer,
        SelfConsistencyScorer,
    )

    # Suppress noisy HTTP loggers
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    backend_map = {
        "ollama": LLMBackend.OLLAMA,
        "openai": LLMBackend.OPENAI,
        "azure_openai": LLMBackend.AZURE_OPENAI,
    }
    problem_type_map = {
        "tool_routing": ProblemType.TOOL_ROUTING,
        "classification": ProblemType.CLASSIFICATION,
    }

    backend = backend_map.get(cfg.get("backend", "ollama"), LLMBackend.OLLAMA)
    model = cfg.get("model", "llama3.2")

    noeval_config = NoEvalConfig(
        iterations=cfg.get("iterations", 5),
        population_size=cfg.get("populationSize", 6),
        num_islands=cfg.get("numIslands", 2),
        problem_type=problem_type_map.get(
            cfg.get("problemType", "tool_routing"), ProblemType.TOOL_ROUTING
        ),
        adaptive_mutations=cfg.get("adaptiveMutations", True),
        llm_mutation_rate=cfg.get("llmMutationRate", 0.3),
        refine_after_splice=cfg.get("refineAfterSplice", True),
        backend=backend,
    )

    # Build LLM client for scorers
    llm_cfg = PromptEvolverConfig(
        backend=backend,
        ollama_model=model,
        openai_model=model,
        azure_deployment=model,
    )
    client = LLMClient(llm_cfg)

    _emit({"type": "status", "message": "Checking LLM backend..."})
    if not client.is_available():
        _emit({
            "type": "log", "level": "warn",
            "message": f"LLM backend {backend.value} not available — running in mock mode.",
        })

    # Build scorer
    rubric = cfg.get("rubric", "Score the output 0-10 for quality and relevance.")
    judge = LLMJudge(rubric=rubric)
    consistency = SelfConsistencyScorer(num_samples=3)

    proxy_checks_raw = cfg.get("proxyChecks", [])
    checks = []
    for pc in proxy_checks_raw:
        name = pc.get("name", "check")
        condition = pc.get("condition", "")
        weight = pc.get("weight", 1.0)
        # Build check function from condition string
        if condition == "valid_json":
            fn = lambda x: _is_valid_json(x)
        elif condition == "length_under_1000":
            fn = lambda x: len(x) < 1000
        else:
            # Generic: check if condition string appears in output
            _cond = condition
            fn = lambda x, c=_cond: c.lower() in x.lower()
        checks.append(ProxyCheck(name=name, check_fn=fn, weight=weight))

    if not checks:
        # Default proxy checks
        checks = [
            ProxyCheck(name="valid_json", check_fn=_is_valid_json, weight=3.0),
            ProxyCheck(
                name="not_empty",
                check_fn=lambda x: len(x.strip()) > 10,
                weight=1.0,
            ),
        ]

    proxy = ProxyMetricsScorer(checks=checks)
    scorer = CompositeScorer([
        (judge, 0.35),
        (consistency, 0.30),
        (proxy, 0.35),
    ])

    _emit({"type": "status", "message": f"Scorer: {scorer.name()}"})

    task_description = cfg.get("taskDescription", "")
    test_inputs = cfg.get("testInputs", [])
    custom_seeds = cfg.get("seeds", [])
    custom_mutations = cfg.get("mutations", [])

    _emit({
        "type": "log", "level": "info",
        "message": f"Task: {task_description[:80]}...",
    })
    _emit({
        "type": "log", "level": "info",
        "message": f"Test inputs: {len(test_inputs)}, Seeds: {len(custom_seeds)}, Mutations: {len(custom_mutations)}",
    })

    if not test_inputs:
        # Generate default test inputs from task description
        test_inputs = [
            f"Process this request: {task_description[:100]}",
            "Hello, can you help me?",
            "What can you do?",
        ]
        _emit({
            "type": "log", "level": "warn",
            "message": f"No test inputs provided — using {len(test_inputs)} defaults.",
        })

    # Create evolver
    evolver = NoEvalPromptEvolver(
        task_description=task_description,
        test_inputs=test_inputs,
        scorer=scorer,
        config=noeval_config,
        seed_templates=custom_seeds or None,
        custom_mutations=custom_mutations or None,
    )

    # --- Inline the run loop to emit streaming events ---
    t0 = time.perf_counter()

    # Seed evaluation
    _emit({
        "type": "status",
        "message": f"Seeding {len(evolver._seed_templates)} templates across {noeval_config.num_islands} islands...",
    })

    islands: list[list[PromptCandidate]] = [
        [] for _ in range(noeval_config.num_islands)
    ]
    all_candidates: list[PromptCandidate] = []

    for i, template in enumerate(evolver._seed_templates):
        if _stop_event.is_set():
            _emit({"type": "stopped", "generation": 0, "reason": "user"})
            return

        isl_id = i % noeval_config.num_islands
        candidate = PromptCandidate(
            template=template,
            temperature=float(evolver._rng.uniform(*noeval_config.temperature_range)),
            top_p=float(evolver._rng.uniform(*noeval_config.top_p_range)),
            generation=0,
            island_id=isl_id,
            operation="seed",
        )
        candidate.score = evolver._evaluate(candidate)
        islands[isl_id].append(candidate)
        all_candidates.append(candidate)

        _emit({
            "type": "seed",
            "index": i,
            "total": len(evolver._seed_templates),
            "hash": candidate.hash,
            "score": round(candidate.score, 2),
            "island": isl_id,
            "operation": "seed",
            "template": candidate.template[:200],
        })

    best_overall = max(all_candidates, key=_feasibility_key)
    no_improvement_count = 0

    _emit({
        "type": "seedComplete",
        "bestScore": round(best_overall.score, 2),
        "bestHash": best_overall.hash,
        "candidateCount": len(all_candidates),
    })

    # Generation loop
    for gen in range(1, noeval_config.iterations + 1):
        if _stop_event.is_set():
            _emit({"type": "stopped", "generation": gen - 1, "reason": "user"})
            break

        gen_t0 = time.perf_counter()
        gen_candidates: list[dict] = []

        for island_id in range(noeval_config.num_islands):
            island = islands[island_id]
            if not island:
                continue

            new_candidates: list[PromptCandidate] = []
            for _ in range(noeval_config.population_size):
                if _stop_event.is_set():
                    break

                child = evolver._breed(island, gen)
                child.island_id = island_id
                child.score = evolver._evaluate(child)
                new_candidates.append(child)
                all_candidates.append(child)

                gen_candidates.append({
                    "hash": child.hash,
                    "parentHashes": child.parent_hashes,
                    "score": round(child.score, 2),
                    "island": island_id,
                    "operation": child.operation,
                    "template": child.template[:200],
                    "temperature": round(child.temperature, 4),
                    "topP": round(child.top_p, 4),
                })

            if _stop_event.is_set():
                break

            # Merge and select elite
            combined = island + new_candidates
            combined.sort(key=_feasibility_key, reverse=True)
            islands[island_id] = combined[:noeval_config.elite_size]

        if _stop_event.is_set():
            _emit({"type": "stopped", "generation": gen, "reason": "user"})
            break

        # Migration
        if gen % noeval_config.migration_interval == 0:
            evolver._migrate(islands)

        # Track best
        gen_best = max(
            (c for isl in islands for c in isl),
            key=_feasibility_key,
        )
        improved = _feasibility_key(gen_best) > _feasibility_key(best_overall)
        if improved:
            best_overall = gen_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        gen_elapsed = time.perf_counter() - gen_t0

        gen_event = {
            "type": "generation",
            "generation": gen,
            "totalGenerations": noeval_config.iterations,
            "bestScore": round(best_overall.score, 2),
            "bestHash": best_overall.hash,
            "genBestScore": round(gen_best.score, 2),
            "improved": improved,
            "candidateCount": len(all_candidates),
            "elapsed": round(gen_elapsed, 2),
            "candidates": gen_candidates,
            "migrated": gen % noeval_config.migration_interval == 0,
            "noImprovementCount": no_improvement_count,
        }
        _emit(gen_event)

        # Early stopping
        early_stop_threshold = cfg.get("earlyStopThreshold", 100.0)
        early_stop_patience = cfg.get("earlyStopPatience", 3)

        if best_overall.score >= early_stop_threshold:
            _emit({
                "type": "stopped", "generation": gen,
                "reason": "threshold",
                "message": f"Reached target score {early_stop_threshold}%",
            })
            break

        if no_improvement_count >= early_stop_patience:
            _emit({
                "type": "stopped", "generation": gen,
                "reason": "patience",
                "message": f"No improvement for {early_stop_patience} generations",
            })
            break

    # Final results
    wall_time = time.perf_counter() - t0

    lineage = [
        {
            "hash": c.hash,
            "parentHashes": c.parent_hashes,
            "operation": c.operation,
            "generation": c.generation,
            "island": c.island_id,
            "score": round(c.score, 2),
            "temperature": round(c.temperature, 4),
            "topP": round(c.top_p, 4),
            "template": c.template,
        }
        for c in sorted(all_candidates, key=lambda c: c.score, reverse=True)
    ]

    _emit({
        "type": "done",
        "bestPrompt": best_overall.template,
        "bestScore": round(best_overall.score, 2),
        "bestTemperature": round(best_overall.temperature, 4),
        "bestTopP": round(best_overall.top_p, 4),
        "wallTime": round(wall_time, 2),
        "totalCandidates": len(all_candidates),
        "iterationsRun": gen if 'gen' in dir() else 0,
        "lineage": lineage,
    })


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Ground-truth evolution (original mode)
# ---------------------------------------------------------------------------

def _run_ground_truth_evolution(cfg: dict) -> None:
    """Run PromptEvolver (ground-truth) with streaming events."""

    from MutaGenAI.prompt_evolver import (
        PromptEvolver,
        PromptEvolverConfig,
        PromptCandidate,
        Tool,
        EvalSample,
        LLMBackend,
        _feasibility_key,
        _SEED_TEMPLATES,
    )

    # ── Build config ──────────────────────────────────────────────────
    backend_map = {
        "ollama": LLMBackend.OLLAMA,
        "openai": LLMBackend.OPENAI,
        "azure_openai": LLMBackend.AZURE_OPENAI,
    }

    config = PromptEvolverConfig(
        iterations=cfg.get("iterations", 5),
        population_size=cfg.get("populationSize", 6),
        num_islands=cfg.get("numIslands", 2),
        elite_size=cfg.get("eliteSize", 4),
        mutation_rate=cfg.get("mutationRate", 0.6),
        crossover_rate=cfg.get("crossoverRate", 0.3),
        backend=backend_map.get(cfg.get("backend", "ollama"), LLMBackend.OLLAMA),
        ollama_model=cfg.get("model", "llama3.2"),
        openai_model=cfg.get("model", "gpt-4o-mini"),
        adaptive_mutations=cfg.get("adaptiveMutations", False),
        llm_mutation_rate=cfg.get("llmMutationRate", 0.0),
        refine_after_splice=cfg.get("refineAfterSplice", False),
    )

    # ── Load seed templates ───────────────────────────────────────────
    seed_template_name = cfg.get("seedTemplate", "")
    seed_templates_dir = cfg.get("seedTemplatesDir", "")
    custom_seeds: list[str] = cfg.get("seeds", [])

    if custom_seeds:
        # Patch the module-level _SEED_TEMPLATES list
        import MutaGenAI.prompt_evolver as _pe
        _pe._SEED_TEMPLATES.clear()
        _pe._SEED_TEMPLATES.extend(custom_seeds)
    elif seed_template_name:
        from MutaGenAI.seed_loader import load_seed_templates
        seeds = load_seed_templates(seed_template_name)
        import MutaGenAI.prompt_evolver as _pe
        _pe._SEED_TEMPLATES.clear()
        _pe._SEED_TEMPLATES.extend(seeds)

    # ── Build tools and eval dataset ──────────────────────────────────
    tools = [
        Tool(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters", {}),
        )
        for t in cfg.get("tools", [])
    ]

    eval_dataset = [
        EvalSample(
            query=s["query"],
            expected_tool=s.get("expected_tool", ""),
            expected_params=s.get("expected_params", {}),
        )
        for s in cfg.get("evalDataset", [])
    ]

    # ── Prompt-only mode (no tools/eval) ──────────────────────────────
    prompt_text = cfg.get("promptText", "")
    early_stop_threshold = cfg.get("earlyStopThreshold", 100.0)
    early_stop_patience = cfg.get("earlyStopPatience", 3)

    # ── Create evolver ────────────────────────────────────────────────
    evolver = PromptEvolver(
        tools=tools,
        eval_dataset=eval_dataset,
        config=config,
        seed=cfg.get("seed", 42),
        verbose=False,  # We stream events instead
    )

    # ── Patched run loop ──────────────────────────────────────────────
    # Instead of calling evolver.run() which blocks, we inline the loop
    # so we can emit streaming events and check the stop flag.

    import numpy as np

    t0 = time.perf_counter()
    rng = evolver._rng
    client = evolver._client

    _emit({"type": "status", "message": "Checking LLM availability..."})

    if not client.is_available():
        _emit({
            "type": "log",
            "level": "warn",
            "message": f"LLM backend {config.backend.value} not available — running in mock mode.",
        })

    # Initialise islands with seed templates
    import MutaGenAI.prompt_evolver as _pe
    seeds = list(_pe._SEED_TEMPLATES)
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]
    all_candidates: list[PromptCandidate] = []
    history: list[dict] = []

    _emit({"type": "status", "message": f"Seeding {len(seeds)} templates across {config.num_islands} islands..."})

    for i, template in enumerate(seeds):
        if _stop_event.is_set():
            _emit({"type": "stopped", "generation": 0, "reason": "user"})
            return

        assigned_island = i % config.num_islands
        candidate = PromptCandidate(
            template=template,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
            island_id=assigned_island,
            operation="seed",
        )
        candidate.score = evolver._evaluate_candidate(candidate)
        islands[assigned_island].append(candidate)
        all_candidates.append(candidate)

        _emit({
            "type": "seed",
            "index": i,
            "total": len(seeds),
            "hash": candidate.hash,
            "score": round(candidate.score, 2),
            "island": assigned_island,
            "operation": "seed",
            "template": candidate.template[:200],
        })

    best_overall = max(all_candidates, key=_feasibility_key)
    no_improvement_count = 0

    _emit({
        "type": "seedComplete",
        "bestScore": round(best_overall.score, 2),
        "bestHash": best_overall.hash,
        "candidateCount": len(all_candidates),
    })

    # ── Generation loop ───────────────────────────────────────────────
    for gen in range(1, config.iterations + 1):
        if _stop_event.is_set():
            _emit({"type": "stopped", "generation": gen - 1, "reason": "user"})
            break

        gen_t0 = time.perf_counter()
        gen_candidates: list[dict] = []

        for island_id in range(config.num_islands):
            island = islands[island_id]
            if not island:
                continue

            new_candidates: list[PromptCandidate] = []
            for _ in range(config.population_size):
                if _stop_event.is_set():
                    break

                child = evolver._breed(island, gen)
                child.island_id = island_id
                child.score = evolver._evaluate_candidate(child)
                new_candidates.append(child)
                all_candidates.append(child)

                gen_candidates.append({
                    "hash": child.hash,
                    "parentHashes": child.parent_hashes,
                    "score": round(child.score, 2),
                    "island": island_id,
                    "operation": child.operation,
                    "template": child.template[:200],
                    "temperature": round(child.temperature, 4),
                    "topP": round(child.top_p, 4),
                })

            if _stop_event.is_set():
                break

            # Merge and select elite
            combined = island + new_candidates
            combined.sort(key=_feasibility_key, reverse=True)
            islands[island_id] = combined[:config.elite_size]

        if _stop_event.is_set():
            _emit({"type": "stopped", "generation": gen, "reason": "user"})
            break

        # Migration every 5 generations
        did_migrate = False
        if gen % 5 == 0:
            evolver._migrate(islands)
            did_migrate = True

        # Track best
        gen_best = max(
            (c for isl in islands for c in isl),
            key=_feasibility_key,
        )
        improved = _feasibility_key(gen_best) > _feasibility_key(best_overall)
        if improved:
            best_overall = gen_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        gen_elapsed = time.perf_counter() - gen_t0

        gen_event = {
            "type": "generation",
            "generation": gen,
            "totalGenerations": config.iterations,
            "bestScore": round(best_overall.score, 2),
            "bestHash": best_overall.hash,
            "genBestScore": round(gen_best.score, 2),
            "improved": improved,
            "candidateCount": len(all_candidates),
            "elapsed": round(gen_elapsed, 2),
            "candidates": gen_candidates,
            "migrated": did_migrate,
            "noImprovementCount": no_improvement_count,
        }
        history.append(gen_event)
        _emit(gen_event)

        # ── Early stopping checks ────────────────────────────────────
        if best_overall.score >= early_stop_threshold:
            _emit({
                "type": "stopped",
                "generation": gen,
                "reason": "threshold",
                "message": f"Reached target score {early_stop_threshold}%",
            })
            break

        if no_improvement_count >= early_stop_patience:
            _emit({
                "type": "stopped",
                "generation": gen,
                "reason": "patience",
                "message": f"No improvement for {early_stop_patience} generations",
            })
            break

    # ── Final results ─────────────────────────────────────────────────
    wall_time = time.perf_counter() - t0

    # Build full lineage
    lineage = [
        {
            "hash": c.hash,
            "parentHashes": c.parent_hashes,
            "operation": c.operation,
            "generation": c.generation,
            "island": c.island_id,
            "score": round(c.score, 2),
            "temperature": round(c.temperature, 4),
            "topP": round(c.top_p, 4),
            "template": c.template,
        }
        for c in sorted(all_candidates, key=lambda c: c.score, reverse=True)
    ]

    _emit({
        "type": "done",
        "bestPrompt": best_overall.template,
        "bestScore": round(best_overall.score, 2),
        "bestTemperature": round(best_overall.temperature, 4),
        "bestTopP": round(best_overall.top_p, 4),
        "wallTime": round(wall_time, 2),
        "totalCandidates": len(all_candidates),
        "iterationsRun": gen if 'gen' in dir() else 0,
        "lineage": lineage,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Read config from stdin, run evolution, stream events to stdout."""

    try:
        _emit({
            "type": "log", "level": "info",
            "message": f"Sidecar started: cwd={os.getcwd()} argv={sys.argv[1:]} sys.path[:3]={sys.path[:3]}",
        })

        # Read config from first stdin line BEFORE starting the stop
        # listener — otherwise the listener thread races and may consume
        # the config line.
        raw = ""
        while not raw:
            if sys.stdin.readable():
                raw = sys.stdin.readline().strip()
            if not raw:
                time.sleep(0.05)

        cfg = json.loads(raw)

        # Now start the stop-listener thread (it will read subsequent
        # stdin lines for {"type": "stop"} messages).
        listener = threading.Thread(target=_listen_for_stop, daemon=True)
        listener.start()

        _emit({"type": "started", "config": cfg})
        _run_evolution(cfg)

    except json.JSONDecodeError as e:
        _emit({"type": "error", "message": f"Invalid config JSON: {e}"})
        sys.exit(1)
    except Exception as e:
        _emit({
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
