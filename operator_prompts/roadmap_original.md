You are a product owner defining a development roadmap for TheFinalSim — an
AI-powered personal life simulator running on free local Ollama models.

TheFinalSim starts with minimal inputs (name, birth year, country) and
incrementally deepens as the user shares more real-life details. It uses a
multi-model Ollama architecture: qwen3:8b (orchestrator), qwen3:14b (narrator),
deepseek-r1:14b (decisions), hermes3:8b (personality), dolphin3:8b (realism),
with qwen3:8b as universal fallback. Ticks are event-driven OR time-driven
(user's choice). Tone is user-selectable and changeable. Stack: Python, Ollama
HTTP API, Pydantic, CLI + optional FastAPI web UI, JSON persistence.

Key paths: simulator/ (main package), simulator/models/ (Pydantic LifeState),
simulator/engine/ (tick engine, event system, decision logic),
simulator/ollama/ (model client, routing, fallback), simulator/narrative/
(prose generation, tone system), simulator/cli.py (CLI interface),
simulator/events/ (event catalogue templates), tests/ (pytest suite).

Based on the following assessment of the codebase at {local_path}:

--- ASSESSMENT ---
{assessment}
--- END ASSESSMENT ---

Create a prioritised roadmap of improvements. For each item:
1. Title (short, actionable)
2. Priority (P0 = critical, P1 = high, P2 = medium, P3 = low)
3. Effort estimate (S / M / L / XL)
4. Description of what needs to be done
5. Acceptance criteria

Prioritisation guidance for TheFinalSim:
- P0: Anything that prevents a new user from starting or playing (broken
  install, missing CLI, no narrative output, Ollama connection failures,
  crash on first tick)
- P1: Major simulation gaps (missing life domains, no decision system, no
  event variety, broken save/load, no model fallback)
- P2: Quality of life (better narrative prose, tone selection, incremental
  input flow, NPC depth, editable event catalogue, timeline forking)
- P3: Internal quality (refactors, docs, performance, test coverage)

Remember the north star: make the most compelling AI life simulator that is
fun from tick one, deeply personal over time, trivially easy for new users,
and runs entirely on free local Ollama models.

Order by priority then effort (smallest first for quick wins).
Return at most 15 items.
Format each item clearly so it can be converted into a prompt.
