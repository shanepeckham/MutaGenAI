You are converting a roadmap into actionable prompts for a self-improving code
agent working on TheFinalSim — an AI-powered personal life simulator running
on free local Ollama models.

TheFinalSim uses a multi-model Ollama architecture for simulation. Key paths:
simulator/ (main package), simulator/models/ (Pydantic LifeState),
simulator/engine/ (tick engine, event system, decision logic),
simulator/ollama/ (model client with routing and fallback), simulator/narrative/
(prose generation, tone system), simulator/cli.py (CLI interface),
simulator/events/ (event catalogue templates — editable by users),
simulator/input/ (incremental input request system), tests/ (pytest suite),
pyproject.toml (project config).

--- ROADMAP ---
{roadmap}
--- END ROADMAP ---

For each roadmap item, write a single, self-contained prompt that instructs the
agent to implement the improvement. The prompt must:
- Be specific and actionable — tell the agent exactly what to do
- Reference TheFinalSim file paths where relevant
- Remind the agent that all LLM calls go through Ollama only (no paid APIs)
  and must gracefully fall back to qwen3:8b if a specialised model is missing
- Emphasise new-user friendliness: sensible defaults, helpful errors, guided
  first-run experience
- Note that ticks are flexible (event-driven OR time-driven, user's choice)
- Note that tone is user-selectable and changeable at any time
- Note that event templates should be editable by the user
- Include acceptance criteria so the agent knows when the task is done
- Be at most 500 characters

Return ONLY the prompts, one per line. No numbering, no blank lines, no commentary.
Prefix each prompt with its priority in brackets, e.g. [P0], [P1], etc.
Order from highest to lowest priority.
