"""Interactive wizard for bootstrapping prompt evolution projects.

Guides users through a step-by-step questionnaire and generates a
ready-to-run Python script tailored to their agent, evaluation data,
and scoring strategy.

Usage::

    prompture init                    # interactive walkthrough
    prompture init --output my_evo.py # specify output file
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Rich / fallback helpers ──────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table

    _console = Console()
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]


def _print(msg: str = "") -> None:
    if _HAS_RICH:
        _console.print(msg)
    else:
        print(msg)


def _ask(prompt: str, *, default: str = "", choices: list[str] | None = None) -> str:
    if _HAS_RICH:
        return Prompt.ask(prompt, default=default, choices=choices)
    suffix = f" [{default}]" if default else ""
    if choices:
        suffix += f" ({'/'.join(choices)})"
    return input(f"{prompt}{suffix}: ").strip() or default


def _ask_int(prompt: str, *, default: int = 0) -> int:
    if _HAS_RICH:
        return IntPrompt.ask(prompt, default=default)
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a number.")


def _confirm(prompt: str, *, default: bool = True) -> bool:
    if _HAS_RICH:
        return Confirm.ask(prompt, default=default)
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{prompt}{suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _banner() -> None:
    text = (
        "[bold cyan]🧬 Prompture — Prompt Evolution Wizard[/bold cyan]\n"
        "This wizard will generate a ready-to-run script that\n"
        "evolves your agent's system prompt for maximum performance."
    )
    if _HAS_RICH:
        _console.print(Panel(text, border_style="cyan", padding=(1, 2)))
    else:
        print("\n" + "=" * 56)
        print("  🧬 Prompture — Prompt Evolution Wizard")
        print("  Generates a prompt-evolution script for your agent.")
        print("=" * 56)


# ── Wizard state ─────────────────────────────────────────────────────────

@dataclass
class WizardState:
    """Captures every answer from the interactive walkthrough."""

    problem_type: str = "tool_routing"  # tool_routing / classification
    task_description: str = ""
    has_ground_truth: str = "no"  # yes / no / partial

    # Ground-truth eval data
    eval_file: str = ""  # path to JSON/CSV
    eval_examples: list[dict[str, str]] = field(default_factory=list)

    # Test inputs (unlabelled)
    test_inputs: list[str] = field(default_factory=list)
    test_input_file: str = ""

    # Scoring
    strategies: list[str] = field(default_factory=list)  # selected scorers
    llm_judge_rubric: str = ""
    proxy_checks: list[str] = field(default_factory=list)

    # Domain mutations
    has_domain_mutations: bool = False
    domain_mutations: list[str] = field(default_factory=list)

    # Human evaluation
    human_eval: str = "no"  # always / tiebreaker / no

    # Seed templates
    has_seed_templates: bool = False
    seed_templates: list[str] = field(default_factory=list)

    # Backend
    backend: str = "ollama"  # ollama / openai / azure_openai
    model: str = "llama3.2"

    # Config
    config_preset: str = "standard"  # standard / deep / custom
    iterations: int = 3
    population_size: int = 4
    num_islands: int = 2


# ── Individual wizard steps ──────────────────────────────────────────────

def _step_problem_type(state: WizardState) -> None:
    _print("\n[bold yellow]Step 1 of 10 — Problem Type[/bold yellow]" if _HAS_RICH
           else "\nStep 1 of 10 — Problem Type")
    _print("What kind of task will the evolved prompt perform?\n"
           "This determines which mutation snippets guide evolution.\n")

    if _HAS_RICH:
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("[bold green]tool_routing[/bold green]",
                     "Map user queries to tool/function calls (JSON output)")
        tbl.add_row("[bold blue]classification[/bold blue]",
                     "Classify input text into one of several categories")
        _console.print(tbl)
    else:
        print("  tool_routing    — Map queries to tool calls")
        print("  classification  — Classify text into categories")

    state.problem_type = _ask("\n  Problem type",
                               default="tool_routing",
                               choices=["tool_routing", "classification"])


def _step_task(state: WizardState) -> None:
    _print("\n[bold yellow]Step 2 of 10 — Task Description[/bold yellow]" if _HAS_RICH
           else "\nStep 2 of 10 — Task Description")
    _print("Describe what your agent does. Be specific — this drives mutation\n"
           "generation and LLM-as-Judge rubrics.\n")
    _print("[dim]Example: 'You are an API-calling assistant that maps natural-language\n"
           "queries to tool calls in the format [ToolName(param=value)].'[/dim]" if _HAS_RICH
           else "Example: 'You are an API-calling assistant that maps natural-language\n"
                "queries to tool calls in the format [ToolName(param=value)].'")
    state.task_description = _ask("\n  Task description")
    while not state.task_description.strip():
        _print("  [red]Task description cannot be empty.[/red]" if _HAS_RICH
               else "  Task description cannot be empty.")
        state.task_description = _ask("  Task description")


def _step_ground_truth(state: WizardState) -> None:
    _print("\n[bold yellow]Step 3 of 10 — Ground Truth[/bold yellow]" if _HAS_RICH
           else "\nStep 3 of 10 — Ground Truth")
    _print("Do you have labelled evaluation data — input/output pairs\n"
           "where you know the correct answer?\n")

    if _HAS_RICH:
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("[bold green]yes[/bold green]",
                     "I have a dataset of inputs with expected outputs")
        tbl.add_row("[bold yellow]partial[/bold yellow]",
                     "I have some labels but not a complete dataset")
        tbl.add_row("[bold red]no[/bold red]",
                     "No labels — I need label-free evaluation strategies")
        _console.print(tbl)

    state.has_ground_truth = _ask("\n  Ground truth availability",
                                  default="no",
                                  choices=["yes", "partial", "no"])

    if state.has_ground_truth in ("yes", "partial"):
        _print("\n  Point to a JSON file with eval data, or enter examples interactively.")
        _print("  JSON format: [{\"input\": \"...\", \"expected\": \"...\"}]")
        choice = _ask("  Load from file or enter interactively?",
                       default="interactive",
                       choices=["file", "interactive"])
        if choice == "file":
            state.eval_file = _ask("  Path to JSON eval file")
        else:
            _print("  Enter input/expected pairs. Type 'done' when finished.\n")
            while True:
                inp = _ask("  Input (or 'done')")
                if inp.lower() == "done":
                    break
                exp = _ask("  Expected output")
                state.eval_examples.append({"input": inp, "expected": exp})
            _print(f"  Collected {len(state.eval_examples)} examples.")


def _step_test_inputs(state: WizardState) -> None:
    _print("\n[bold yellow]Step 4 of 10 — Test Inputs[/bold yellow]" if _HAS_RICH
           else "\nStep 4 of 10 — Test Inputs")
    _print("Provide unlabelled test inputs your agent should handle.\n"
           "These are used during evolution to generate outputs for scoring.\n")

    choice = _ask("  Load from file or enter interactively?",
                   default="interactive",
                   choices=["file", "interactive"])
    if choice == "file":
        state.test_input_file = _ask("  Path to text file (one input per line)")
    else:
        _print("  Enter test inputs your agent would receive. Type 'done' when finished.\n")
        while True:
            inp = _ask("  Test input (or 'done')")
            if inp.lower() == "done":
                break
            state.test_inputs.append(inp)
        _print(f"  Collected {len(state.test_inputs)} test inputs.")

    if (not state.test_inputs and not state.test_input_file
            and state.has_ground_truth == "no"):
        _print("  [yellow]⚠ No test inputs provided. The generated script will include\n"
               "  placeholder inputs you should replace.[/yellow]" if _HAS_RICH
               else "  ⚠ No test inputs. The script will include placeholders.")


def _step_scoring(state: WizardState) -> None:
    _print("\n[bold yellow]Step 5 of 10 — Scoring Strategy[/bold yellow]" if _HAS_RICH
           else "\nStep 5 of 10 — Scoring Strategy")

    if state.has_ground_truth == "yes":
        _print("  Since you have full ground-truth data, we'll use [green]automated "
               "scoring[/green] against your labels." if _HAS_RICH
               else "  With full ground truth, we'll use automated scoring.")
        state.strategies = ["ground_truth"]
        want_extra = _confirm("  Also add a no-eval strategy as comparison?",
                               default=False)
        if want_extra:
            _show_strategy_picker(state)
        return

    _print("  Without full ground truth, select one or more scoring strategies.\n"
           "  EvoSim will combine them to approximate a fitness signal.\n")
    _show_strategy_picker(state)


def _show_strategy_picker(state: WizardState) -> None:
    strategies = [
        ("llm_judge", "LLM-as-Judge",
         "A second LLM call rates each output against a rubric (0-10)"),
        ("self_consistency", "Self-Consistency",
         "Run the same input multiple times — consistent outputs score higher"),
        ("proxy_metrics", "Proxy Metrics",
         "Structural checks: valid JSON, correct format, length, keywords"),
        ("tool_success", "Tool-Use Success",
         "Actually execute tool calls and score by HTTP status / return code"),
        ("preference", "Preference Pairs",
         "Compare against hand-crafted good/bad output examples"),
        ("human", "Human-as-Judge",
         "You rate outputs interactively during evolution"),
        ("composite", "Composite (recommended)",
         "Weighted mix of the above — best results in benchmarks"),
    ]

    if _HAS_RICH:
        tbl = Table(show_header=True, box=None, padding=(0, 2))
        tbl.add_column("#", style="bold", width=3)
        tbl.add_column("Strategy", width=22)
        tbl.add_column("Description")
        for i, (_, name, desc) in enumerate(strategies, 1):
            tbl.add_row(str(i), name, desc)
        _console.print(tbl)
    else:
        for i, (_, name, desc) in enumerate(strategies, 1):
            print(f"  {i}. {name} — {desc}")

    _print("\n  Enter numbers separated by commas (e.g. '1,3,6' or '7' for composite).")
    raw = _ask("  Strategies", default="7")
    indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
    for idx in indices:
        if 0 <= idx < len(strategies):
            state.strategies.append(strategies[idx][0])

    if not state.strategies:
        state.strategies = ["composite"]
        _print("  Defaulting to [green]Composite[/green]." if _HAS_RICH
               else "  Defaulting to Composite.")

    # LLM-as-Judge rubric
    if "llm_judge" in state.strategies or "composite" in state.strategies:
        _print("\n  LLM-as-Judge needs a rubric. Enter custom or use auto-generated.")
        custom = _confirm("  Provide a custom rubric?", default=False)
        if custom:
            state.llm_judge_rubric = _ask("  Rubric (criteria for a good output)")
        # else: auto-generated from task_description

    # Proxy checks
    if "proxy_metrics" in state.strategies or "composite" in state.strategies:
        _print("\n  Proxy Metrics — what structural checks apply?")
        checks = [
            ("valid_json", "Output is valid JSON"),
            ("has_function_name", "Contains a function/tool name"),
            ("bracket_format", "Uses bracket call format [Fn(args)]"),
            ("max_length", "Output under 500 chars"),
            ("no_explanation", "No prose explanation, just the call"),
        ]
        if _HAS_RICH:
            for i, (_, desc) in enumerate(checks, 1):
                _print(f"  {i}. {desc}")
        else:
            for i, (_, desc) in enumerate(checks, 1):
                print(f"  {i}. {desc}")
        raw = _ask("  Checks (e.g. '1,2,3' or 'all')", default="all")
        if raw.lower() == "all":
            state.proxy_checks = [c[0] for c in checks]
        else:
            idxs = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
            state.proxy_checks = [checks[i][0] for i in idxs if 0 <= i < len(checks)]


def _step_mutations(state: WizardState) -> None:
    _print("\n[bold yellow]Step 6 of 10 — Domain Mutations[/bold yellow]" if _HAS_RICH
           else "\nStep 6 of 10 — Domain Mutations")
    _print("Mutations are rewrite instructions applied to prompts each generation.\n"
           "Domain-specific mutations produce better results than generic ones.\n")
    _print("[dim]Examples:[/dim]" if _HAS_RICH else "Examples:")
    _print("  • 'Add chain-of-thought reasoning before the tool call'")
    _print("  • 'Enforce strict JSON output format'")
    _print("  • 'Add error recovery instructions for malformed queries'\n")

    state.has_domain_mutations = _confirm("  Do you have domain-specific mutations?",
                                           default=False)
    if state.has_domain_mutations:
        _print("  Enter mutations one per line. Type 'done' when finished.\n")
        while True:
            m = _ask("  Mutation (or 'done')")
            if m.lower() == "done":
                break
            state.domain_mutations.append(m)
        _print(f"  Collected {len(state.domain_mutations)} mutations.")
    else:
        _print("  No problem — the script will auto-generate mutations from your task\n"
               "  description using the LLM at runtime.")


def _step_human_eval(state: WizardState) -> None:
    _print("\n[bold yellow]Step 7 of 10 — Human Evaluation[/bold yellow]" if _HAS_RICH
           else "\nStep 7 of 10 — Human Evaluation")
    _print("Human-in-the-loop evaluation lets you rate outputs during evolution.\n"
           "This is the gold standard but requires your time each generation.\n")

    if _HAS_RICH:
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("[bold green]always[/bold green]",
                     "Human rates every generation (highest quality, most effort)")
        tbl.add_row("[bold yellow]final[/bold yellow]",
                     "Human picks the winner from the top-K after evolution finishes")
        tbl.add_row("[bold red]no[/bold red]",
                     "Fully automated — no human involvement")
        _console.print(tbl)
    else:
        print("  always  — Human rates every generation")
        print("  final   — Human picks winner from top-K after evolution")
        print("  no      — Fully automated")

    state.human_eval = _ask("\n  Human evaluation mode",
                             default="final",
                             choices=["always", "final", "no"])


def _step_seeds(state: WizardState) -> None:
    _print("\n[bold yellow]Step 8 of 10 — Seed Templates[/bold yellow]" if _HAS_RICH
           else "\nStep 8 of 10 — Seed Templates")
    _print("Seed templates are the starting prompts for evolution.\n"
           "Using your existing best prompt gives evolution a head start.\n")

    state.has_seed_templates = _confirm(
        "  Do you have existing prompts to seed with?", default=False)
    if state.has_seed_templates:
        _print("  Enter seed prompts one per line. Type 'done' when finished.\n")
        while True:
            t = _ask("  Seed template (or 'done')")
            if t.lower() == "done":
                break
            state.seed_templates.append(t)
        _print(f"  Collected {len(state.seed_templates)} seeds.")
    else:
        _print("  The script will generate seeds from your task description.")


def _step_backend(state: WizardState) -> None:
    _print("\n[bold yellow]Step 9 of 10 — LLM Backend[/bold yellow]" if _HAS_RICH
           else "\nStep 9 of 10 — LLM Backend")
    _print("Which LLM backend will power the evolution?\n")

    if _HAS_RICH:
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("[bold green]ollama[/bold green]",
                     "Local model via Ollama (free, private, needs ollama running)")
        tbl.add_row("[bold blue]openai[/bold blue]",
                     "OpenAI API (GPT-4o-mini etc, needs OPENAI_API_KEY)")
        tbl.add_row("[bold cyan]azure[/bold cyan]",
                     "Azure OpenAI (enterprise, needs endpoint + deployment)")
        _console.print(tbl)
    else:
        print("  ollama  — Local model via Ollama")
        print("  openai  — OpenAI API")
        print("  azure   — Azure OpenAI")

    state.backend = _ask("\n  Backend", default="ollama",
                          choices=["ollama", "openai", "azure"])

    if state.backend == "ollama":
        state.model = _ask("  Model name", default="llama3.2")
    elif state.backend == "openai":
        state.model = _ask("  Model name", default="gpt-4o-mini")
    else:
        state.model = _ask("  Deployment name", default="gpt-4o-mini")


def _step_config(state: WizardState) -> None:
    _print("\n[bold yellow]Step 10 of 10 — Evolution Configuration[/bold yellow]" if _HAS_RICH
           else "\nStep 10 of 10 — Evolution Configuration")
    _print("Choose a configuration preset or customise.\n")

    if _HAS_RICH:
        tbl = Table(show_header=True, box=None, padding=(0, 2))
        tbl.add_column("Preset", width=12)
        tbl.add_column("Generations", width=12)
        tbl.add_column("Population", width=12)
        tbl.add_column("Islands", width=10)
        tbl.add_column("Best for")
        tbl.add_row("standard", "3", "4", "2", "Quick exploration (~5 min)")
        tbl.add_row("[green]deep[/green]", "5", "6", "3",
                     "Thorough search (~15 min)")
        tbl.add_row("custom", "—", "—", "—", "You set everything")
        _console.print(tbl)
    else:
        print("  standard — 3 gen, 4 pop, 2 islands (quick)")
        print("  deep     — 5 gen, 6 pop, 3 islands (thorough)")
        print("  custom   — You set everything")

    state.config_preset = _ask("\n  Preset", default="standard",
                                choices=["standard", "deep", "custom"])
    if state.config_preset == "standard":
        state.iterations = 3
        state.population_size = 4
        state.num_islands = 2
    elif state.config_preset == "deep":
        state.iterations = 5
        state.population_size = 6
        state.num_islands = 3
    else:
        state.iterations = _ask_int("  Generations", default=5)
        state.population_size = _ask_int("  Population size per island", default=4)
        state.num_islands = _ask_int("  Number of islands", default=2)


# ── Script generation ────────────────────────────────────────────────────

def _generate_script(state: WizardState) -> str:
    """Build a self-contained Python script from wizard answers."""
    sections: list[str] = []

    # ── Header ───────────────────────────────────────────
    sections.append(textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Prompt evolution script generated by EvoSim Wizard.

        Task: {state.task_description[:80]}
        Run:  uv run python {{script_name}}
        """

        from __future__ import annotations

        import json
        import os
        import sys
        import textwrap
        import time
        from pathlib import Path
    '''))

    # ── Imports ──────────────────────────────────────────
    imports = [
        "from prompture.prompt_evolver import (",
        "    LLMBackend,",
        "    LLMClient,",
        "    PromptCandidate,",
        "    PromptEvolverConfig,",
        ")",
    ]
    strategy_imports: list[str] = []
    if state.has_ground_truth != "yes" or len(state.strategies) > 1:
        strategy_imports.append("from prompture.strategies import (")
        needed = set()
        if "llm_judge" in state.strategies or "composite" in state.strategies:
            needed.add("    LLMJudge,")
        if "self_consistency" in state.strategies or "composite" in state.strategies:
            needed.add("    SelfConsistencyScorer,")
        if "proxy_metrics" in state.strategies or "composite" in state.strategies:
            needed.update(["    ProxyMetricsScorer,", "    ProxyCheck,"])
        if "tool_success" in state.strategies:
            needed.update(["    ToolSuccessScorer,", "    ToolResult,"])
        if "preference" in state.strategies:
            needed.update(["    PreferenceScorer,", "    PreferencePair,"])
        if "human" in state.strategies or state.human_eval == "always":
            needed.add("    HumanTournament,")
        if "composite" in state.strategies or len(state.strategies) > 1:
            needed.add("    CompositeScorer,")
        needed.update([
            "    NoEvalPromptEvolver,",
            "    NoEvalConfig,",
            "    ProblemType,",
            "    Scorer,",
        ])
        strategy_imports.extend(sorted(needed))
        strategy_imports.append(")")

    sections.append("\n".join(imports))
    if strategy_imports:
        sections.append("\n".join(strategy_imports))

    # ── Constants ────────────────────────────────────────
    task_escaped = state.task_description.replace("\\", "\\\\").replace('"', '\\"')
    sections.append(textwrap.dedent(f'''\

        # ── Task description ──────────────────────────────────
        TASK_DESCRIPTION = """{task_escaped}"""
    '''))

    # ── Backend config ───────────────────────────────────
    backend_map = {
        "ollama": "LLMBackend.OLLAMA",
        "openai": "LLMBackend.OPENAI",
        "azure": "LLMBackend.AZURE_OPENAI",
    }
    backend_enum = backend_map[state.backend]
    model_escaped = state.model.replace('"', '\\"')

    if state.backend == "ollama":
        backend_block = textwrap.dedent(f'''\
            BACKEND = {backend_enum}
            MODEL = os.getenv("OLLAMA_MODEL", "{model_escaped}")
        ''')
    elif state.backend == "openai":
        backend_block = textwrap.dedent(f'''\
            BACKEND = {backend_enum}
            MODEL = os.getenv("OPENAI_MODEL", "{model_escaped}")
        ''')
    else:
        backend_block = textwrap.dedent(f'''\
            BACKEND = {backend_enum}
            MODEL = os.getenv("AZURE_DEPLOYMENT", "{model_escaped}")
        ''')
    sections.append(backend_block)

    # ── Seed templates ───────────────────────────────────
    if state.seed_templates:
        seeds_code = "SEED_TEMPLATES = [\n"
        for tpl in state.seed_templates:
            tpl_escaped = tpl.replace("\\", "\\\\").replace('"', '\\"')
            seeds_code += f'    "{tpl_escaped}",\n'
        seeds_code += "]\n"
    else:
        seeds_code = textwrap.dedent(f'''\
            SEED_TEMPLATES = [
                TASK_DESCRIPTION,
                TASK_DESCRIPTION + "\\n\\nThink step-by-step before answering.",
                TASK_DESCRIPTION + "\\n\\nBe concise. Output only the result.",
                TASK_DESCRIPTION + "\\n\\nFollow the format exactly. No extra text.",
            ]
        ''')
    sections.append(seeds_code)

    # ── Test inputs ──────────────────────────────────────
    if state.test_input_file:
        sections.append(textwrap.dedent(f'''\
            # Load test inputs from file
            _test_path = Path("{state.test_input_file}")
            TEST_INPUTS = [line.strip() for line in _test_path.read_text().splitlines() if line.strip()]
        '''))
    elif state.test_inputs:
        inputs_code = "TEST_INPUTS = [\n"
        for ti in state.test_inputs:
            ti_escaped = ti.replace("\\", "\\\\").replace('"', '\\"')
            inputs_code += f'    "{ti_escaped}",\n'
        inputs_code += "]\n"
        sections.append(inputs_code)
    else:
        sections.append(textwrap.dedent('''\
            # TODO: Replace with real test inputs for your agent
            TEST_INPUTS = [
                "Example query 1 — replace with a real user request",
                "Example query 2 — replace with another request",
                "Example query 3 — replace with an edge case",
            ]
        '''))

    # ── Domain mutations ─────────────────────────────────
    if state.domain_mutations:
        mut_code = "DOMAIN_MUTATIONS = [\n"
        for m in state.domain_mutations:
            m_escaped = m.replace("\\", "\\\\").replace('"', '\\"')
            mut_code += f'    "{m_escaped}",\n'
        mut_code += "]\n"
        sections.append(mut_code)
    else:
        sections.append(textwrap.dedent('''\
            # Auto-generated mutation ideas — customise for your domain
            DOMAIN_MUTATIONS = [
                "Add chain-of-thought reasoning",
                "Enforce strict output format",
                "Add error recovery instructions",
                "Inject few-shot examples",
                "Emphasise parameter extraction",
                "Add role-play framing",
                "Shorten to single paragraph",
                "Add 'think before answering' preamble",
                "Specify forbidden output patterns",
                "Add edge-case handling rules",
                "Rewrite in imperative voice",
                "Add output validation step",
            ]
        '''))

    # ── Eval data (ground-truth only) ────────────────────
    if state.has_ground_truth in ("yes", "partial"):
        if state.eval_file:
            sections.append(textwrap.dedent(f'''\

                # ── Ground-truth evaluation data ──────────────────────
                with open("{state.eval_file}") as _f:
                    EVAL_DATA = json.load(_f)
            '''))
        elif state.eval_examples:
            sections.append("\n# ── Ground-truth evaluation data ──────────────────────")
            sections.append("EVAL_DATA = " + json.dumps(state.eval_examples, indent=4))
        else:
            sections.append(textwrap.dedent('''\

                # ── Ground-truth evaluation data ──────────────────────
                # TODO: Add your labelled examples
                EVAL_DATA = [
                    {"input": "example query", "expected": "expected output"},
                ]
            '''))

    # ── Scorer setup ─────────────────────────────────────
    scorer_setup = _build_scorer_setup(state)
    sections.append(scorer_setup)

    # ── Human evaluation functions ───────────────────────
    if state.human_eval in ("always", "final"):
        sections.append(_build_human_eval_block(state))

    # ── Main evolution logic ─────────────────────────────
    sections.append(_build_main_block(state))

    raw = "\n\n".join(sections)
    # Replace {script_name} placeholder
    return raw.replace("{script_name}", "evolve_prompt.py")


def _build_scorer_setup(state: WizardState) -> str:
    """Generate the scorer construction code."""
    lines: list[str] = [
        "",
        "# ── Scoring setup ─────────────────────────────────────────",
        "def build_scorer(client: LLMClient) -> Scorer:",
        '    """Build the scoring strategy for prompt evaluation."""',
    ]

    scorers_with_weights: list[tuple[str, str]] = []

    if "llm_judge" in state.strategies or "composite" in state.strategies:
        rubric = state.llm_judge_rubric or (
            f"Rate this output for the task: {state.task_description[:100]}. "
            "Score 0-10 on: correctness, format compliance, completeness."
        )
        rubric_escaped = rubric.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    judge = LLMJudge(rubric="{rubric_escaped}")')
        scorers_with_weights.append(("judge", "0.3"))

    if "self_consistency" in state.strategies or "composite" in state.strategies:
        lines.append("    consistency = SelfConsistencyScorer(num_samples=3)")
        scorers_with_weights.append(("consistency", "0.25"))

    if "proxy_metrics" in state.strategies or "composite" in state.strategies:
        lines.append("    checks = [")
        for check_name in (state.proxy_checks or ["valid_json", "max_length"]):
            if check_name == "valid_json":
                lines.append("        ProxyCheck(")
                lines.append('            name="valid_json",')
                lines.append("            check_fn=_is_valid_json,")
                lines.append("            weight=2.0,")
                lines.append("        ),")
            elif check_name == "has_function_name":
                lines.append("        ProxyCheck(")
                lines.append('            name="has_function_name",')
                lines.append('            check_fn=lambda x: any(c.isupper() for c in x),')
                lines.append("            weight=1.5,")
                lines.append("        ),")
            elif check_name == "bracket_format":
                lines.append("        ProxyCheck(")
                lines.append('            name="bracket_format",')
                lines.append("            check_fn=lambda x: '[' in x and '(' in x,")
                lines.append("            weight=2.0,")
                lines.append("        ),")
            elif check_name == "max_length":
                lines.append("        ProxyCheck(")
                lines.append('            name="max_length",')
                lines.append("            check_fn=lambda x: len(x) < 500,")
                lines.append("            weight=1.0,")
                lines.append("        ),")
            elif check_name == "no_explanation":
                lines.append("        ProxyCheck(")
                lines.append('            name="no_explanation",')
                lines.append("            check_fn=lambda x: len(x.splitlines()) <= 3,")
                lines.append("            weight=1.0,")
                lines.append("        ),")
        lines.append("    ]")
        lines.append("    proxy = ProxyMetricsScorer(checks=checks)")
        scorers_with_weights.append(("proxy", "0.2"))

    if "tool_success" in state.strategies:
        lines.append("")
        lines.append("    # TODO: Implement your tool executor")
        lines.append("    def _tool_executor(name: str, params: dict) -> ToolResult:")
        lines.append('        """Execute a tool call and return result."""')
        lines.append("        # Replace with your actual tool execution logic")
        lines.append("        return ToolResult(success=True, return_code=200)")
        lines.append("")
        lines.append("    tool_scorer = ToolSuccessScorer(tool_executor=_tool_executor)")
        scorers_with_weights.append(("tool_scorer", "0.25"))

    if "preference" in state.strategies:
        lines.append("")
        lines.append("    # TODO: Add your preference pairs")
        lines.append("    pairs = [")
        lines.append("        PreferencePair(")
        lines.append('            input_text="example input",')
        lines.append('            good_output="correct format output",')
        lines.append('            bad_output="incorrect format output",')
        lines.append("        ),")
        lines.append("    ]")
        lines.append("    pref = PreferenceScorer(pairs=pairs)")
        scorers_with_weights.append(("pref", "0.2"))

    if "human" in state.strategies or state.human_eval == "always":
        lines.append("")
        lines.append("    human = HumanTournament()")
        scorers_with_weights.append(("human", "0.3"))

    # Combine into composite or return single
    if len(scorers_with_weights) > 1:
        lines.append("")
        lines.append("    return CompositeScorer([")
        for name, weight in scorers_with_weights:
            lines.append(f"        ({name}, {weight}),")
        lines.append("    ])")
    elif len(scorers_with_weights) == 1:
        lines.append(f"    return {scorers_with_weights[0][0]}")
    else:
        lines.append("    # Fallback: LLM-as-Judge")
        rubric = f"Rate this output for: {state.task_description[:80]}"
        lines.append(f'    return LLMJudge(rubric="{rubric}")')

    # Helper function for JSON check
    if any(c == "valid_json" for c in state.proxy_checks):
        lines.insert(2, "")
        lines.insert(3, "def _is_valid_json(text: str) -> bool:")
        lines.insert(4, "    try:")
        lines.insert(5, "        json.loads(text)")
        lines.insert(6, "        return True")
        lines.insert(7, "    except (json.JSONDecodeError, ValueError):")
        lines.insert(8, "        return False")
        lines.insert(9, "")

    return "\n".join(lines)


def _build_human_eval_block(state: WizardState) -> str:
    """Generate the human evaluation function."""
    return textwrap.dedent('''\
        # ── Human evaluation ──────────────────────────────────────
        def human_select_winner(candidates: list[dict]) -> dict:
            """Present top candidates to a human for final selection."""
            print()
            print("=" * 60)
            print("  HUMAN EVALUATION — Pick the best prompt")
            print("=" * 60)
            for i, c in enumerate(candidates, 1):
                print(f"\\n  [{i}] Score: {c['score']:.1%}")
                print(f"      {textwrap.shorten(c['prompt'], width=120)}")
            print()
            while True:
                try:
                    choice = int(input(f"  Your pick (1-{len(candidates)}): ")) - 1
                    if 0 <= choice < len(candidates):
                        return candidates[choice]
                except (ValueError, EOFError):
                    pass
                print(f"  Enter a number between 1 and {len(candidates)}")
    ''')


def _build_main_block(state: WizardState) -> str:
    """Generate the main execution block."""
    backend_kwarg = ""
    if state.backend == "ollama":
        backend_kwarg = (
            "        backend=BACKEND,\n"
            "        ollama_model=MODEL,\n"
        )
    elif state.backend == "openai":
        backend_kwarg = (
            "        backend=BACKEND,\n"
            "        openai_model=MODEL,\n"
        )
    else:
        backend_kwarg = (
            "        backend=BACKEND,\n"
            "        azure_deployment=MODEL,\n"
        )

    config_class = "NoEvalConfig" if state.has_ground_truth != "yes" else "PromptEvolverConfig"
    if state.has_ground_truth == "yes" and "ground_truth" in state.strategies:
        config_class = "PromptEvolverConfig"

    lines = [
        "",
        "# ── Main ─────────────────────────────────────────────────",
        'if __name__ == "__main__":',
        "    print()",
        '    print("=" * 60)',
        f'    print("  Prompture Prompt Evolution")',
        f'    print("  Task: {state.task_description[:50]}...")',
        '    print("=" * 60)',
        "    print()",
        "",
    ]

    # Build config
    problem_type_enum = {
        "tool_routing": "ProblemType.TOOL_ROUTING",
        "classification": "ProblemType.CLASSIFICATION",
    }[state.problem_type]
    lines.append(f"    config = NoEvalConfig(")
    lines.append(f"        iterations={state.iterations},")
    lines.append(f"        population_size={state.population_size},")
    lines.append(f"        num_islands={state.num_islands},")
    lines.append(f"        problem_type={problem_type_enum},")
    lines.append(f"{backend_kwarg}    )")

    # Build client for scorer
    lines.append("")
    lines.append("    # Build LLM client and scorer")
    lines.append("    from prompture.prompt_evolver import LLMClient, PromptEvolverConfig")
    lines.append("    _llm_cfg = PromptEvolverConfig(")
    lines.append(f"{backend_kwarg}    )")
    lines.append("    client = LLMClient(_llm_cfg)")
    lines.append("    scorer = build_scorer(client)")

    # Run evolution
    lines.append("")
    lines.append("    # Run evolution")
    lines.append("    evolver = NoEvalPromptEvolver(")
    lines.append("        task_description=TASK_DESCRIPTION,")
    lines.append("        test_inputs=TEST_INPUTS,")
    lines.append("        scorer=scorer,")
    lines.append("        config=config,")
    lines.append("        seed_templates=SEED_TEMPLATES,")
    lines.append("    )")
    lines.append("    print()")
    lines.append('    print("  Starting evolution...")')
    lines.append("    start = time.time()")
    lines.append("    result = evolver.run()")
    lines.append("    elapsed = time.time() - start")

    # Results
    lines.append("")
    lines.append("    # Results")
    lines.append("    print()")
    lines.append('    print("=" * 60)')
    lines.append('    print("  EVOLUTION COMPLETE")')
    lines.append('    print("=" * 60)')
    lines.append('    print(f"  Best fitness:  {result.best_score:.1%}")')
    lines.append('    print(f"  Temperature:   {result.best_temperature:.3f}")')
    lines.append('    print(f"  Top-p:         {result.best_top_p:.3f}")')
    lines.append('    print(f"  Wall time:     {elapsed:.1f}s")')
    lines.append("    print()")
    lines.append('    print("  Best prompt:")')
    lines.append('    print(f"    {result.best_prompt[:200]}")')

    # Human final selection
    if state.human_eval == "final":
        lines.append("")
        lines.append("    # Human final selection from top candidates")
        lines.append("    top_k = sorted(result.all_candidates, key=lambda c: c.score, reverse=True)[:5]")
        lines.append('    candidates = [{"prompt": c.template, "score": c.score} for c in top_k]')
        lines.append("    winner = human_select_winner(candidates)")
        lines.append("    print()")
        lines.append('    print("  ★ Human-selected winner:")')
        lines.append('    print(f"    {winner[\'prompt\'][:200]}")')

    # Save
    lines.append("")
    lines.append("    # Save results")
    lines.append('    out_path = Path("evolution_results.json")')
    lines.append("    out_path.write_text(json.dumps({")
    lines.append('        "task": TASK_DESCRIPTION,')
    lines.append('        "best_prompt": result.best_prompt,')
    lines.append('        "best_score": result.best_score,')
    lines.append('        "best_temperature": result.best_temperature,')
    lines.append('        "best_top_p": result.best_top_p,')
    lines.append('        "wall_time": elapsed,')
    lines.append('        "iterations": result.iterations_run,')
    lines.append("    }, indent=2))")
    lines.append('    print(f"\\n  Results saved to {out_path}")')
    lines.append('    print("  Done.")')

    return "\n".join(lines)


# ── Wizard summary ───────────────────────────────────────────────────────

def _show_summary(state: WizardState) -> None:
    """Display a summary of wizard choices before generating."""
    _print("")
    if _HAS_RICH:
        tbl = Table(title="Configuration Summary", show_header=True,
                     border_style="cyan")
        tbl.add_column("Setting", style="bold", width=22)
        tbl.add_column("Value")

        tbl.add_row("Problem type", state.problem_type)
        tbl.add_row("Task", textwrap.shorten(state.task_description, width=50))
        tbl.add_row("Ground truth", state.has_ground_truth)
        tbl.add_row("Strategies", ", ".join(state.strategies))
        tbl.add_row("Human eval", state.human_eval)
        tbl.add_row("Domain mutations",
                     f"{len(state.domain_mutations)} custom" if state.domain_mutations
                     else "auto-generated")
        tbl.add_row("Seed templates",
                     f"{len(state.seed_templates)} custom" if state.seed_templates
                     else "auto-generated")
        tbl.add_row("Backend", f"{state.backend} ({state.model})")
        tbl.add_row("Config",
                     f"{state.config_preset} ({state.iterations} gen, "
                     f"{state.population_size} pop, {state.num_islands} islands)")
        _console.print(tbl)
    else:
        print("  Configuration Summary:")
        print(f"  Problem type: {state.problem_type}")
        print(f"  Task:         {state.task_description[:50]}")
        print(f"  Ground truth: {state.has_ground_truth}")
        print(f"  Strategies:   {', '.join(state.strategies)}")
        print(f"  Human eval:   {state.human_eval}")
        print(f"  Backend:      {state.backend} ({state.model})")
        print(f"  Config:       {state.config_preset}")


# ── Public entry point ───────────────────────────────────────────────────

def run_wizard(output: str = "evolve_prompt.py") -> str:
    """Run the interactive wizard and return the generated script path.

    Parameters
    ----------
    output :
        Path for the generated Python script.

    Returns
    -------
    str
        Absolute path of the generated file.
    """
    _banner()

    state = WizardState()

    # Walk through each step
    _step_problem_type(state)
    _step_task(state)
    _step_ground_truth(state)
    _step_test_inputs(state)
    _step_scoring(state)
    _step_mutations(state)
    _step_human_eval(state)
    _step_seeds(state)
    _step_backend(state)
    _step_config(state)

    # Confirm
    _show_summary(state)
    if not _confirm("\n  Generate the script?", default=True):
        _print("  Aborted.")
        return ""

    # Generate
    script = _generate_script(state)
    out_path = Path(output)
    out_path.write_text(script)
    out_path.chmod(0o755)

    _print(f"\n  [green]✓ Generated:[/green] {out_path.resolve()}" if _HAS_RICH
           else f"\n  ✓ Generated: {out_path.resolve()}")
    _print(f"\n  Run it with:")
    _print(f"    uv run python {output}")
    _print("")

    return str(out_path.resolve())
