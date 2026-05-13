#!/usr/bin/env python3
"""
Cookbook Recipe — BrowserGym WebLINX Action Prediction Prompt Evolution
======================================================================

Evolves system prompts for the **BrowserGym** leaderboard's web-browsing
action prediction task using the **WebLINX** dataset — 2,300+ expert
demonstrations across 150 real websites with 100K+ interaction turns.

WebLINX provides offline evaluation: each turn contains a browsing
context (goal, chat history, accessibility tree snapshot) and a
ground-truth browser action.  The model must predict the correct next
action given the context — no live browser required.

Scoring
-------
* **Intent accuracy**  (0.40 weight) — correct action type (click, fill, etc.)
* **Element F1**       (0.35 weight) — correct target element (uid match)
* **Argument accuracy** (0.25 weight) — correct action arguments (text, URL, etc.)

Task categories
---------------
* **click**       — clicking links, buttons, checkboxes
* **fill**        — typing into input fields
* **select**      — dropdown selection
* **navigate**    — goto, go_back, go_forward
* **communicate** — say responses to the user
* **other**       — scroll, hover, noop, tab_*

Algorithm
---------
* **Deep** — 8 iterations, pop 10, 3 islands, LLM-assisted mutations,
  adaptive error analysis, post-splice refinement.  Targets GPT-4.1
  via Azure OpenAI.

Usage::

    uv sync --extra llm
    pip install datasets  # for HuggingFace dataset download
    uv run python examples/cookbook/browsergym/prompt_evolution_browsergym.py

The script saves an experiment log to
``examples/cookbook/browsergym/logs/browsergym_evolution_log.json``
for dashboard consumption, including lineage data.

References
----------
* BrowserGym leaderboard : https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard
* BrowserGym repo        : https://github.com/ServiceNow/BrowserGym
* WebLINX dataset        : https://huggingface.co/datasets/McGill-NLP/WebLINX
* WebLINX paper          : https://arxiv.org/abs/2402.05930
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ensure the repo root is on sys.path so MutaGenAI is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Force unbuffered stdout so output appears immediately in VS Code
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
    PromptEvolverResult,
    ProblemType,
    _crossover_templates,
    _llm_mutate_template,
    _mutate_template,
    _refine_template,
)
from MutaGenAI.seed_loader import (
    load_seed_template_config,
    penalties_to_proxy_checks,
)
import MutaGenAI.prompt_evolver as _pe


# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = Path(__file__).resolve().parent
SEED_DIR = EXPERIMENT_DIR / "seed_templates"
LOG_DIR = EXPERIMENT_DIR / "logs"
CACHE_DIR = EXPERIMENT_DIR / ".weblinx_cache"


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BrowsingTurn:
    """A single action-prediction turn from WebLINX."""

    turn_id: str
    demo_name: str
    goal: str
    chat_history: list[dict[str, str]]
    axtree_snapshot: str          # accessibility tree text
    url: str                      # current page URL
    action_type: str              # ground-truth action type
    action_args: dict[str, Any]   # ground-truth arguments
    action_raw: str               # raw ground-truth action string
    category: str                 # click, fill, select, navigate, communicate, other

    @property
    def action_bid(self) -> str | None:
        """Extract the target element uid from action arguments."""
        return self.action_args.get("uid")

    @property
    def action_text(self) -> str | None:
        """Extract text argument (for text_input, say, etc.)."""
        return (
            self.action_args.get("text")
            or self.action_args.get("utterance")
            or self.action_args.get("value")
            or self.action_args.get("option")
        )


# ─────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────

# Match ACTION: action_type(args) or just action_type(args)
_ACTION_RE = re.compile(
    r"(?:ACTION:\s*)?(\w+)\s*\(([^)]*)\)",
    re.IGNORECASE,
)

# Known action types — includes both WebLINX and BrowserGym names
_KNOWN_ACTIONS = frozenset({
    "click", "fill", "text_input", "select_option", "goto", "go_back",
    "go_forward", "scroll", "hover", "send_msg_to_user", "say",
    "submit", "load", "noop", "tab_close", "tab_focus", "new_tab",
})

# Normalise model-predicted action names to ground-truth WebLINX names
_ACTION_ALIASES: dict[str, str] = {
    "fill": "text_input",
    "type": "text_input",
    "input": "text_input",
    "send_msg_to_user": "say",
    "respond": "say",
    "reply": "say",
    "goto": "load",
    "navigate": "load",
    "open": "load",
}


def parse_action(response: str) -> tuple[str, dict[str, Any]]:
    """Parse a model response to extract action type and arguments.

    Handles both WebLINX-native format (``click(uid="...")``) and
    BrowserGym-style format (``ACTION: click("...")``) and normalises
    action names via ``_ACTION_ALIASES``.

    Returns (action_type, args_dict).
    """
    if not response:
        return "", {}

    # Try structured ACTION: prefix first
    m = _ACTION_RE.search(response)
    if m:
        action = m.group(1).lower()
        action = _ACTION_ALIASES.get(action, action)
        raw_args = m.group(2).strip()

        if action in _KNOWN_ACTIONS:
            args = _parse_action_args(action, raw_args)
            return action, args

    # Try each known action as a keyword
    resp_lower = response.lower()
    for action in _KNOWN_ACTIONS:
        pattern = re.compile(rf"\b{action}\s*\(([^)]*)\)", re.IGNORECASE)
        m2 = pattern.search(response)
        if m2:
            args = _parse_action_args(action, m2.group(1).strip())
            return action, args

    # Try aliases
    for alias, canonical in _ACTION_ALIASES.items():
        pattern = re.compile(rf"\b{alias}\s*\(([^)]*)\)", re.IGNORECASE)
        m3 = pattern.search(response)
        if m3:
            args = _parse_action_args(canonical, m3.group(1).strip())
            return canonical, args

    # Last resort: look for just the action name
    for action in _KNOWN_ACTIONS:
        if action in resp_lower:
            return action, {}

    return "", {}


def _parse_action_args(action: str, raw: str) -> dict[str, Any]:
    """Parse raw argument string into a dict.

    Handles both positional args (``"abc", "def"``) and keyword args
    (``uid="abc", text="def"``).
    """
    if not raw:
        return {}

    args: dict[str, Any] = {}

    # Try keyword-argument parsing first (WebLINX format)
    kv_matches = list(_WEBLINX_KV_RE.finditer(raw))
    if kv_matches:
        for kv in kv_matches:
            key = kv.group(1)
            value = kv.group(2) if kv.group(2) is not None else kv.group(3)
            if kv.group(3) is not None:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            args[key] = value
        return args

    # Fall back to positional argument parsing (BrowserGym format)
    parts = _split_args(raw)

    if action in ("click", "hover", "submit"):
        if parts:
            args["uid"] = _unquote(parts[0])
    elif action in ("text_input", "fill"):
        if len(parts) >= 2:
            args["text"] = _unquote(parts[0])
            args["uid"] = _unquote(parts[1])
        elif len(parts) == 1:
            args["text"] = _unquote(parts[0])
    elif action == "select_option":
        if len(parts) >= 2:
            args["uid"] = _unquote(parts[0])
            args["option"] = _unquote(parts[1])
        elif len(parts) == 1:
            args["uid"] = _unquote(parts[0])
    elif action in ("goto", "load"):
        if parts:
            args["url"] = _unquote(parts[0])
    elif action == "scroll":
        if len(parts) >= 2:
            args["x"] = parts[0]
            args["y"] = parts[1]
    elif action in ("send_msg_to_user", "say"):
        if parts:
            args["utterance"] = _unquote(parts[0])

    return args


def _split_args(s: str) -> list[str]:
    """Split comma-separated arguments, respecting quoted strings."""
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    quote_char = ""

    for ch in s:
        if ch in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == quote_char and in_quote:
            in_quote = False
            current.append(ch)
        elif ch == "," and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current).strip())

    return [p for p in parts if p]


def _unquote(s: str) -> str:
    """Remove surrounding quotes from a string."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


# ─────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────


def _normalise(s: str) -> str:
    """Normalise for comparison."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def score_action(
    predicted: tuple[str, dict[str, Any]],
    expected: BrowsingTurn,
) -> tuple[float, dict[str, Any]]:
    """Score a predicted action against the ground truth.

    Returns (score 0.0–1.0, detail dict).

    Weights:
      - Intent accuracy   (0.40): correct action type
      - Element F1        (0.35): correct target element bid
      - Argument accuracy (0.25): correct arguments (text, URL, etc.)
    """
    pred_action, pred_args = predicted
    exp_action = expected.action_type
    exp_args = expected.action_args
    exp_bid = expected.action_bid
    exp_text = expected.action_text

    detail: dict[str, Any] = {
        "intent_match": False,
        "element_match": False,
        "arg_score": 0.0,
        "error": None,
    }

    # Normalise predicted action name through aliases
    pred_action_norm = _ACTION_ALIASES.get(pred_action, pred_action)
    exp_action_norm = _ACTION_ALIASES.get(exp_action, exp_action)

    # Intent accuracy — action type match
    intent_match = _normalise(pred_action_norm) == _normalise(exp_action_norm)
    detail["intent_match"] = intent_match
    intent_score = 1.0 if intent_match else 0.0

    if not pred_action:
        detail["error"] = "NO_ACTION_PARSED"
        return 0.0, detail

    if not intent_match:
        detail["error"] = f"INTENT_MISMATCH: predicted={pred_action}, expected={exp_action}"
        # Partial credit if the action is related (e.g. click vs submit)
        _related = {
            frozenset({"click", "hover", "submit"}),
            frozenset({"load", "goto", "go_back", "go_forward"}),
            frozenset({"text_input", "fill"}),
            frozenset({"say", "send_msg_to_user"}),
        }
        for group in _related:
            if pred_action_norm in group and exp_action_norm in group:
                intent_score = 0.3
                break

    # Element F1 — uid match
    pred_uid = pred_args.get("uid")
    if exp_bid is not None and pred_uid is not None:
        element_match = _normalise(str(pred_uid)) == _normalise(str(exp_bid))
        detail["element_match"] = element_match
        element_score = 1.0 if element_match else 0.0
    elif exp_bid is None:
        # No element expected (e.g. load, say, scroll, noop)
        element_score = 1.0 if pred_uid is None else 0.5
        detail["element_match"] = pred_uid is None
    else:
        # Expected element but none predicted
        element_score = 0.0

    # Argument accuracy — text/utterance/URL match
    if exp_text is not None:
        pred_text = (
            pred_args.get("text")
            or pred_args.get("utterance")
            or pred_args.get("value")
            or pred_args.get("option")
            or pred_args.get("url")
            or ""
        )
        if _normalise(pred_text) == _normalise(exp_text):
            arg_score = 1.0
        elif _normalise(exp_text) in _normalise(pred_text) or _normalise(pred_text) in _normalise(exp_text):
            arg_score = 0.6  # partial text match
        elif pred_text:
            arg_score = 0.2  # something attempted
        else:
            arg_score = 0.0
    elif exp_action in ("goto", "load"):
        pred_url = pred_args.get("url", "")
        exp_url = exp_args.get("url", "")
        if exp_url and pred_url:
            arg_score = 1.0 if _normalise(pred_url) == _normalise(exp_url) else 0.3
        else:
            arg_score = 1.0
    else:
        # No text/URL argument expected
        arg_score = 1.0

    detail["arg_score"] = arg_score

    total = 0.40 * intent_score + 0.35 * element_score + 0.25 * arg_score
    return total, detail


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading from HuggingFace (WebLINX)
# ─────────────────────────────────────────────────────────────────────────

_HF_REPO = "McGill-NLP/WebLINX"

# WebLINX action types → our categories
_ACTION_CATEGORY: dict[str, str] = {
    "click": "click",
    "text_input": "fill",
    "submit": "click",
    "load": "navigate",
    "say": "communicate",
    "scroll": "other",
}

# Regex to parse WebLINX action strings like:
#   click(uid="7c464033-172f-4613")
#   text_input(text="Hire Contractors", uid="d8a117c3-5d04-4626")
#   say(speaker="navigator", utterance="Here are the options...")
#   scroll(x=0, y=250)
#   submit(uid="c95d2472-337f-4bf4")
#   load(url="https://example.com")
_WEBLINX_ACTION_RE = re.compile(r"^(\w+)\((.+)\)$", re.DOTALL)
_WEBLINX_KV_RE = re.compile(r'(\w+)=(?:"((?:[^"\\]|\\.)*)"|(\d+(?:\.\d+)?))')

# Regex to extract instructor goals from action_history
_INSTRUCTOR_RE = re.compile(
    r'say\(speaker="instructor",\s*utterance="((?:[^"\\]|\\.)*)"\)',
)


def _parse_weblinx_action(action_str: str) -> tuple[str, dict[str, Any]]:
    """Parse a WebLINX action string into (action_type, args_dict)."""
    m = _WEBLINX_ACTION_RE.match(action_str.strip())
    if not m:
        return action_str.strip(), {}

    action_type = m.group(1)
    raw_args = m.group(2)

    args: dict[str, Any] = {}
    for kv in _WEBLINX_KV_RE.finditer(raw_args):
        key = kv.group(1)
        value = kv.group(2) if kv.group(2) is not None else kv.group(3)
        # Convert numeric strings
        if kv.group(3) is not None:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        args[key] = value

    return action_type, args


def _extract_goal_from_history(action_history: str) -> tuple[str, list[dict[str, str]]]:
    """Extract the instructor goal and chat history from action_history.

    WebLINX action_history contains say() statements from both
    instructor and navigator.
    """
    chat_history: list[dict[str, str]] = []
    goal = ""

    for m in _INSTRUCTOR_RE.finditer(action_history or ""):
        utterance = m.group(1).replace('\\"', '"').replace("\\n", "\n")
        if not goal:
            goal = utterance
        chat_history.append({"role": "user", "content": utterance})

    return goal, chat_history


def _download_weblinx_data(
    max_demos: int = 50,
    max_turns_per_demo: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Download WebLINX turns from HuggingFace and cache locally.

    Downloads a subset of demonstrations for tractable evolution.
    Each turn is an action-prediction task with full context.

    WebLINX schema
    ~~~~~~~~~~~~~~
    * ``action``         — ground-truth action string, e.g. ``click(uid="...")``
    * ``action_history`` — prior actions + instructor/navigator say()s
    * ``candidates``     — candidate elements with uid, tag, text, bbox
    * ``clean_html``     — cleaned DOM tree
    * ``demo``           — demonstration id
    * ``turn``           — turn index within the demo
    * ``utterances``     — instructor utterance (may be None)
    * ``viewport``       — viewport dimensions

    Parameters
    ----------
    max_demos : int
        Maximum number of demonstrations to sample.
    max_turns_per_demo : int
        Maximum action turns per demonstration.
    seed : int
        Random seed for reproducible subsampling.
    """
    cache_file = CACHE_DIR / f"weblinx_turns_{max_demos}d_{max_turns_per_demo}t.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        print(f"    Loaded {len(data)} cached turns from {cache_file.name}")
        return data

    print("  Downloading WebLINX dataset from HuggingFace...")
    print(f"    (max {max_demos} demos × {max_turns_per_demo} turns each)")

    rng = np.random.default_rng(seed)
    turns: list[dict[str, Any]] = []

    try:
        from datasets import load_dataset
        import huggingface_hub

        # Increase HF Hub timeout to avoid ReadTimeout on slow connections
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 120

        # Load the test split — used for leaderboard evaluation.
        # Use streaming to avoid downloading the full ~20 GB dataset.
        ds = load_dataset(
            _HF_REPO,
            split="test",
            streaming=True,
            download_config=__import__("datasets").DownloadConfig(
                max_retries=3,
            ),
        )

        # Group rows by demonstration name
        demos: dict[str, list[dict[str, Any]]] = {}
        row_count = 0
        for row in ds:
            demo_name = row["demo"]
            # Skip load() actions — they're navigation events, not
            # user-predicted actions
            action_type = row["action"].split("(")[0]
            if action_type == "load":
                continue
            demos.setdefault(demo_name, []).append(dict(row))
            row_count += 1

        print(f"    Found {len(demos)} demonstrations, {row_count} action turns")

        # Sample demonstrations
        demo_names = list(demos.keys())
        if len(demo_names) > max_demos:
            sample_idx = rng.choice(len(demo_names), size=max_demos, replace=False)
            demo_names = [demo_names[int(i)] for i in sample_idx]

        for demo_name in demo_names:
            demo_turns = demos[demo_name]

            # Sort by turn index
            demo_turns.sort(key=lambda t: t["turn"])

            # Sample turns from this demo
            if len(demo_turns) > max_turns_per_demo:
                idx = rng.choice(
                    len(demo_turns), size=max_turns_per_demo, replace=False,
                )
                demo_turns = [demo_turns[int(i)] for i in sorted(idx)]

            for t in demo_turns:
                action_raw = t["action"]
                action_type, action_args = _parse_weblinx_action(action_raw)
                category = _ACTION_CATEGORY.get(action_type, "other")

                # Extract goal from action_history
                goal, chat_history = _extract_goal_from_history(
                    t.get("action_history", ""),
                )
                if not goal:
                    # Fall back to utterances field
                    utt = t.get("utterances")
                    if utt and isinstance(utt, str) and utt.strip() and "N o   i n s t r u c t o r" not in utt:
                        goal = utt
                    else:
                        goal = f"Complete the browsing task in demo {demo_name}"

                # Use candidates as the element context (like an AXTree)
                candidates = t.get("candidates", "") or ""
                if isinstance(candidates, str) and len(candidates) > 3000:
                    candidates = candidates[:3000] + "\n... (truncated)"

                turn_data = {
                    "turn_id": f"{demo_name}_t{t['turn']}",
                    "demo_name": demo_name,
                    "goal": goal,
                    "chat_history": chat_history,
                    "axtree_snapshot": candidates if candidates else "(no candidates available)",
                    "url": "",
                    "action_type": action_type,
                    "action_args": action_args,
                    "action_raw": action_raw,
                    "category": category,
                }
                turns.append(turn_data)

        print(f"    Extracted {len(turns)} action turns from {len(demo_names)} demos")

    except ImportError:
        print("    ✗ 'datasets' library not installed. Install with: pip install datasets")
        print("    Generating synthetic placeholder turns for structure testing...")
        turns = _generate_synthetic_turns(max_demos * max_turns_per_demo, rng)

    except Exception as e:
        import traceback
        print(f"    ✗ Dataset download failed: {e}")
        traceback.print_exc()
        print("    Generating synthetic placeholder turns for structure testing...")
        turns = _generate_synthetic_turns(max_demos * max_turns_per_demo, rng)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(turns, f, indent=2)
    print(f"    Cached {len(turns)} turns to {cache_file.name}")

    return turns


def _generate_synthetic_turns(n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Generate synthetic browsing turns for offline testing.

    These allow the script to run end-to-end without the HuggingFace
    dataset, producing valid (if trivially scored) outputs.
    """
    templates = [
        {
            "goal": "Search for 'machine learning' on Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "axtree": '(uid = abc-001) [[tag]] input [[text]] Search Wikipedia [[bbox]] x=100 y=50 width=200 height=30',
            "action_type": "click",
            "action_args": {"uid": "abc-001"},
            "action_raw": 'click(uid="abc-001")',
            "category": "click",
        },
        {
            "goal": "Search for 'machine learning' on Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "axtree": '(uid = abc-001) [[tag]] input [[text]] Search Wikipedia [[bbox]] x=100 y=50 width=200 height=30',
            "action_type": "text_input",
            "action_args": {"text": "machine learning", "uid": "abc-001"},
            "action_raw": 'text_input(text="machine learning", uid="abc-001")',
            "category": "fill",
        },
        {
            "goal": "Search for 'machine learning' on Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "axtree": '(uid = abc-002) [[tag]] button [[text]] Search [[bbox]] x=310 y=50 width=80 height=30',
            "action_type": "submit",
            "action_args": {"uid": "abc-002"},
            "action_raw": 'submit(uid="abc-002")',
            "category": "click",
        },
        {
            "goal": "Go to reddit.com and find the top post on r/programming",
            "url": "about:blank",
            "axtree": "(uid = def-001) [[tag]] body [[text]]  [[bbox]] x=0 y=0 width=1024 height=768",
            "action_type": "load",
            "action_args": {"url": "https://www.reddit.com/r/programming"},
            "action_raw": 'load(url="https://www.reddit.com/r/programming")',
            "category": "navigate",
        },
        {
            "goal": "Fill out the contact form with name 'John Doe'",
            "url": "https://example.com/contact",
            "axtree": '(uid = ghi-001) [[tag]] input [[text]] Full Name [[bbox]] x=50 y=100 width=300 height=30\n(uid = ghi-002) [[tag]] input [[text]] Email [[bbox]] x=50 y=140 width=300 height=30\n(uid = ghi-003) [[tag]] button [[text]] Submit [[bbox]] x=50 y=200 width=100 height=35',
            "action_type": "text_input",
            "action_args": {"text": "John Doe", "uid": "ghi-001"},
            "action_raw": 'text_input(text="John Doe", uid="ghi-001")',
            "category": "fill",
        },
        {
            "goal": "Scroll down to see more results",
            "url": "https://example.com/results",
            "axtree": '(uid = jkl-001) [[tag]] div [[text]] Result 1 [[bbox]] x=0 y=0 width=800 height=100',
            "action_type": "scroll",
            "action_args": {"x": 0, "y": 250},
            "action_raw": 'scroll(x=0, y=250)',
            "category": "other",
        },
        {
            "goal": "What is the current price of AAPL stock?",
            "url": "https://finance.yahoo.com/quote/AAPL",
            "axtree": '(uid = mno-001) [[tag]] h1 [[text]] Apple Inc. (AAPL) [[bbox]] x=50 y=20 width=400 height=30\n(uid = mno-002) [[tag]] span [[text]] Price: $185.92 [[bbox]] x=50 y=60 width=150 height=20',
            "action_type": "say",
            "action_args": {"speaker": "navigator", "utterance": "The current price of AAPL is $185.92"},
            "action_raw": 'say(speaker="navigator", utterance="The current price of AAPL is $185.92")',
            "category": "communicate",
        },
    ]

    turns: list[dict[str, Any]] = []
    for i in range(n):
        t = templates[i % len(templates)].copy()
        t["turn_id"] = f"synthetic_{i}"
        t["demo_name"] = f"demo_{i // 5}"
        t["chat_history"] = [{"role": "user", "content": t["goal"]}]
        t["axtree_snapshot"] = t.pop("axtree")
        turns.append(t)

    return turns


def load_browsing_turns(
    max_demos: int = 50,
    max_turns_per_demo: int = 5,
    categories: list[str] | None = None,
) -> dict[str, list[BrowsingTurn]]:
    """Load and group WebLINX turns by action category.

    Parameters
    ----------
    max_demos : int
        Max demonstrations to download/sample.
    max_turns_per_demo : int
        Max action turns per demonstration.
    categories : list[str] | None
        Which categories to include. Defaults to all.

    Returns
    -------
    dict mapping category name to list of BrowsingTurn
    """
    if categories is None:
        categories = ["click", "fill", "navigate", "communicate", "other"]

    raw = _download_weblinx_data(max_demos, max_turns_per_demo)

    by_category: dict[str, list[BrowsingTurn]] = {c: [] for c in categories}

    for t in raw:
        cat = t.get("category", "other")
        if cat not in by_category:
            cat = "other"

        turn = BrowsingTurn(
            turn_id=t["turn_id"],
            demo_name=t["demo_name"],
            goal=t["goal"],
            chat_history=t.get("chat_history", []),
            axtree_snapshot=t.get("axtree_snapshot", ""),
            url=t.get("url", ""),
            action_type=t["action_type"],
            action_args=t.get("action_args", {}),
            action_raw=t.get("action_raw", ""),
            category=cat,
        )
        by_category[cat].append(turn)

    # Remove empty categories
    return {k: v for k, v in by_category.items() if v}


# ─────────────────────────────────────────────────────────────────────────
# Context formatting
# ─────────────────────────────────────────────────────────────────────────


def format_browsing_context(turn: BrowsingTurn) -> str:
    """Format a browsing turn into the user message for the LLM.

    Builds a structured prompt with GOAL, CHAT HISTORY, CURRENT URL,
    and CANDIDATE ELEMENTS (from WebLINX candidates field).
    """
    lines: list[str] = []

    lines.append(f"GOAL: {turn.goal}")
    lines.append("")

    if turn.chat_history:
        lines.append("CHAT HISTORY:")
        for msg in turn.chat_history[-5:]:  # last 5 messages
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"  {role}: {content}")
        lines.append("")

    if turn.url:
        lines.append(f"CURRENT URL: {turn.url}")
        lines.append("")

    lines.append("CANDIDATE ELEMENTS:")
    # Limit candidates to prevent context overflow
    candidates = turn.axtree_snapshot
    if len(candidates) > 2500:
        candidates = candidates[:2500] + "\n... (truncated)"
    lines.append(candidates)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Seed templates
# ─────────────────────────────────────────────────────────────────────────

_SEED_CONFIG = load_seed_template_config(
    "browsergym_weblinx", directory=SEED_DIR,
)
SEED_TEMPLATES = _SEED_CONFIG.seeds
_PENALTY_CHECKS = penalties_to_proxy_checks(_SEED_CONFIG.penalties)


# ─────────────────────────────────────────────────────────────────────────
# Domain-specific mutations
# ─────────────────────────────────────────────────────────────────────────

DOMAIN_MUTATIONS = [
    # Element targeting
    "Add instruction: always reference elements by their uid from the candidate elements — never invent uids",
    "Add instruction: prefer clicking interactive elements (buttons, links) over static text",
    "Add instruction: for form inputs, use text_input() not click() — click only selects the field",
    "Add instruction: when multiple elements have similar text, use the uid closest to the user's intent",
    # Action selection
    "Add instruction: use submit() for form submission buttons, click() for general navigation",
    "Add instruction: use load() only when navigating to a new domain or URL — prefer clicking links for same-site navigation",
    "Add instruction: use say() when the goal asks a question and the answer is visible on the page",
    "Add instruction: for scrolling, use scroll(x=0, y=N) where positive N scrolls down and negative scrolls up",
    # Context awareness
    "Add instruction: read the FULL candidate elements before choosing an action — the target may not be the first element",
    "Add instruction: look at the chat history to understand what has already been attempted",
    "Add instruction: if a form field is already filled, skip it and fill the next empty field",
    "Add instruction: the action_history shows prior actions — do not repeat actions already taken",
    # Workflow patterns
    "Add instruction: common patterns — search: click search → text_input query → submit; form: text_input fields → submit",
    "Add instruction: for login flows, text_input username first, then password, then submit",
    "Add instruction: for multi-step tasks, focus on the IMMEDIATE next action only, not the entire plan",
    # Output discipline
    "Add instruction: output EXACTLY ONE action per turn — never chain multiple actions",
    "Add instruction: do not include explanations, reasoning, or commentary — only the ACTION: line",
    "Add instruction: use the exact format ACTION: action_type(key=\"value\") with proper quoting",
    # Chain of thought
    "Prepend: think step by step: what is the goal? what page am I on? what element should I interact with?",
    "Add instruction: before outputting, verify your selected uid exists in the candidate elements",
]


# ─────────────────────────────────────────────────────────────────────────
# LLM configuration — GPT-4.1 via Azure OpenAI
# ─────────────────────────────────────────────────────────────────────────

# Backend selection: set MUTAGENAI_BACKEND=ollama to use local Ollama
_backend_env = os.getenv("MUTAGENAI_BACKEND", "azure").lower()
if _backend_env == "ollama":
    BACKEND = LLMBackend.OLLAMA
    MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
else:
    BACKEND = LLMBackend.AZURE_OPENAI
    MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# ── Token minimization (optional) ────────────────────────────────────────
# When enabled, shorter prompts receive a bonus during evolution.
# The final score is: accuracy * (1 - TOKEN_WEIGHT) + brevity_bonus * TOKEN_WEIGHT
# Set TOKEN_WEIGHT=0.0 to disable (pure accuracy optimisation).
MINIMIZE_TOKENS: bool = bool(os.getenv("MUTAGENAI_MINIMIZE_TOKENS", "0") not in ("0", "", "false"))
TOKEN_WEIGHT: float = float(os.getenv("MUTAGENAI_TOKEN_WEIGHT", "0.15"))
MAX_PROMPT_TOKENS: int = int(os.getenv("MUTAGENAI_MAX_PROMPT_TOKENS", "500"))


def _count_prompt_tokens(text: str) -> int:
    """Count tokens in a prompt string. Uses tiktoken if available, else char estimate."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except (ImportError, KeyError):
        # Fallback: ~4 chars per token for English text
        return len(text) // 4


# ─────────────────────────────────────────────────────────────────────────
# Custom PromptEvolver subclass with BrowserGym action scoring
# ─────────────────────────────────────────────────────────────────────────


class BrowserGymEvolver(PromptEvolver):
    """PromptEvolver subclass that scores via WebLINX action prediction."""

    def __init__(
        self,
        config: PromptEvolverConfig,
        turns: list[BrowsingTurn],
        client: LLMClient,
    ) -> None:
        super().__init__(tools=[], eval_dataset=[], config=config)
        self._turns = turns
        self._bg_client = client
        self._bg_rng = np.random.default_rng(42)
        self._client = client
        self._require_tool_schemas = False

    def _breed(
        self, island: list[PromptCandidate], generation: int,
    ) -> PromptCandidate:
        """Breed using domain-specific browser action mutations."""
        parent_a = self._tournament_select(island)
        parent_hashes = [parent_a.hash]
        operation = "mutation"

        if self._rng.random() < self.config.crossover_rate and len(island) > 1:
            parent_b = self._tournament_select(island)
            child_template = _crossover_templates(
                parent_a.template,
                parent_b.template,
                self._rng,
                require_tool_schemas=False,
            )
            parent_hashes = [parent_a.hash, parent_b.hash]
            operation = "crossover"
        else:
            child_template = parent_a.template

        mutations = list(DOMAIN_MUTATIONS)
        if self._adaptive_pool:
            mutations = mutations + self._adaptive_pool

        # LLM-assisted mutation using failure examples
        if (
            self.config.llm_mutation_rate > 0
            and self._rng.random() < self.config.llm_mutation_rate
            and self._failure_examples
        ):
            n_fail = min(6, len(self._failure_examples))
            fail_idx = self._rng.choice(
                len(self._failure_examples), size=n_fail, replace=False,
            )
            sampled_failures = [self._failure_examples[int(i)] for i in fail_idx]
            child_template = _llm_mutate_template(
                child_template,
                sampled_failures,
                self._client,
                ProblemType.CLASSIFICATION,
                require_tool_schemas=False,
            )
            operation = "llm_mutation"
        elif self._rng.random() < self.config.mutation_rate:
            child_template = _mutate_template(
                child_template,
                self._rng,
                self.config.mutation_rate,
                require_tool_schemas=False,
                mutations=mutations,
            )

        # Optional LLM-based clarity refinement
        if self.config.refine_after_splice:
            child_template = _refine_template(
                child_template,
                self._client,
                require_tool_schemas=False,
            )

        # Strip any tool_schemas contamination
        child_template = child_template.replace("{tool_schemas}", "")
        child_template = child_template.replace("{}", "").strip()

        # Mutate continuous params
        temp = parent_a.temperature + float(self._rng.normal(0, 0.1))
        temp = float(np.clip(temp, *self.config.temperature_range))
        top_p = parent_a.top_p + float(self._rng.normal(0, 0.05))
        top_p = float(np.clip(top_p, *self.config.top_p_range))

        return PromptCandidate(
            template=child_template,
            temperature=temp,
            top_p=top_p,
            generation=generation,
            parent_hashes=parent_hashes,
            operation=operation,
        )

    def _evaluate_candidate(self, candidate: PromptCandidate) -> float:
        """Evaluate a candidate prompt across browsing turns.

        Returns a score in [0, 100] — percentage accuracy.
        """
        # Subsample for speed if configured
        if (
            self.config.eval_sample_size
            and self.config.eval_sample_size < len(self._turns)
        ):
            indices = self._bg_rng.choice(
                len(self._turns), size=self.config.eval_sample_size, replace=False,
            )
            eval_turns = [self._turns[int(i)] for i in indices]
        else:
            eval_turns = self._turns

        total = 0.0
        for turn in eval_turns:
            user_message = format_browsing_context(turn)
            system_prompt = candidate.template

            response = self._bg_client.complete(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )

            if response is None:
                total += float(self._bg_rng.uniform(0, 0.2))
                continue

            predicted = parse_action(response)
            score, detail = score_action(predicted, turn)
            total += score

            # Track error profile for adaptive mutations
            correct = score >= 0.6
            self._error_profile.record(turn.category, correct)
            if not correct:
                pred_str = (
                    f"{predicted[0]}({predicted[1]})"
                    if predicted[0]
                    else response[:80]
                )
                self._failure_examples.append(
                    (user_message[:120], pred_str, turn.action_raw)
                )

        n = len(eval_turns)
        raw_accuracy = (total / n * 100.0) if n else 0.0

        # ── Token minimization bonus ─────────────────────────────────
        if MINIMIZE_TOKENS and TOKEN_WEIGHT > 0:
            prompt_tokens = _count_prompt_tokens(candidate.template)
            length_ratio = min(prompt_tokens / MAX_PROMPT_TOKENS, 1.0)
            brevity_bonus = (1.0 - length_ratio) * 100.0
            return raw_accuracy * (1 - TOKEN_WEIGHT) + brevity_bonus * TOKEN_WEIGHT

        return raw_accuracy


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_PROMPT = (
    "You are a web browsing agent. Given the user's goal and the current "
    "page's candidate elements, predict the next browser action.\n\n"
    "Output your action as: ACTION: action_type(key=\"value\")\n\n"
    "Valid actions: click(uid=\"...\"), text_input(text=\"...\", uid=\"...\"), "
    "say(speaker=\"navigator\", utterance=\"...\"), scroll(x=0, y=N), "
    "submit(uid=\"...\"), load(url=\"...\")"
)


def evaluate_baseline(
    turns: list[BrowsingTurn],
    prompt: str,
    client: LLMClient,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate a prompt across all turns. Returns (avg_score, details)."""
    details: list[dict[str, Any]] = []
    total = 0.0

    for turn in turns:
        user_message = format_browsing_context(turn)

        response = client.complete(
            system_prompt=prompt,
            user_message=user_message,
            temperature=0.3,
            top_p=0.9,
        )

        if response is None:
            details.append({
                "turn_id": turn.turn_id,
                "category": turn.category,
                "expected": turn.action_raw,
                "predicted": None,
                "score": 0.0,
            })
            continue

        predicted = parse_action(response)
        score, detail = score_action(predicted, turn)
        total += score

        details.append({
            "turn_id": turn.turn_id,
            "category": turn.category,
            "expected": turn.action_raw,
            "predicted_action": predicted[0],
            "predicted_args": predicted[1],
            "intent_match": detail["intent_match"],
            "element_match": detail["element_match"],
            "arg_score": round(detail["arg_score"], 3),
            "score": round(score, 4),
            "response_preview": (response or "")[:150],
        })

    avg = (total / len(turns) * 100.0) if turns else 0.0
    return avg, details


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  MutaGenAI × BrowserGym — WebLINX Action Prediction            ║
    ║  Deep Evolution · Lineage Tracking · GPT-4.1                   ║
    ║  2,300+ demos · click/fill/select/navigate/communicate         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Configuration ─────────────────────────────────────────────────
    MAX_DEMOS = 50             # Number of demonstrations to sample
    MAX_TURNS_PER_DEMO = 5     # Action turns per demo
    EVAL_SAMPLE_SIZE = 20      # Turns per candidate evaluation (speed)

    # ── Load dataset ──────────────────────────────────────────────────
    print("  Loading WebLINX browsing dataset...")
    by_category = load_browsing_turns(
        max_demos=MAX_DEMOS,
        max_turns_per_demo=MAX_TURNS_PER_DEMO,
    )

    total_turns = sum(len(v) for v in by_category.values())
    print(f"    Total turns: {total_turns}")
    for cat, turns in sorted(by_category.items()):
        print(f"      {cat:<14}: {len(turns):3d} turns")

    print(f"    Seed templates: {len(SEED_TEMPLATES)}")
    print(f"    Domain mutations: {len(DOMAIN_MUTATIONS)}")
    print()

    # ── Flatten all turns ─────────────────────────────────────────────
    all_turns: list[BrowsingTurn] = []
    for turns in by_category.values():
        all_turns.extend(turns)

    # ── LLM client ────────────────────────────────────────────────────
    if BACKEND == LLMBackend.OLLAMA:
        base_cfg = PromptEvolverConfig(
            backend=BACKEND,
            ollama_model=MODEL,
            timeout=120.0,
        )
    else:
        base_cfg = PromptEvolverConfig(
            backend=BACKEND,
            azure_deployment=MODEL,
            azure_use_rbac=True,
            timeout=60.0,
        )
    client = LLMClient(base_cfg)

    print("  Checking LLM backend...", end=" ", flush=True)
    if client.is_available():
        print("OK")
    else:
        print("UNAVAILABLE (running in mock mode)")

    # ── Phase 1: Baseline ─────────────────────────────────────────────
    print("\n  Phase 1: Default prompt baseline")
    print("  " + "─" * 56)

    baseline_score, baseline_details = evaluate_baseline(
        all_turns, _DEFAULT_PROMPT, client,
    )
    baseline_tokens = _count_prompt_tokens(_DEFAULT_PROMPT)
    print(f"    Default prompt score: {baseline_score:.1f}%")
    print(f"    Default prompt tokens: {baseline_tokens}")
    if MINIMIZE_TOKENS:
        print(f"    Token optimization: ON (weight={TOKEN_WEIGHT}, max={MAX_PROMPT_TOKENS})")
    else:
        print(f"    Token optimization: OFF")

    # Per-category breakdown
    cat_scores: dict[str, list[float]] = {}
    for d in baseline_details:
        cat_scores.setdefault(d["category"], []).append(d["score"])
    for cat, scores in sorted(cat_scores.items()):
        avg = sum(scores) / len(scores) * 100 if scores else 0
        print(f"      {cat:<14}: {avg:5.1f}% ({len(scores)} turns)")

    intent_hits = sum(1 for d in baseline_details if d.get("intent_match"))
    print(f"    Intent accuracy: {intent_hits}/{len(baseline_details)} "
          f"({intent_hits / max(1, len(baseline_details)) * 100:.1f}%)")

    # ── Phase 2: Deep Evolution ───────────────────────────────────────
    print("\n  Phase 2: Deep Evolution")
    print("  " + "─" * 56)

    config = PromptEvolverConfig(
        iterations=8,
        population_size=10,
        num_islands=3,
        elite_size=5,
        mutation_rate=0.65,
        crossover_rate=0.35,
        eval_sample_size=EVAL_SAMPLE_SIZE,
        adaptive_mutations=True,
        llm_mutation_rate=0.4,
        refine_after_splice=True,
        describe_entities=False,
        backend=BACKEND,
        **({"ollama_model": MODEL} if BACKEND == LLMBackend.OLLAMA else {
            "azure_deployment": MODEL,
            "azure_use_rbac": True,
        }),
        timeout=120.0 if BACKEND == LLMBackend.OLLAMA else 60.0,
    )

    print(f"    Generations:      {config.iterations}")
    print(f"    Population:       {config.population_size} per island")
    print(f"    Islands:          {config.num_islands}")
    print(f"    Elite size:       {config.elite_size}")
    print(f"    Mutation rate:    {config.mutation_rate}")
    print(f"    Crossover rate:   {config.crossover_rate}")
    print(f"    Adaptive:         {config.adaptive_mutations}")
    print(f"    LLM mutation:     {config.llm_mutation_rate}")
    print(f"    Refine splices:   {config.refine_after_splice}")
    print(f"    Eval sample:      {EVAL_SAMPLE_SIZE} turns per candidate")
    print(f"    Total turns:      {len(all_turns)}")
    print(f"    Model:            {MODEL}")
    print()

    # Inject seed templates
    original_seeds = list(_pe._SEED_TEMPLATES)
    _pe._SEED_TEMPLATES = list(SEED_TEMPLATES)

    # Create and run
    t0 = time.perf_counter()
    evolver = BrowserGymEvolver(config, all_turns, client)
    result = evolver.run()
    wall_time = time.perf_counter() - t0

    # Restore original seeds
    _pe._SEED_TEMPLATES = original_seeds

    best_prompt = result.best_prompt

    # ── Phase 3: Full evaluation of evolved prompt ────────────────────
    print(f"\n  Phase 3: Evolved prompt — full evaluation")
    print("  " + "─" * 56)

    evolved_score, evolved_details = evaluate_baseline(
        all_turns, best_prompt, client,
    )
    evolved_tokens = _count_prompt_tokens(best_prompt)
    print(f"    Evolved prompt score: {evolved_score:.1f}%")
    print(f"    Evolved prompt tokens: {evolved_tokens} (baseline: {baseline_tokens}, delta: {evolved_tokens - baseline_tokens:+d})")

    # Per-category breakdown
    evolved_cat: dict[str, list[float]] = {}
    for d in evolved_details:
        evolved_cat.setdefault(d["category"], []).append(d["score"])
    for cat, scores in sorted(evolved_cat.items()):
        avg = sum(scores) / len(scores) * 100 if scores else 0
        baseline_cat_avg = (
            sum(cat_scores.get(cat, [])) / max(1, len(cat_scores.get(cat, []))) * 100
        )
        delta = avg - baseline_cat_avg
        print(f"      {cat:<14}: {avg:5.1f}% (Δ{delta:+.1f}%)")

    evolved_intent = sum(1 for d in evolved_details if d.get("intent_match"))
    print(f"    Intent accuracy: {evolved_intent}/{len(evolved_details)} "
          f"({evolved_intent / max(1, len(evolved_details)) * 100:.1f}%)")

    # ── Results summary ───────────────────────────────────────────────
    print()
    print("  " + "═" * 60)
    print("  RESULTS")
    print("  " + "═" * 60)
    print(f"    Default baseline:   {baseline_score:.1f}%")
    print(f"    Evolved best:       {evolved_score:.1f}%")
    delta = evolved_score - baseline_score
    print(f"    Delta:              {delta:+.1f} pp")
    print(f"    Wall time:          {wall_time:.1f}s")
    print(f"    Total candidates:   {len(result.all_candidates)}")
    print(f"    Temperature:        {result.best_temperature:.4f}")
    print(f"    Top-p:              {result.best_top_p:.4f}")
    print(f"    Prompt tokens:      {baseline_tokens} → {evolved_tokens} ({evolved_tokens - baseline_tokens:+d})")
    if MINIMIZE_TOKENS:
        print(f"    Token weight:       {TOKEN_WEIGHT}")
        token_saving_pct = (1 - evolved_tokens / max(1, baseline_tokens)) * 100
        print(f"    Token saving:       {token_saving_pct:+.1f}%")
    print()
    print("  Best evolved prompt:")
    print("  " + "─" * 60)
    for line in best_prompt.split("\n"):
        print(f"    {line}")
    print("  " + "─" * 60)

    # ── Export lineage ────────────────────────────────────────────────
    lineage = result.lineage_json()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    lineage_path = LOG_DIR / "browsergym_lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)
    print(f"\n  Lineage data saved to {lineage_path}")
    print(f"    ({len(lineage)} candidates — load in docs/lineage_tree.html)")

    # ── Experiment log ────────────────────────────────────────────────
    experiment_log = {
        "experiment": "browsergym_weblinx_deep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL,
        "backend": BACKEND.value,
        "dataset": "WebLINX (McGill-NLP/WebLINX-full)",
        "leaderboard": "https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard",
        "config": {
            "iterations": config.iterations,
            "population_size": config.population_size,
            "num_islands": config.num_islands,
            "elite_size": config.elite_size,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "adaptive_mutations": config.adaptive_mutations,
            "llm_mutation_rate": config.llm_mutation_rate,
            "refine_after_splice": config.refine_after_splice,
            "eval_sample_size": EVAL_SAMPLE_SIZE,
            "max_demos": MAX_DEMOS,
            "max_turns_per_demo": MAX_TURNS_PER_DEMO,
        },
        "scoring": {
            "formula": "0.40 * intent_accuracy + 0.35 * element_f1 + 0.25 * argument_accuracy",
            "intent_accuracy": "action type matches ground truth",
            "element_f1": "target element bid matches ground truth",
            "argument_accuracy": "action arguments (text, URL) match ground truth",
        },
        "turns": len(all_turns),
        "categories": {cat: len(turns) for cat, turns in by_category.items()},
        "seed_templates": len(SEED_TEMPLATES),
        "domain_mutations": len(DOMAIN_MUTATIONS),
        "baseline": {
            "score": round(baseline_score, 2),
            "intent_accuracy": round(intent_hits / max(1, len(baseline_details)) * 100, 2),
            "per_category": {
                cat: round(sum(s) / max(1, len(s)) * 100, 2)
                for cat, s in cat_scores.items()
            },
            "details": baseline_details,
        },
        "evolved": {
            "score": round(evolved_score, 2),
            "intent_accuracy": round(evolved_intent / max(1, len(evolved_details)) * 100, 2),
            "per_category": {
                cat: round(sum(s) / max(1, len(s)) * 100, 2)
                for cat, s in evolved_cat.items()
            },
            "delta": round(delta, 2),
            "temperature": round(result.best_temperature, 4),
            "top_p": round(result.best_top_p, 4),
            "best_prompt": best_prompt,
            "details": evolved_details,
        },
        "wall_time": round(wall_time, 1),
        "history": result.history,
        "lineage_size": len(lineage),
        "token_optimization": {
            "enabled": MINIMIZE_TOKENS,
            "token_weight": TOKEN_WEIGHT,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "baseline_tokens": baseline_tokens,
            "evolved_tokens": evolved_tokens,
            "token_delta": evolved_tokens - baseline_tokens,
        },
    }

    log_path = LOG_DIR / "browsergym_evolution_log.json"
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"  Experiment log saved to {log_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
