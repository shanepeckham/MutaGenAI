"""
Interactive visualisation for Prompture prompt evolution experiments.

Provides Plotly (interactive) dashboards with Matplotlib fallbacks for
viewing prompt evolution results across benchmark suites.

Quick start::

    from prompture.dashboard import plot_bfcl_evolution
    plot_bfcl_evolution("bfcl_experiment_log.json")

Available dashboards::

    plot_bfcl_evolution          # BFCL V4 function-calling
    plot_tau_bench_evolution     # τ-bench agent domains
    plot_xlam_evolution          # xLAM / APIGen function-calling
    plot_toolbench_evolution     # ToolBench tool-use
    plot_apibank_evolution       # API-Bank API selection + params
    plot_browser_agent_evolution # Browser agent navigation
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _has_plotly() -> bool:
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ipywidgets() -> bool:
    try:
        import ipywidgets  # noqa: F401
        return True
    except ImportError:
        return False


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# BFCL V4 Prompt Evolution Dashboard
# ---------------------------------------------------------------------------


def plot_bfcl_evolution(
    log_path: str = "bfcl_experiment_log.json",
    interactive: bool | None = None,
) -> Any:
    """Plot BFCL prompt evolution results from an experiment log.

    Shows a multi-panel view:
    - Convergence curves per category
    - Baseline vs evolved comparison bar chart
    - Evolution history per experiment

    Parameters
    ----------
    log_path :
        Path to the ``bfcl_experiment_log.json`` produced by the BFCL
        cookbook recipe.
    interactive :
        Force Plotly (True) or Matplotlib (False). ``None`` auto-detects.
    """
    from pathlib import Path

    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"BFCL experiment log not found at {log_path}. "
            "Run the BFCL cookbook recipe first."
        )

    with open(path) as f:
        log = json.load(f)

    use_plotly = (
        interactive
        if interactive is not None
        else (_has_plotly() and _in_notebook())
    )

    if use_plotly:
        return _bfcl_plotly(log)
    return _bfcl_mpl(log)


def _bfcl_plotly(log: dict[str, Any]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    n_exp = len(experiments)
    if n_exp == 0:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence per Experiment",
            "Baseline → Evolved Score",
            "Score by Category",
            "Temperature & Top-p",
        ),
    )

    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # Panel 1: convergence curves
    for i, exp in enumerate(experiments):
        history = exp.get("history", [])
        if history:
            gens = [h[0] for h in history]
            scores = [h[1] for h in history]
            label = f"{exp['category']} ({exp['algorithm']})"
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=scores,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5),
                ),
                row=1,
                col=1,
            )

    # Panel 2: baseline vs evolved bars
    categories = [f"{e['category']}\n({e['algorithm']})" for e in experiments]
    baselines = [e["baseline_score"] for e in experiments]
    evolved = [e["evolved_score"] for e in experiments]

    fig.add_trace(
        go.Bar(
            name="Seed Baseline",
            x=categories,
            y=baselines,
            marker_color="#90CAF9",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Evolved",
            x=categories,
            y=evolved,
            marker_color="#1565C0",
        ),
        row=1,
        col=2,
    )

    # Panel 3: default vs evolved by category (unique cats only)
    seen: dict[str, dict[str, float]] = {}
    for exp in experiments:
        cat = exp["category"]
        if cat not in seen or exp["evolved_score"] > seen[cat]["evolved"]:
            seen[cat] = {
                "default": defaults.get(cat, 0.0),
                "evolved": exp["evolved_score"],
            }
    cats = list(seen.keys())
    fig.add_trace(
        go.Bar(
            name="BFCL Default Prompt",
            x=cats,
            y=[seen[c]["default"] for c in cats],
            marker_color="#FFCC80",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Best Evolved",
            x=cats,
            y=[seen[c]["evolved"] for c in cats],
            marker_color="#E65100",
        ),
        row=2,
        col=1,
    )

    # Panel 4: temperature and top-p scatter
    for i, exp in enumerate(experiments):
        label = f"{exp['category']} ({exp['algorithm']})"
        fig.add_trace(
            go.Scatter(
                x=[exp["best_temperature"]],
                y=[exp["best_top_p"]],
                mode="markers+text",
                text=[f"{exp['evolved_score']:.0f}%"],
                textposition="top center",
                name=label,
                marker=dict(
                    size=exp["evolved_score"] / 5,
                    color=colors[i % len(colors)],
                ),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    fig.update_yaxes(title_text="Top-p", row=2, col=2)

    fig.update_layout(
        title="BFCL V4 Prompt Evolution Dashboard",
        height=700,
        template="plotly_white",
        barmode="group",
    )
    fig.show()
    return fig


def _bfcl_mpl(log: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # Panel 1: convergence curves
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        history = exp.get("history", [])
        if history:
            gens = [h[0] for h in history]
            scores = [h[1] for h in history]
            label = f"{exp['category']} ({exp['algorithm']})"
            ax1.plot(
                gens,
                scores,
                "-o",
                label=label,
                color=colors[i % len(colors)],
                markersize=4,
            )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Convergence per Experiment")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: baseline vs evolved
    ax2 = axes[0, 1]
    labels = [f"{e['category']}\n({e['algorithm']})" for e in experiments]
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(
        x - w / 2,
        [e["baseline_score"] for e in experiments],
        w,
        label="Seed Baseline",
        color="#90CAF9",
    )
    ax2.bar(
        x + w / 2,
        [e["evolved_score"] for e in experiments],
        w,
        label="Evolved",
        color="#1565C0",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7, rotation=15)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Baseline → Evolved")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: default prompt vs best evolved per category
    ax3 = axes[1, 0]
    seen: dict[str, dict[str, float]] = {}
    for exp in experiments:
        cat = exp["category"]
        if cat not in seen or exp["evolved_score"] > seen[cat]["evolved"]:
            seen[cat] = {
                "default": defaults.get(cat, 0.0),
                "evolved": exp["evolved_score"],
            }
    cats = list(seen.keys())
    x3 = np.arange(len(cats))
    ax3.bar(
        x3 - w / 2,
        [seen[c]["default"] for c in cats],
        w,
        label="BFCL Default",
        color="#FFCC80",
    )
    ax3.bar(
        x3 + w / 2,
        [seen[c]["evolved"] for c in cats],
        w,
        label="Evolved",
        color="#E65100",
    )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats, fontsize=8)
    ax3.set_ylabel("Score (%)")
    ax3.set_title("BFCL Default vs Evolved")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: temperature & top-p scatter
    ax4 = axes[1, 1]
    for i, exp in enumerate(experiments):
        label = f"{exp['category']} ({exp['algorithm']})"
        ax4.scatter(
            exp["best_temperature"],
            exp["best_top_p"],
            s=exp["evolved_score"] * 2,
            c=colors[i % len(colors)],
            label=label,
            edgecolors="k",
            alpha=0.8,
        )
        ax4.annotate(
            f"{exp['evolved_score']:.0f}%",
            (exp["best_temperature"], exp["best_top_p"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Top-p")
    ax4.set_title("Optimal Parameters")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "BFCL V4 Prompt Evolution Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# τ-Bench Prompt Evolution Dashboard
# ---------------------------------------------------------------------------


def plot_tau_bench_evolution(
    log_path: str = "tau_bench_experiment_log.json",
    interactive: bool | None = None,
) -> Any:
    """Plot τ-bench prompt evolution results from an experiment log.

    Shows a multi-panel view:
    - Convergence curves per domain/algorithm
    - Baseline vs evolved comparison bar chart
    - Sub-score breakdown (tool, info, assertion)
    - Temperature & top-p scatter

    Parameters
    ----------
    log_path :
        Path to ``tau_bench_experiment_log.json`` produced by the τ-bench
        cookbook recipe.
    interactive :
        Force Plotly (True) or Matplotlib (False). ``None`` auto-detects.
    """
    from pathlib import Path

    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"τ-bench experiment log not found at {log_path}. "
            "Run the τ-bench cookbook recipe first."
        )

    with open(path) as f:
        log = json.load(f)

    use_plotly = (
        interactive
        if interactive is not None
        else (_has_plotly() and _in_notebook())
    )

    if use_plotly:
        return _tau_plotly(log)
    return _tau_mpl(log)


def _tau_plotly(log: dict[str, Any]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence per Experiment",
            "Baseline → Evolved Score",
            "Sub-score Breakdown",
            "Temperature & Top-p",
        ),
    )

    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # Panel 1: convergence curves
    for i, exp in enumerate(experiments):
        history = exp.get("history", [])
        if history:
            gens = [h[0] for h in history]
            scores = [h[1] for h in history]
            label = f"{exp['domain']} ({exp['algorithm']})"
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=scores,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5),
                ),
                row=1,
                col=1,
            )

    # Panel 2: baseline vs evolved bars
    labels = [f"{e['domain']}\n({e['algorithm']})" for e in experiments]
    baselines = [e["baseline_score"] for e in experiments]
    evolved = [e["evolved_score"] for e in experiments]

    fig.add_trace(
        go.Bar(
            name="Seed Baseline",
            x=labels,
            y=baselines,
            marker_color="#90CAF9",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Evolved",
            x=labels,
            y=evolved,
            marker_color="#1565C0",
        ),
        row=1,
        col=2,
    )

    # Panel 3: sub-score breakdown
    sub_labels: list[str] = []
    tool_scores: list[float] = []
    info_scores: list[float] = []
    assertion_scores: list[float] = []
    for exp in experiments:
        sub = exp.get("sub_scores", {})
        sub_labels.append(f"{exp['domain']}\n({exp['algorithm']})")
        tool_scores.append(sub.get("tool_score", 0.0))
        info_scores.append(sub.get("info_score", 0.0))
        assertion_scores.append(sub.get("assertion_score", 0.0))

    fig.add_trace(
        go.Bar(name="Tool Call", x=sub_labels, y=tool_scores, marker_color="#4CAF50"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(name="Info Comm.", x=sub_labels, y=info_scores, marker_color="#FF9800"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(name="Assertions", x=sub_labels, y=assertion_scores, marker_color="#9C27B0"),
        row=2, col=1,
    )

    # Panel 4: temperature and top-p scatter
    for i, exp in enumerate(experiments):
        label = f"{exp['domain']} ({exp['algorithm']})"
        fig.add_trace(
            go.Scatter(
                x=[exp["best_temperature"]],
                y=[exp["best_top_p"]],
                mode="markers+text",
                text=[f"{exp['evolved_score']:.0f}%"],
                textposition="top center",
                name=label,
                marker=dict(
                    size=max(8, exp["evolved_score"] / 5),
                    color=colors[i % len(colors)],
                ),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    fig.update_yaxes(title_text="Top-p", row=2, col=2)

    fig.update_layout(
        title="τ-Bench Prompt Evolution Dashboard",
        height=700,
        template="plotly_white",
        barmode="group",
    )
    fig.show()
    return fig


def _tau_mpl(log: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # Panel 1: convergence curves
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        history = exp.get("history", [])
        if history:
            gens = [h[0] for h in history]
            scores = [h[1] for h in history]
            label = f"{exp['domain']} ({exp['algorithm']})"
            ax1.plot(
                gens, scores, "-o",
                label=label, color=colors[i % len(colors)], markersize=4,
            )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Convergence per Experiment")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: baseline vs evolved
    ax2 = axes[0, 1]
    labels = [f"{e['domain']}\n({e['algorithm']})" for e in experiments]
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(
        x - w / 2,
        [e["baseline_score"] for e in experiments],
        w, label="Seed Baseline", color="#90CAF9",
    )
    ax2.bar(
        x + w / 2,
        [e["evolved_score"] for e in experiments],
        w, label="Evolved", color="#1565C0",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7, rotation=15)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Baseline → Evolved")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: sub-score breakdown
    ax3 = axes[1, 0]
    sub_labels: list[str] = []
    tool_scores: list[float] = []
    info_scores: list[float] = []
    assertion_scores: list[float] = []
    for exp in experiments:
        sub = exp.get("sub_scores", {})
        sub_labels.append(f"{exp['domain']}\n({exp['algorithm']})")
        tool_scores.append(sub.get("tool_score", 0.0))
        info_scores.append(sub.get("info_score", 0.0))
        assertion_scores.append(sub.get("assertion_score", 0.0))

    x3 = np.arange(len(sub_labels))
    bar_w = 0.25
    ax3.bar(x3 - bar_w, tool_scores, bar_w, label="Tool Call", color="#4CAF50")
    ax3.bar(x3, info_scores, bar_w, label="Info Comm.", color="#FF9800")
    ax3.bar(x3 + bar_w, assertion_scores, bar_w, label="Assertions", color="#9C27B0")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(sub_labels, fontsize=7)
    ax3.set_ylabel("Score (%)")
    ax3.set_title("Sub-score Breakdown")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: temperature & top-p scatter
    ax4 = axes[1, 1]
    for i, exp in enumerate(experiments):
        label = f"{exp['domain']} ({exp['algorithm']})"
        ax4.scatter(
            exp["best_temperature"], exp["best_top_p"],
            s=max(40, exp["evolved_score"] * 2),
            c=colors[i % len(colors)],
            label=label, edgecolors="k", alpha=0.8,
        )
        ax4.annotate(
            f"{exp['evolved_score']:.0f}%",
            (exp["best_temperature"], exp["best_top_p"]),
            textcoords="offset points", xytext=(5, 5), fontsize=7,
        )
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Top-p")
    ax4.set_title("Optimal Parameters")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "τ-Bench Prompt Evolution Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# xLAM / APIGen Function-Calling Prompt Evolution Dashboard
# ──────────────────────────────────────────────────────────────────────────


def plot_xlam_evolution(log_path: str = "xlam_experiment_log.json") -> Any:
    """Visualise xLAM prompt-evolution results.

    Reads the JSON log saved by ``prompt_evolution_xlam.py`` and renders
    a 4-panel dashboard via Plotly (interactive) with a Matplotlib
    fallback.

    Parameters
    ----------
    log_path :
        Path to the JSON experiment log.

    Returns
    -------
    Plotly Figure or Matplotlib Figure
    """
    with open(log_path) as f:
        log = json.load(f)

    try:
        return _xlam_plotly(log)
    except ImportError:
        return _xlam_mpl(log)


def _xlam_plotly(log: dict[str, Any]) -> Any:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Convergence Curves",
            "Baseline vs Evolved (by Category)",
            "Score Distribution by Backend",
            "Temperature / Top-p Landscape",
        ],
    )

    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence curves
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']} ({exp['backend']})"
        fig.add_trace(
            go.Scatter(
                x=gens, y=scores, name=label,
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)]),
            ),
            row=1, col=1,
        )

    # P2: grouped bar — default vs baseline vs evolved
    cats = []
    default_vals = []
    baseline_vals = []
    evolved_vals = []
    for exp in experiments:
        cat_label = f"{exp['category']}/{exp['algorithm'][:3]}/{exp['backend'][:3]}"
        cats.append(cat_label)
        default_vals.append(defaults.get(exp["category"], 0))
        baseline_vals.append(exp["baseline_score"])
        evolved_vals.append(exp["evolved_score"])

    fig.add_trace(
        go.Bar(name="Default", x=cats, y=default_vals, marker_color="#BBDEFB"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(name="Baseline", x=cats, y=baseline_vals, marker_color="#64B5F6"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(name="Evolved", x=cats, y=evolved_vals, marker_color="#1565C0"),
        row=1, col=2,
    )

    # P3: box / violin by backend
    for backend in sorted({e["backend"] for e in experiments}):
        scores = [e["evolved_score"] for e in experiments if e["backend"] == backend]
        fig.add_trace(
            go.Box(y=scores, name=backend, boxmean=True),
            row=2, col=1,
        )

    # P4: temp vs top-p scatter
    for i, exp in enumerate(experiments):
        fig.add_trace(
            go.Scatter(
                x=[exp["best_temperature"]],
                y=[exp["best_top_p"]],
                mode="markers",
                marker=dict(
                    size=exp["evolved_score"] / 5,
                    color=colors[i % len(colors)],
                ),
                name=f"{exp['category']} ({exp['evolved_score']:.0f}%)",
                showlegend=True,
            ),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=2)
    fig.update_yaxes(title_text="Score (%)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    fig.update_yaxes(title_text="Top-p", row=2, col=2)

    fig.update_layout(
        title="xLAM / APIGen Prompt Evolution Dashboard",
        height=700,
        template="plotly_white",
        barmode="group",
    )
    fig.show()
    return fig


def _xlam_mpl(log: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']}/{exp['algorithm'][:3]}"
        ax1.plot(gens, scores, "-o", color=colors[i % len(colors)],
                 label=label, markersize=4)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Convergence Curves")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # P2: baseline vs evolved
    ax2 = axes[0, 1]
    labels = [f"{e['category'][:6]}/{e['algorithm'][:3]}" for e in experiments]
    x = range(len(labels))
    base = [e["baseline_score"] for e in experiments]
    evolved = [e["evolved_score"] for e in experiments]
    w = 0.35
    ax2.bar([xi - w / 2 for xi in x], base, w, label="Baseline", color="#64B5F6")
    ax2.bar([xi + w / 2 for xi in x], evolved, w, label="Evolved", color="#1565C0")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Baseline vs Evolved")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # P3: delta by category
    ax3 = axes[1, 0]
    deltas = [e["evolved_score"] - e["baseline_score"] for e in experiments]
    bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    ax3.bar(list(x), deltas, color=bar_colors)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Δ Score (%)")
    ax3.set_title("Improvement (Δ)")
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # P4: parameter landscape
    ax4 = axes[1, 1]
    for i, exp in enumerate(experiments):
        ax4.scatter(
            exp["best_temperature"],
            exp["best_top_p"],
            s=exp["evolved_score"] * 2,
            c=colors[i % len(colors)],
            label=f"{exp['category'][:6]} ({exp['evolved_score']:.0f}%)",
            alpha=0.7,
        )
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Top-p")
    ax4.set_title("Optimal Parameters")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "xLAM / APIGen Prompt Evolution Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# ToolBench prompt-evolution dashboard
# ─────────────────────────────────────────────────────────────────────────


def plot_toolbench_evolution(
    log_path: str = "toolbench_experiment_log.json",
) -> Any:
    """Visualise ToolBench prompt-evolution results.

    Reads the JSON log saved by ``prompt_evolution_toolbench.py`` and
    renders a 4-panel dashboard via Plotly (interactive) with a
    Matplotlib fallback.

    Parameters
    ----------
    log_path :
        Path to the JSON experiment log.

    Returns
    -------
    Plotly Figure or Matplotlib Figure
    """
    with open(log_path) as f:
        log = json.load(f)

    try:
        return _toolbench_plotly(log)
    except ImportError:
        return _toolbench_mpl(log)


def _toolbench_plotly(log: dict[str, Any]) -> Any:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Convergence Curves",
            "Baseline vs Evolved (by Split)",
            "Score Distribution by Backend",
            "Temperature / Top-p Landscape",
        ],
    )

    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence curves
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']} ({exp['backend']})"
        fig.add_trace(
            go.Scatter(
                x=gens, y=scores, name=label,
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)]),
            ),
            row=1, col=1,
        )

    # P2: grouped bar — default vs baseline vs evolved
    cats = []
    default_vals = []
    baseline_vals = []
    evolved_vals = []
    for exp in experiments:
        cat_label = (
            f"{exp['category']}/{exp['algorithm'][:3]}/{exp['backend'][:3]}"
        )
        cats.append(cat_label)
        default_vals.append(defaults.get(exp["category"], 0))
        baseline_vals.append(exp["baseline_score"])
        evolved_vals.append(exp["evolved_score"])

    fig.add_trace(
        go.Bar(
            name="Default", x=cats, y=default_vals, marker_color="#BBDEFB",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Baseline", x=cats, y=baseline_vals, marker_color="#64B5F6",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Evolved", x=cats, y=evolved_vals, marker_color="#1565C0",
        ),
        row=1, col=2,
    )

    # P3: box / violin by backend
    for backend in sorted({e["backend"] for e in experiments}):
        scores = [
            e["evolved_score"]
            for e in experiments
            if e["backend"] == backend
        ]
        fig.add_trace(
            go.Box(y=scores, name=backend, boxmean=True),
            row=2, col=1,
        )

    # P4: temp vs top-p scatter
    for i, exp in enumerate(experiments):
        fig.add_trace(
            go.Scatter(
                x=[exp["best_temperature"]],
                y=[exp["best_top_p"]],
                mode="markers",
                marker=dict(
                    size=exp["evolved_score"] / 5,
                    color=colors[i % len(colors)],
                ),
                name=f"{exp['category'][:10]} ({exp['evolved_score']:.0f}%)",
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title_text="ToolBench Prompt Evolution Dashboard",
        height=800,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    fig.update_yaxes(title_text="Top-p", row=2, col=2)

    return fig


def _toolbench_mpl(log: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']}"
        ax1.plot(
            gens, scores, "o-", color=colors[i % len(colors)],
            label=label, markersize=4,
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Convergence Curves")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # P2: baseline vs evolved
    ax2 = axes[0, 1]
    labels = [
        f"{e['category'][:8]}/{e['algorithm'][:3]}" for e in experiments
    ]
    x = range(len(labels))
    base = [e["baseline_score"] for e in experiments]
    evolved = [e["evolved_score"] for e in experiments]
    w = 0.35
    ax2.bar(
        [xi - w / 2 for xi in x], base, w,
        label="Baseline", color="#64B5F6",
    )
    ax2.bar(
        [xi + w / 2 for xi in x], evolved, w,
        label="Evolved", color="#1565C0",
    )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Baseline vs Evolved")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # P3: delta by category
    ax3 = axes[1, 0]
    deltas = [e["evolved_score"] - e["baseline_score"] for e in experiments]
    bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    ax3.bar(list(x), deltas, color=bar_colors)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Δ Score (%)")
    ax3.set_title("Improvement (Δ)")
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # P4: parameter landscape
    ax4 = axes[1, 1]
    for i, exp in enumerate(experiments):
        ax4.scatter(
            exp["best_temperature"],
            exp["best_top_p"],
            s=exp["evolved_score"] * 2,
            c=colors[i % len(colors)],
            label=f"{exp['category'][:8]} ({exp['evolved_score']:.0f}%)",
            alpha=0.7,
        )
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Top-p")
    ax4.set_title("Optimal Parameters")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "ToolBench Prompt Evolution Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# API-Bank Prompt Evolution Dashboard
# ─────────────────────────────────────────────────────────────────────────


def plot_apibank_evolution(log_path: str = "apibank_experiment_log.json") -> Any:
    """Render the API-Bank prompt-evolution dashboard.

    Accepts a path to the JSON experiment log produced by
    ``prompt_evolution_apibank.py``.  Uses Plotly when available,
    falling back to Matplotlib.
    """
    from pathlib import Path

    p = Path(log_path)
    if not p.exists():
        print(f"  No API-Bank experiment log at {p}")
        return None

    with open(p) as f:
        log = json.load(f)

    try:
        return _apibank_plotly(log)
    except Exception:
        return _apibank_mpl(log)


def _apibank_plotly(log: dict[str, Any]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence Curves",
            "Default vs Baseline vs Evolved",
            "API Name vs Param Accuracy",
            "Temperature vs Top-p Landscape",
        ),
    )

    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']} ({exp['backend']})"
        fig.add_trace(
            go.Scatter(
                x=gens, y=scores, mode="lines+markers",
                name=label, line=dict(color=colors[i % len(colors)]),
                marker=dict(size=5),
            ),
            row=1, col=1,
        )

    # P2: grouped bars — default / baseline / evolved
    categories = []
    dflt_vals = []
    base_vals = []
    evo_vals = []
    for exp in experiments:
        lbl = f"{exp['category']}/{exp['algorithm'][:3]}/{exp['backend'][:5]}"
        categories.append(lbl)
        dflt_vals.append(defaults.get(exp["category"], 0))
        base_vals.append(exp["baseline_score"])
        evo_vals.append(exp["evolved_score"])

    fig.add_trace(
        go.Bar(x=categories, y=dflt_vals, name="Default", marker_color="#B0BEC5"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=categories, y=base_vals, name="Baseline", marker_color="#64B5F6"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=categories, y=evo_vals, name="Evolved", marker_color="#1565C0"),
        row=1, col=2,
    )
    fig.update_layout(barmode="group")

    # P3: API name vs param accuracy (grouped bar)
    name_accs = [exp.get("api_name_accuracy", 0) for exp in experiments]
    param_accs = [exp.get("param_accuracy", 0) for exp in experiments]
    fig.add_trace(
        go.Bar(x=categories, y=name_accs, name="API Name Acc", marker_color="#4CAF50"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=categories, y=param_accs, name="Param Acc", marker_color="#FF9800"),
        row=2, col=1,
    )

    # P4: temperature vs top-p
    for i, exp in enumerate(experiments):
        fig.add_trace(
            go.Scatter(
                x=[exp["best_temperature"]],
                y=[exp["best_top_p"]],
                mode="markers",
                marker=dict(
                    size=max(8, exp["evolved_score"] / 5),
                    color=colors[i % len(colors)],
                    opacity=0.7,
                ),
                name=f"{exp['category']} ({exp['evolved_score']:.0f}%)",
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title_text="API-Bank Prompt Evolution Dashboard",
        height=800,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    fig.update_yaxes(title_text="Top-p", row=2, col=2)

    return fig


def _apibank_mpl(log: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']}"
        ax1.plot(
            gens, scores, "o-", color=colors[i % len(colors)],
            label=label, markersize=4,
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Convergence Curves")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # P2: baseline vs evolved
    ax2 = axes[0, 1]
    labels = [
        f"{e['category'][:8]}/{e['algorithm'][:3]}" for e in experiments
    ]
    x = range(len(labels))
    base = [e["baseline_score"] for e in experiments]
    evolved = [e["evolved_score"] for e in experiments]
    w = 0.35
    ax2.bar(
        [xi - w / 2 for xi in x], base, w,
        label="Baseline", color="#64B5F6",
    )
    ax2.bar(
        [xi + w / 2 for xi in x], evolved, w,
        label="Evolved", color="#1565C0",
    )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Baseline vs Evolved")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # P3: API name accuracy vs param accuracy
    ax3 = axes[1, 0]
    name_accs = [exp.get("api_name_accuracy", 0) for exp in experiments]
    param_accs = [exp.get("param_accuracy", 0) for exp in experiments]
    w2 = 0.35
    ax3.bar(
        [xi - w2 / 2 for xi in x], name_accs, w2,
        label="API Name Acc", color="#4CAF50",
    )
    ax3.bar(
        [xi + w2 / 2 for xi in x], param_accs, w2,
        label="Param Acc", color="#FF9800",
    )
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("API Name vs Parameter Accuracy")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="y")

    # P4: parameter landscape
    ax4 = axes[1, 1]
    for i, exp in enumerate(experiments):
        ax4.scatter(
            exp["best_temperature"],
            exp["best_top_p"],
            s=exp["evolved_score"] * 2,
            c=colors[i % len(colors)],
            label=f"{exp['category'][:8]} ({exp['evolved_score']:.0f}%)",
            alpha=0.7,
        )
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Top-p")
    ax4.set_title("Optimal Parameters")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "API-Bank Prompt Evolution Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return fig


# ── Browser Agent dashboard ──────────────────────────────────────────


def plot_browser_agent_evolution(
    log_path: str = "browser_agent_experiment_log.json",
) -> Any:
    """Render the Browser Agent prompt-evolution dashboard.

    Accepts a path to the JSON experiment log produced by
    ``prompt_evolution_browser_agent.py``.  Uses Plotly when available,
    falling back to Matplotlib.
    """
    from pathlib import Path

    p = Path(log_path)
    if not p.exists():
        print(f"  No Browser Agent experiment log at {p}")
        return None

    with open(p) as f:
        log = json.load(f)

    try:
        return _browser_agent_plotly(log)
    except Exception:
        return _browser_agent_mpl(log)


def _browser_agent_plotly(log: dict[str, Any]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence Curves",
            "Default vs Baseline vs Evolved",
            "Func Name vs Argument Accuracy",
            "Temperature vs Top-p Landscape",
        ),
    )

    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']} ({exp['backend']})"
        fig.add_trace(
            go.Scatter(
                x=gens, y=scores, mode="lines+markers",
                name=label, line=dict(color=colors[i % len(colors)]),
                marker=dict(size=5),
            ),
            row=1, col=1,
        )

    # P2: grouped bars — default / baseline / evolved
    categories = []
    dflt_vals = []
    base_vals = []
    evo_vals = []
    for exp in experiments:
        lbl = f"{exp['category']}/{exp['algorithm'][:3]}/{exp['backend'][:5]}"
        categories.append(lbl)
        dflt_vals.append(defaults.get(exp["category"], 0))
        base_vals.append(exp["baseline_score"])
        evo_vals.append(exp["evolved_score"])

    fig.add_trace(
        go.Bar(x=categories, y=dflt_vals, name="Default", marker_color="#B0BEC5"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=categories, y=base_vals, name="Baseline", marker_color="#64B5F6"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=categories, y=evo_vals, name="Evolved", marker_color="#1565C0"),
        row=1, col=2,
    )
    fig.update_layout(barmode="group")

    # P3: Func name vs argument accuracy (grouped bar)
    name_accs = [exp.get("func_name_accuracy", 0) for exp in experiments]
    arg_accs = [exp.get("arg_accuracy", 0) for exp in experiments]
    fig.add_trace(
        go.Bar(
            x=categories, y=name_accs,
            name="Func Name Acc", marker_color="#4CAF50",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=categories, y=arg_accs,
            name="Arg Acc", marker_color="#FF9800",
        ),
        row=2, col=1,
    )

    # P4: temperature vs top-p
    for i, exp in enumerate(experiments):
        fig.add_trace(
            go.Scatter(
                x=[exp["best_temperature"]],
                y=[exp["best_top_p"]],
                mode="markers",
                marker=dict(
                    size=max(8, exp["evolved_score"] / 5),
                    color=colors[i % len(colors)],
                    opacity=0.7,
                ),
                name=f"{exp['category']} ({exp['evolved_score']:.0f}%)",
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title_text="Browser Agent Prompt Evolution Dashboard",
        height=800,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_yaxes(title_text="Score (%)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature", row=2, col=2)
    fig.update_yaxes(title_text="Top-p", row=2, col=2)

    return fig


def _browser_agent_mpl(log: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    experiments = log.get("experiments", [])
    defaults = log.get("default_baselines", {})
    if not experiments:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = [
        "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]

    # P1: convergence
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        hist = exp.get("history", [])
        if not hist:
            continue
        gens = [h[0] for h in hist]
        scores = [h[1] for h in hist]
        label = f"{exp['category']} / {exp['algorithm']}"
        ax1.plot(
            gens, scores, "o-", color=colors[i % len(colors)],
            label=label, markersize=4,
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Convergence Curves")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # P2: baseline vs evolved
    ax2 = axes[0, 1]
    labels = [
        f"{e['category'][:8]}/{e['algorithm'][:3]}" for e in experiments
    ]
    x = range(len(labels))
    base = [e["baseline_score"] for e in experiments]
    evolved = [e["evolved_score"] for e in experiments]
    w = 0.35
    ax2.bar(
        [xi - w / 2 for xi in x], base, w,
        label="Baseline", color="#64B5F6",
    )
    ax2.bar(
        [xi + w / 2 for xi in x], evolved, w,
        label="Evolved", color="#1565C0",
    )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Baseline vs Evolved")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # P3: func name vs arg accuracy
    ax3 = axes[1, 0]
    name_accs = [exp.get("func_name_accuracy", 0) for exp in experiments]
    arg_accs = [exp.get("arg_accuracy", 0) for exp in experiments]
    w2 = 0.35
    ax3.bar(
        [xi - w2 / 2 for xi in x], name_accs, w2,
        label="Func Name Acc", color="#4CAF50",
    )
    ax3.bar(
        [xi + w2 / 2 for xi in x], arg_accs, w2,
        label="Arg Acc", color="#FF9800",
    )
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Function Name vs Argument Accuracy")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="y")

    # P4: parameter landscape
    ax4 = axes[1, 1]
    for i, exp in enumerate(experiments):
        ax4.scatter(
            exp["best_temperature"],
            exp["best_top_p"],
            s=exp["evolved_score"] * 2,
            c=colors[i % len(colors)],
            label=f"{exp['category'][:8]} ({exp['evolved_score']:.0f}%)",
            alpha=0.7,
        )
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Top-p")
    ax4.set_title("Optimal Parameters")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Browser Agent Prompt Evolution Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return fig
