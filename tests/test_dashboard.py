"""Tests for MutaGenAI.dashboard — visualisation module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from MutaGenAI.dashboard import (
    _has_matplotlib,
    _has_plotly,
    _in_notebook,
    plot_bfcl_evolution,
    plot_tau_bench_evolution,
    plot_xlam_evolution,
    plot_toolbench_evolution,
    plot_apibank_evolution,
    plot_browser_agent_evolution,
)


# ---------------------------------------------------------------------------
# Backend detection tests
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_has_plotly(self):
        # Should return True (plotly is installed in dev deps) or False
        result = _has_plotly()
        assert isinstance(result, bool)

    def test_has_matplotlib(self):
        result = _has_matplotlib()
        assert isinstance(result, bool)

    def test_in_notebook_outside_notebook(self):
        # Running in pytest, not a notebook
        assert _in_notebook() is False

    def test_has_plotly_import_error(self):
        with patch.dict("sys.modules", {"plotly": None}):
            # Force ImportError for plotly
            import MutaGenAI.dashboard as dash
            # Direct call to the function
            assert isinstance(dash._has_plotly(), bool)

    def test_has_matplotlib_import_error(self):
        with patch.dict("sys.modules", {"matplotlib": None}):
            import MutaGenAI.dashboard as dash
            assert isinstance(dash._has_matplotlib(), bool)


# ---------------------------------------------------------------------------
# File-not-found tests for each dashboard
# ---------------------------------------------------------------------------


class TestDashboardFileNotFound:
    def test_bfcl_missing_file(self):
        with pytest.raises(FileNotFoundError, match="BFCL experiment log not found"):
            plot_bfcl_evolution("/nonexistent/path.json")

    def test_tau_bench_missing_file(self):
        with pytest.raises(FileNotFoundError, match="τ-bench experiment log not found"):
            plot_tau_bench_evolution("/nonexistent/path.json")

    def test_xlam_missing_file(self):
        with pytest.raises(FileNotFoundError):
            plot_xlam_evolution("/nonexistent/path.json")

    def test_toolbench_missing_file(self):
        with pytest.raises(FileNotFoundError):
            plot_toolbench_evolution("/nonexistent/path.json")

    def test_apibank_missing_file(self):
        # apibank checks .exists() and returns None instead of raising
        result = plot_apibank_evolution("/nonexistent/path.json")
        assert result is None

    def test_browser_agent_missing_file(self):
        # browser_agent checks .exists() and returns None instead of raising
        result = plot_browser_agent_evolution("/nonexistent/path.json")
        assert result is None


# ---------------------------------------------------------------------------
# Matplotlib fallback rendering tests
# ---------------------------------------------------------------------------


def _write_bfcl_log(path: Path) -> None:
    log = {
        "default_baselines": {"simple": 0.5, "parallel": 0.4},
        "experiments": [
            {
                "category": "simple",
                "algorithm": "island_ga",
                "baseline_score": 0.5,
                "evolved_score": 0.8,
                "improvement": 0.3,
                "best_temperature": 0.5,
                "best_top_p": 0.9,
                "history": [[1, 0.5], [2, 0.6], [3, 0.8]],
            },
            {
                "category": "parallel",
                "algorithm": "island_ga",
                "baseline_score": 0.4,
                "evolved_score": 0.7,
                "improvement": 0.3,
                "best_temperature": 0.6,
                "best_top_p": 0.85,
                "history": [[1, 0.4], [2, 0.5], [3, 0.7]],
            },
        ],
    }
    path.write_text(json.dumps(log))


def _write_tau_log(path: Path) -> None:
    log = {
        "experiments": [
            {
                "domain": "airline",
                "algorithm": "island_ga",
                "baseline_score": 0.3,
                "evolved_score": 0.6,
                "improvement": 0.3,
                "best_temperature": 0.5,
                "best_top_p": 0.9,
                "history": [[1, 0.3], [2, 0.5], [3, 0.6]],
            }
        ],
    }
    path.write_text(json.dumps(log))


def _write_xlam_log(path: Path) -> None:
    log = {
        "experiments": [
            {
                "category": "func_call",
                "algorithm": "island_ga",
                "baseline_score": 0.5,
                "evolved_score": 0.9,
                "improvement": 0.4,
                "best_temperature": 0.5,
                "best_top_p": 0.9,
                "history": [[1, 0.5], [2, 0.7], [3, 0.9]],
            }
        ],
    }
    path.write_text(json.dumps(log))


def _write_toolbench_log(path: Path) -> None:
    log = {
        "experiments": [
            {
                "category": "G1",
                "algorithm": "island_ga",
                "baseline_score": 0.6,
                "evolved_score": 0.9,
                "improvement": 0.3,
                "best_temperature": 0.5,
                "best_top_p": 0.9,
                "history": [[1, 0.6], [2, 0.7], [3, 0.9]],
            }
        ],
    }
    path.write_text(json.dumps(log))


def _write_apibank_log(path: Path) -> None:
    log = {
        "default_baselines": {"Level-1": 0.4},
        "experiments": [
            {
                "category": "Level-1",
                "algorithm": "island_ga",
                "baseline_score": 0.4,
                "evolved_score": 0.75,
                "improvement": 0.35,
                "best_temperature": 0.5,
                "best_top_p": 0.9,
                "history": [[1, 0.4], [2, 0.6], [3, 0.75]],
                "api_name_accuracy": 0.8,
                "param_accuracy": 0.7,
            }
        ],
    }
    path.write_text(json.dumps(log))


def _write_browser_log(path: Path) -> None:
    log = {
        "default_baselines": {"navigation": 0.3},
        "experiments": [
            {
                "category": "navigation",
                "algorithm": "island_ga",
                "baseline_score": 0.3,
                "evolved_score": 0.65,
                "improvement": 0.35,
                "best_temperature": 0.5,
                "best_top_p": 0.9,
                "history": [[1, 0.3], [2, 0.5], [3, 0.65]],
                "func_name_accuracy": 0.7,
                "arg_accuracy": 0.6,
            }
        ],
    }
    path.write_text(json.dumps(log))


class TestBfclMpl:
    def test_renders_matplotlib(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "bfcl_log.json"
        _write_bfcl_log(log_path)
        fig = plot_bfcl_evolution(str(log_path), interactive=False)
        assert fig is not None
        plt.close("all")

    def test_empty_experiments(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "bfcl_empty.json"
        log_path.write_text(json.dumps({
            "default_baselines": {},
            "experiments": [],
        }))
        result = plot_bfcl_evolution(str(log_path), interactive=False)  # noqa: F841
        plt.close("all")


class TestTauMpl:
    def test_renders_matplotlib(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "tau_log.json"
        _write_tau_log(log_path)
        fig = plot_tau_bench_evolution(str(log_path), interactive=False)
        assert fig is not None
        plt.close("all")


class TestXlamMpl:
    def test_renders(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "xlam_log.json"
        _write_xlam_log(log_path)
        with patch("matplotlib.pyplot.show"):
            fig = plot_xlam_evolution(str(log_path))
        assert fig is not None
        plt.close("all")


class TestToolbenchMpl:
    def test_renders(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "toolbench_log.json"
        _write_toolbench_log(log_path)
        with patch("matplotlib.pyplot.show"):
            fig = plot_toolbench_evolution(str(log_path))
        assert fig is not None
        plt.close("all")


class TestApibankMpl:
    def test_renders(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "apibank_log.json"
        _write_apibank_log(log_path)
        with patch("matplotlib.pyplot.show"):
            fig = plot_apibank_evolution(str(log_path))
        assert fig is not None
        plt.close("all")


class TestBrowserAgentMpl:
    def test_renders(self, tmp_path: Path):
        import matplotlib.pyplot as plt
        log_path = tmp_path / "browser_log.json"
        _write_browser_log(log_path)
        with patch("matplotlib.pyplot.show"):
            fig = plot_browser_agent_evolution(str(log_path))
        assert fig is not None
        plt.close("all")
