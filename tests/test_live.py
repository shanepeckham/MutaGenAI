"""Tests for MutaGenAI.live — streaming SSE dashboard."""
from __future__ import annotations

import json
import urllib.request

import pytest

from MutaGenAI.live import (
    LiveDashboardServer,
    format_sse,
    run_with_live_dashboard,
)


class TestFormatSSE:
    def test_frame_shape(self):
        frame = format_sse({"type": "generation", "best_score": 42.0})
        assert frame.startswith("data: ")
        assert frame.endswith("\n\n")
        payload = json.loads(frame[len("data: "):].strip())
        assert payload["type"] == "generation"


class TestPubSub:
    def test_publish_and_snapshot(self):
        server = LiveDashboardServer()
        server.publish({"type": "a"})
        server.publish({"type": "b"})
        snap = server.snapshot()
        assert [e["type"] for e in snap] == ["a", "b"]
        # Snapshot is a copy — mutating it doesn't affect the server.
        snap.clear()
        assert len(server.snapshot()) == 2

    def test_subscriber_receives_events(self):
        server = LiveDashboardServer()
        q = server._subscribe()
        server.publish({"type": "x"})
        assert q.get_nowait()["type"] == "x"
        server._unsubscribe(q)
        server.publish({"type": "y"})
        assert q.empty()


class TestHttpServer:
    @pytest.fixture
    def server(self):
        srv = LiveDashboardServer().start()
        yield srv
        srv.stop()

    def test_serves_html(self, server):
        html = urllib.request.urlopen(server.url, timeout=3).read().decode()
        assert "Live Evolution" in html
        assert "EventSource" in html

    def test_unknown_path_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(server.url + "nope", timeout=3)
        assert exc.value.code == 404

    def test_events_snapshot_mode(self, server):
        server.publish({"type": "run_start", "iterations": 3})
        server.publish({"type": "generation", "best_score": 50.0})
        body = urllib.request.urlopen(
            server.url + "events?once=1", timeout=3
        ).read().decode()
        # Both published events are replayed as SSE frames then the
        # connection closes (once mode), so the read completes.
        assert body.count("data: ") == 2
        assert "run_start" in body

    def test_port_is_assigned(self, server):
        assert server.port > 0
        assert server.url.startswith("http://127.0.0.1:")


class TestRunWithLiveDashboard:
    def test_runs_and_streams(self):
        from MutaGenAI.prompt_evolver import (
            EvalSample, LLMBackend, PromptEvolver, PromptEvolverConfig, Tool,
        )

        tools = [Tool("get_weather", "w", {"location": "string"})]
        ds = [EvalSample("weather", "get_weather", {"location": "x"})]
        cfg = PromptEvolverConfig(
            iterations=1, population_size=1, num_islands=1, elite_size=1,
            backend=LLMBackend.OLLAMA, ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(tools, ds, cfg, seed=1, verbose=False)
        result, server = run_with_live_dashboard(evolver, open_browser=False)
        try:
            types = [e["type"] for e in server.snapshot()]
            assert "run_start" in types
            assert "run_complete" in types
            assert result.best_prompt
        finally:
            server.stop()
