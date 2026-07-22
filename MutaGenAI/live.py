"""Live streaming dashboard for prompt evolution (SSE, stdlib-only).

Instead of writing a JSON log and plotting it *after* a run, this module
streams evolution events to a browser **as they happen** over Server-Sent
Events (SSE).  The page shows a live convergence curve and a lineage graph
that grows candidate-by-candidate.

Wire it up by passing the server's :meth:`LiveDashboardServer.publish` as
the evolver's ``on_event`` callback, or use the one-shot helper::

    from MutaGenAI import PromptEvolver, run_with_live_dashboard

    evolver = PromptEvolver(tools, dataset)
    result, server = run_with_live_dashboard(evolver)  # opens a browser
    server.stop()

The server uses only the Python standard library (``http.server``,
``threading``, ``queue``) — no extra dependencies, matching the rest of the
package's optional-import philosophy.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

__all__ = [
    "LiveDashboardServer",
    "run_with_live_dashboard",
    "format_sse",
]


def format_sse(event: dict[str, Any]) -> str:
    """Format an event dict as an SSE ``data:`` frame."""
    return f"data: {json.dumps(event)}\n\n"


class LiveDashboardServer:
    """A tiny SSE server that streams evolution events to a browser.

    Parameters
    ----------
    host :
        Interface to bind.  Default ``127.0.0.1`` (local only).
    port :
        TCP port; ``0`` (default) picks a free ephemeral port — read the
        actual value from :attr:`port` or :attr:`url` after :meth:`start`.

    The server keeps a replayable history of every published event, so a
    browser that connects mid-run immediately receives the full backlog and
    then live updates.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self._requested_port = port
        self._history: list[dict[str, Any]] = []
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> "LiveDashboardServer":
        """Start serving in a background daemon thread."""
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.host, self._requested_port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="MutaGenAI-live", daemon=True
        )
        self._thread.start()
        logger.info("Live dashboard serving at %s", self.url)
        return self

    def stop(self) -> None:
        """Stop the server and release the port."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._requested_port
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # -- pub/sub -------------------------------------------------------------

    def publish(self, event: dict[str, Any]) -> None:
        """Record an event and fan it out to all connected browsers."""
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(event)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of all events published so far."""
        with self._lock:
            return list(self._history)

    def _subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.add(q)
        return q

    def _unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            self._subscribers.discard(q)


def _make_handler(server: LiveDashboardServer):
    """Build a request handler class bound to ``server``."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # noqa: D401 — silence
            return

        def do_GET(self) -> None:  # noqa: N802 — stdlib API name
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._serve_html()
            elif parsed.path == "/events":
                once = "once=1" in (parsed.query or "")
                self._serve_events(once=once)
            else:
                self.send_error(404, "Not found")

        def _serve_html(self) -> None:
            body = _DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self, once: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close" if once else "keep-alive")
            self.end_headers()

            # Replay history first so a late-joining browser is caught up.
            try:
                for event in server.snapshot():
                    self.wfile.write(format_sse(event).encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                return

            if once:
                return  # snapshot mode (used by tests / polling clients)

            q = server._subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=1.0)
                        self.wfile.write(format_sse(event).encode("utf-8"))
                    except queue.Empty:
                        # Heartbeat — also detects a closed connection.
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                server._unsubscribe(q)

    return _Handler


def run_with_live_dashboard(
    evolver: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> tuple[Any, LiveDashboardServer]:
    """Run an evolver while streaming progress to a live browser dashboard.

    Starts a :class:`LiveDashboardServer`, attaches it as the evolver's
    ``on_event`` listener, (optionally) opens a browser, then runs the
    evolution.  The server keeps running after the run so the final state
    stays viewable — call ``server.stop()`` when done.

    Returns the ``(result, server)`` tuple.
    """
    server = LiveDashboardServer(host=host, port=port).start()
    evolver.on_event = server.publish
    if open_browser:
        try:
            webbrowser.open(server.url)
        except Exception as exc:  # noqa: BLE001 — headless envs
            logger.debug("Could not open browser: %s", exc)
    result = evolver.run()
    return result, server


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>MutaGenAI — Live Evolution</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --fg:#e6edf3; --mut:#8b949e;
          --accent:#3fb950; --edge:#30363d; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--edge);
           display:flex; align-items:baseline; gap:16px; }
  h1 { font-size:18px; margin:0; }
  #status { color:var(--mut); font-size:13px; }
  .stats { display:flex; gap:24px; padding:12px 20px; flex-wrap:wrap; }
  .stat { font-size:13px; color:var(--mut); }
  .stat b { display:block; font-size:20px; color:var(--fg); }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:0 20px 20px; }
  .panel { background:var(--panel); border:1px solid var(--edge);
           border-radius:8px; padding:12px; }
  .panel h2 { font-size:13px; margin:0 0 8px; color:var(--mut);
              text-transform:uppercase; letter-spacing:.05em; }
  svg { width:100%; height:320px; display:block; }
  pre { background:#010409; border:1px solid var(--edge); border-radius:6px;
        padding:12px; white-space:pre-wrap; word-break:break-word;
        font-size:12px; max-height:260px; overflow:auto; }
  @media (max-width:900px){ .grid{ grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>MutaGenAI · Live Evolution</h1>
  <span id="status">connecting…</span>
</header>
<div class="stats">
  <div class="stat">Generation<b id="gen">0</b></div>
  <div class="stat">Best score<b id="best">—</b></div>
  <div class="stat">Candidates<b id="cands">0</b></div>
  <div class="stat">LLM calls<b id="calls">0</b></div>
  <div class="stat">Cache hits<b id="hits">0</b></div>
  <div class="stat">Est. cost<b id="cost">—</b></div>
</div>
<div class="grid">
  <div class="panel"><h2>Convergence (best score / generation)</h2>
    <svg id="conv" viewBox="0 0 600 320" preserveAspectRatio="none"></svg></div>
  <div class="panel"><h2>Lineage (x = generation, y = score)</h2>
    <svg id="tree" viewBox="0 0 600 320" preserveAspectRatio="none"></svg></div>
</div>
<div class="grid">
  <div class="panel" style="grid-column:1 / -1"><h2>Best prompt</h2>
    <pre id="prompt">(waiting for run to complete…)</pre></div>
</div>
<script>
const SVGNS = "http://www.w3.org/2000/svg";
const nodes = {};        // hash -> {gen, score, island}
let history = [];        // [[gen, best], ...]
let maxGen = 1;

function el(id){ return document.getElementById(id); }
function svgEl(name, attrs){
  const e = document.createElementNS(SVGNS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function drawConvergence(){
  const svg = el("conv"); svg.innerHTML = "";
  const W=600, H=320, m=30;
  const g = maxGen || 1;
  // axes
  svg.appendChild(svgEl("line",{x1:m,y1:H-m,x2:W-5,y2:H-m,stroke:"#30363d"}));
  svg.appendChild(svgEl("line",{x1:m,y1:5,x2:m,y2:H-m,stroke:"#30363d"}));
  if (history.length === 0) return;
  const pts = history.map(([gen,score])=>{
    const x = m + (g<=1?0:(gen/g))*(W-m-10);
    const y = (H-m) - (score/100)*(H-m-10);
    return x+","+y;
  }).join(" ");
  svg.appendChild(svgEl("polyline",{points:pts,fill:"none",
     stroke:"#3fb950","stroke-width":2}));
  history.forEach(([gen,score])=>{
    const x = m + (g<=1?0:(gen/g))*(W-m-10);
    const y = (H-m) - (score/100)*(H-m-10);
    svg.appendChild(svgEl("circle",{cx:x,cy:y,r:3,fill:"#3fb950"}));
  });
}

const ISLAND_COLORS = ["#58a6ff","#bc8cff","#f778ba","#ffa657","#3fb950","#e3b341"];

function drawTree(){
  const svg = el("tree"); svg.innerHTML = "";
  const W=600, H=320, m=30;
  const g = Math.max(maxGen,1);
  const px = (gen)=> m + (gen/g)*(W-m-10);
  const py = (score)=> (H-m) - (score/100)*(H-m-10);
  svg.appendChild(svgEl("line",{x1:m,y1:H-m,x2:W-5,y2:H-m,stroke:"#30363d"}));
  svg.appendChild(svgEl("line",{x1:m,y1:5,x2:m,y2:H-m,stroke:"#30363d"}));
  // edges first
  for (const h in nodes){
    const n = nodes[h];
    (n.parents||[]).forEach(ph=>{
      const p = nodes[ph];
      if(!p) return;
      svg.appendChild(svgEl("line",{x1:px(p.gen),y1:py(p.score),
        x2:px(n.gen),y2:py(n.score),stroke:"#30363d","stroke-width":1}));
    });
  }
  for (const h in nodes){
    const n = nodes[h];
    const c = ISLAND_COLORS[(n.island>=0?n.island:0)%ISLAND_COLORS.length];
    svg.appendChild(svgEl("circle",{cx:px(n.gen),cy:py(n.score),r:3.2,
      fill:c,"fill-opacity":0.85}));
  }
}

function addNode(c){
  nodes[c.hash] = {gen:c.generation, score:c.score, island:c.island_id,
                   parents:c.parent_hashes};
  if (c.generation > maxGen) maxGen = c.generation;
}

function handle(ev){
  const d = JSON.parse(ev.data);
  if (d.type === "run_start"){
    el("status").textContent =
      `running · ${d.iterations} generations · ${d.backend}`;
    maxGen = d.iterations || 1;
  } else if (d.type === "seed" || d.type === "candidate"){
    addNode(d.candidate);
    el("cands").textContent = Object.keys(nodes).length;
    drawTree();
  } else if (d.type === "generation"){
    history = d.history;
    el("gen").textContent = d.generation;
    el("best").textContent = d.best_score.toFixed(1) + "%";
    el("calls").textContent = d.llm_calls;
    el("hits").textContent = d.cache_hits;
    if (d.estimated_cost_usd != null)
      el("cost").textContent = "$" + d.estimated_cost_usd.toFixed(4);
    drawConvergence();
  } else if (d.type === "run_complete"){
    el("status").textContent =
      `done · best ${d.best_score.toFixed(1)}% · ${d.stop_reason}`;
    el("prompt").textContent = d.best_prompt || "(empty)";
  }
}

const src = new EventSource("/events");
src.onopen = ()=>{ if(el("status").textContent==="connecting…")
  el("status").textContent = "connected"; };
src.onmessage = handle;
src.onerror = ()=>{ el("status").textContent = "stream closed"; };
</script>
</body>
</html>
"""
