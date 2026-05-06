"use strict";
/**
 * EvolvePanel — webview panel for real-time evolution visualisation.
 *
 * Shows:
 *  - Live score chart (generation × best score)
 *  - Lineage tree (parent → child with operation labels)
 *  - Status bar with generation progress
 *  - Best prompt preview
 *  - Early stop / manual stop controls
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.EvolvePanel = void 0;
const vscode = __importStar(require("vscode"));
class EvolvePanel {
    /** Wire up a logger (call before createOrShow). */
    static setLogger(fn) {
        EvolvePanel.logFn = fn;
    }
    _log(msg) {
        EvolvePanel.logFn(`[EvolvePanel] ${msg}`);
    }
    constructor(panel, extensionUri) {
        this.disposables = [];
        this.webviewReady = false;
        this.eventBuffer = [];
        this.panel = panel;
        this.extensionUri = extensionUri;
        // Ensure scripts are enabled before setting HTML
        this.panel.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(extensionUri, "media"),
            ],
        };
        this._log(`Constructor — enableScripts=${this.panel.webview.options.enableScripts}`);
        this.panel.webview.html = this.getHtml();
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        // Listen for the "ready" signal from the webview
        this.panel.webview.onDidReceiveMessage((msg) => {
            if (msg.command === "ready") {
                this._log(`Webview sent 'ready' — flushing ${this.eventBuffer.length} buffered events`);
                if (this.fallbackTimer) {
                    clearTimeout(this.fallbackTimer);
                    this.fallbackTimer = undefined;
                }
                this.webviewReady = true;
                // Acknowledge the ready signal so webview stops polling
                this.panel.webview.postMessage({ type: "_readyAck" }).then((ok) => this._log(`_readyAck delivered=${ok}`));
                this._flushBuffer();
            }
        }, null, this.disposables);
        // Fallback: if the ready handshake never completes, try to flush
        // the buffer anyway after 3 seconds.
        this.fallbackTimer = setTimeout(() => {
            if (!this.webviewReady) {
                this._log(`FALLBACK: ready signal never received after 3 s — force-flushing ${this.eventBuffer.length} events`);
                this.webviewReady = true;
                this._flushBuffer();
            }
        }, 3000);
    }
    /**
     * Create or reveal the panel.
     */
    static createOrShow(extensionUri) {
        const column = vscode.ViewColumn.Beside;
        if (EvolvePanel.currentPanel) {
            // Reset for a new run: reload the webview HTML and clear buffers
            EvolvePanel.currentPanel._log('Re-showing existing panel — resetting webview');
            EvolvePanel.currentPanel.webviewReady = false;
            EvolvePanel.currentPanel.eventBuffer = [];
            if (EvolvePanel.currentPanel.fallbackTimer) {
                clearTimeout(EvolvePanel.currentPanel.fallbackTimer);
            }
            // Ensure scripts are still enabled
            EvolvePanel.currentPanel.panel.webview.options = {
                enableScripts: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(extensionUri, "media"),
                ],
            };
            EvolvePanel.currentPanel._log(`Re-show — enableScripts=${EvolvePanel.currentPanel.panel.webview.options.enableScripts}`);
            EvolvePanel.currentPanel.panel.webview.html = EvolvePanel.currentPanel.getHtml();
            // Re-arm fallback timer for the reloaded webview
            EvolvePanel.currentPanel.fallbackTimer = setTimeout(() => {
                if (!EvolvePanel.currentPanel?.webviewReady) {
                    EvolvePanel.currentPanel?._log(`FALLBACK (re-show): force-flushing ${EvolvePanel.currentPanel?.eventBuffer.length} events`);
                    if (EvolvePanel.currentPanel) {
                        EvolvePanel.currentPanel.webviewReady = true;
                        EvolvePanel.currentPanel._flushBuffer();
                    }
                }
            }, 3000);
            EvolvePanel.currentPanel.panel.reveal(column);
            return EvolvePanel.currentPanel;
        }
        const panel = vscode.window.createWebviewPanel(EvolvePanel.viewType, "MutaGenAI — Evolution", column, {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.joinPath(extensionUri, "media"),
            ],
        });
        // Force enableScripts on the webview object directly
        panel.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(extensionUri, "media"),
            ],
        };
        EvolvePanel.logFn(`[EvolvePanel] Panel created — enableScripts=${panel.webview.options.enableScripts}`);
        EvolvePanel.currentPanel = new EvolvePanel(panel, extensionUri);
        return EvolvePanel.currentPanel;
    }
    /**
     * Forward a sidecar event to the webview.
     * Buffers events until the webview signals it is ready.
     */
    postEvent(event) {
        if (this.webviewReady) {
            this.panel.webview.postMessage(event).then((ok) => {
                this._log(`postMessage type=${event.type} delivered=${ok}`);
            });
        }
        else {
            this._log(`Buffering event type=${event.type} (buffer=${this.eventBuffer.length + 1})`);
            this.eventBuffer.push(event);
        }
    }
    /** Flush all buffered events to the webview. */
    _flushBuffer() {
        const count = this.eventBuffer.length;
        for (const ev of this.eventBuffer) {
            this.panel.webview.postMessage(ev).then((ok) => {
                this._log(`flush postMessage type=${ev.type} delivered=${ok}`);
            });
        }
        this.eventBuffer = [];
        this._log(`Flushed ${count} events to webview`);
    }
    dispose() {
        EvolvePanel.currentPanel = undefined;
        this.panel.dispose();
        for (const d of this.disposables) {
            d.dispose();
        }
        this.disposables = [];
    }
    /**
     * Register a handler for messages FROM the webview (e.g. stop button).
     */
    onMessage(handler) {
        this.panel.webview.onDidReceiveMessage(handler, null, this.disposables);
    }
    // ---------------------------------------------------------------------------
    // Webview HTML
    // ---------------------------------------------------------------------------
    _getNonce() {
        let text = "";
        const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (let i = 0; i < 32; i++) {
            text += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        return text;
    }
    getHtml() {
        const scriptUri = this.panel.webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, "media", "dashboard.js"));
        return /*html*/ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MutaGenAI Evolution</title>
<style>
  :root {
    --bg: var(--vscode-editor-background);
    --fg: var(--vscode-editor-foreground);
    --border: var(--vscode-panel-border, #444);
    --accent: var(--vscode-textLink-foreground, #4fc1ff);
    --green: #4ec9b0;
    --red: #f14c4c;
    --orange: #cca700;
    --island0: #4fc1ff;
    --island1: #c586c0;
    --island2: #4ec9b0;
    --island3: #ce9178;
    --island4: #dcdcaa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--vscode-font-family, 'Segoe UI', sans-serif);
    font-size: var(--vscode-font-size, 13px);
    color: var(--fg);
    background: var(--bg);
    padding: 12px;
    overflow-x: hidden;
  }

  /* ── Header ────────────────────────────────────────────── */
  .header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .header h2 { font-size: 16px; font-weight: 600; }
  .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
  }
  .badge.running  { background: var(--accent); color: #000; }
  .badge.stopped  { background: var(--orange); color: #000; }
  .badge.done     { background: var(--green); color: #000; }
  .badge.error    { background: var(--red); color: #fff; }
  .badge.idle     { background: var(--border); color: var(--fg); }

  #stopBtn {
    margin-left: auto;
    padding: 4px 14px;
    background: var(--red);
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    display: none;
  }
  #stopBtn:hover { opacity: 0.85; }

  /* ── Progress ──────────────────────────────────────────── */
  .progress-bar-outer {
    width: 100%;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    margin-bottom: 14px;
    overflow: hidden;
  }
  .progress-bar-inner {
    height: 100%;
    width: 0%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  /* ── Stats row ─────────────────────────────────────────── */
  .stats {
    display: flex;
    gap: 20px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .stat-card {
    background: var(--vscode-editorWidget-background, #252526);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 16px;
    min-width: 120px;
    text-align: center;
  }
  .stat-card .label {
    font-size: 10px;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 4px;
  }
  .stat-card .value {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
  }
  .stat-card .value.improved { color: var(--green); }

  /* ── Two-column layout ─────────────────────────────────── */
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
  }
  @media (max-width: 700px) {
    .columns { grid-template-columns: 1fr; }
  }
  .panel-box {
    background: var(--vscode-editorWidget-background, #252526);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    overflow: hidden;
  }
  .panel-box h3 {
    font-size: 12px;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 10px;
  }

  /* ── Score chart (canvas) ──────────────────────────────── */
  #scoreChart {
    width: 100%;
    height: 220px;
    display: block;
  }

  /* ── Lineage tree (SVG) ────────────────────────────────── */
  #lineageTree {
    width: 100%;
    height: 300px;
    overflow: auto;
  }
  #lineageTree svg { display: block; }
  .node circle {
    stroke-width: 2;
    cursor: pointer;
  }
  .node text {
    font-size: 9px;
    fill: var(--fg);
  }
  .link {
    fill: none;
    stroke: var(--border);
    stroke-width: 1.2;
    opacity: 0.5;
  }

  /* ── Best prompt ───────────────────────────────────────── */
  .prompt-box {
    background: var(--vscode-editorWidget-background, #252526);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 14px;
  }
  .prompt-box h3 {
    font-size: 12px;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 8px;
  }
  .prompt-text {
    font-family: var(--vscode-editor-font-family, 'Cascadia Code', monospace);
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
    line-height: 1.5;
    opacity: 0.9;
  }

  /* ── Log ────────────────────────────────────────────────── */
  .log-box {
    background: var(--vscode-editorWidget-background, #252526);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    max-height: 160px;
    overflow-y: auto;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 11px;
    line-height: 1.6;
  }
  .log-line { opacity: 0.7; }
  .log-line.warn { color: var(--orange); }
  .log-line.error { color: var(--red); }
  .log-line .ts { opacity: 0.4; margin-right: 6px; }

  /* ── Tooltip ───────────────────────────────────────────── */
  .tooltip {
    position: absolute;
    background: var(--vscode-editorHoverWidget-background, #2d2d30);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 11px;
    pointer-events: none;
    z-index: 100;
    max-width: 320px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    display: none;
  }
  .tooltip .tt-label { opacity: 0.5; font-size: 10px; }
  .tooltip .tt-value { font-weight: 600; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h2>MutaGenAI Evolution</h2>
  <span id="statusBadge" class="badge idle">Idle</span>
  <span id="genLabel" style="font-size:12px;opacity:0.6;"></span>
  <button id="stopBtn" onclick="doStop()">■ Stop</button>
</div>

<!-- Progress -->
<div class="progress-bar-outer">
  <div id="progressBar" class="progress-bar-inner"></div>
</div>

<!-- Stats -->
<div class="stats">
  <div class="stat-card">
    <div class="label">Best Score</div>
    <div id="bestScore" class="value">—</div>
  </div>
  <div class="stat-card">
    <div class="label">Generation</div>
    <div id="genNum" class="value">0</div>
  </div>
  <div class="stat-card">
    <div class="label">Candidates</div>
    <div id="candidateCount" class="value">0</div>
  </div>
  <div class="stat-card">
    <div class="label">Elapsed</div>
    <div id="elapsed" class="value">—</div>
  </div>
</div>

<!-- Charts -->
<div class="columns">
  <div class="panel-box">
    <h3>Score Evolution</h3>
    <canvas id="scoreChart"></canvas>
  </div>
  <div class="panel-box" style="position:relative;">
    <h3>Lineage Tree</h3>
    <div id="lineageTree"></div>
  </div>
</div>

<!-- Best prompt -->
<div class="prompt-box">
  <h3>Best Prompt</h3>
  <div id="bestPrompt" class="prompt-text">Waiting for evolution to start…</div>
</div>

<!-- Log -->
<div class="log-box" id="logBox"></div>

<!-- Tooltip -->
<div class="tooltip" id="tooltip"></div>

<!-- JS status banner -->
<div id="jsBanner" style="position:fixed;top:0;left:0;right:0;padding:6px 12px;background:#f14c4c;color:#fff;font-weight:bold;font-size:13px;z-index:9999;text-align:center;">⚠ JAVASCRIPT NOT LOADED</div>

<script src="${scriptUri}"></script>
</body>
</html>`;
    }
    getMinimalHtml() {
        return `<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head><body>
<h1 id="t" style="color:red;padding:40px;">SCRIPT OFF</h1>
<div id="log" style="padding:20px;font-family:monospace;font-size:12px;"></div>
<script>
document.getElementById("t").textContent = "SCRIPT ON - " + new Date().toISOString();
document.getElementById("t").style.color = "green";
var vscode = acquireVsCodeApi();
vscode.postMessage({ command: "ready" });
var log = document.getElementById("log");
window.addEventListener("message", function(ev) {
  var d = ev.data;
  if (d && d.type) {
    var p = document.createElement("div");
    p.textContent = d.type + ": " + JSON.stringify(d).substring(0, 200);
    log.appendChild(p);
  }
});
<\/script>
</body></html>`;
    }
}
exports.EvolvePanel = EvolvePanel;
EvolvePanel.viewType = "mutagenai.evolvePanel";
EvolvePanel.logFn = console.log;
//# sourceMappingURL=evolvePanel.js.map