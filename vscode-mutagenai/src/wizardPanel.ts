/**
 * MutaGenAI Wizard — webview-based project bootstrapper.
 *
 * Mirrors the 10-step CLI wizard (MutaGenAI/wizard.py) with a
 * multi-step form inside a VS Code webview panel.  On completion it
 * generates a ready-to-run Python script, identical to the CLI output.
 */

import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import * as cp from "child_process";

export class WizardPanel {
  public static readonly viewType = "mutagenai.wizard";
  private static instance: WizardPanel | undefined;

  private readonly panel: vscode.WebviewPanel;
  private disposables: vscode.Disposable[] = [];

  public static createOrShow(extensionUri: vscode.Uri): WizardPanel {
    if (WizardPanel.instance) {
      WizardPanel.instance.panel.reveal(vscode.ViewColumn.One);
      return WizardPanel.instance;
    }

    const panel = vscode.window.createWebviewPanel(
      WizardPanel.viewType,
      "MutaGenAI — Wizard",
      vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true }
    );

    WizardPanel.instance = new WizardPanel(panel, extensionUri);
    return WizardPanel.instance;
  }

  private constructor(
    panel: vscode.WebviewPanel,
    private readonly extensionUri: vscode.Uri
  ) {
    this.panel = panel;
    this.panel.webview.html = this.getHtml();

    this.panel.onDidDispose(() => {
      WizardPanel.instance = undefined;
      this.disposables.forEach((d) => d.dispose());
    });

    this.panel.webview.onDidReceiveMessage(
      (msg) => this.handleMessage(msg),
      undefined,
      this.disposables
    );
  }

  private async handleMessage(msg: {
    command: string;
    data?: Record<string, unknown>;
  }): Promise<void> {
    console.log("[MutaGenAI Wizard] handleMessage:", msg.command);
    if (msg.command === "generate") {
      const state = msg.data as Record<string, unknown>;
      try {
        await this.generateScript(state);
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error("[MutaGenAI Wizard] generateScript error:", errMsg);
        vscode.window.showErrorMessage(`Wizard error: ${errMsg}`);
      }
    } else if (msg.command === "runNow") {
      // User clicked "Run Now" — launch evolution with the wizard config
      const state = msg.data as Record<string, unknown>;
      await vscode.commands.executeCommand(
        "mutagenai.evolveFromWizard",
        state
      );
    } else if (msg.command === "pickFile") {
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { "Data files": ["json", "csv", "txt"] },
        title: "Select file",
      });
      if (uris && uris.length > 0) {
        this.panel.webview.postMessage({
          command: "filePicked",
          field: msg.data?.field,
          path: uris[0].fsPath,
        });
      }
    }
  }

  private async generateScript(
    state: Record<string, unknown>
  ): Promise<void> {
    // Delegate to the Python wizard's generate_script via sidecar
    // Try workspace folders first, fall back to parent of extension dir
    const workspaceFolder =
      vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
      path.dirname(this.extensionUri.fsPath);

    console.log("[MutaGenAI Wizard] workspaceFolders:", vscode.workspace.workspaceFolders);
    console.log("[MutaGenAI Wizard] extensionUri:", this.extensionUri.fsPath);
    console.log("[MutaGenAI Wizard] resolved workspaceFolder:", workspaceFolder);

    let pythonPath = vscode.workspace
      .getConfiguration("mutagenai")
      .get<string>("pythonPath", "python3") as string;

    // Prefer .venv in workspace over bare "python3" default
    const venvPython = path.join(workspaceFolder, ".venv", "bin", "python");
    if (
      (pythonPath === "python3" || pythonPath === "python") &&
      fs.existsSync(venvPython)
    ) {
      pythonPath = venvPython;
    }

    const sidecarPath = path.join(
      path.dirname(__dirname),
      "sidecar",
      "wizard_gen.py"
    );

    console.log("[MutaGenAI Wizard] pythonPath:", pythonPath);
    console.log("[MutaGenAI Wizard] sidecarPath:", sidecarPath);
    console.log("[MutaGenAI Wizard] workspaceFolder:", workspaceFolder);
    console.log("[MutaGenAI Wizard] sidecar exists:", fs.existsSync(sidecarPath));

    // Write wizard state to a temp file in OS temp dir (workspace may be read-only)
    const os = require("os");
    const tmpFile = path.join(os.tmpdir(), ".mutagenai_wizard_state.json");
    fs.writeFileSync(tmpFile, JSON.stringify(state, null, 2), "utf-8");

    vscode.window.showInformationMessage("Generating evolution script…");

    try {
      const result: { stdout: string; stderr: string } = await new Promise(
        (resolve, reject) => {
          cp.execFile(
            pythonPath,
            [sidecarPath, tmpFile],
            { cwd: workspaceFolder, timeout: 30_000 },
            (err: Error | null, stdout: string, stderr: string) => {
              if (err) {
                reject(new Error(`${err.message}\nstderr: ${stderr}`));
              } else {
                resolve({ stdout, stderr });
              }
            }
          );
        }
      );

      // The sidecar prints the generated script path on stdout
      const scriptPath = result.stdout.trim();
      if (scriptPath && fs.existsSync(scriptPath)) {
        const doc = await vscode.workspace.openTextDocument(scriptPath);
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
        vscode.window.showInformationMessage(
          `Script generated: ${path.basename(scriptPath)}`
        );
      } else {
        // Fall back to showing output inline
        const doc = await vscode.workspace.openTextDocument({
          content: result.stdout || result.stderr,
          language: "python",
        });
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      vscode.window.showErrorMessage(`Wizard generation failed: ${msg}`);
    } finally {
      // Clean up temp file
      try {
        fs.unlinkSync(tmpFile);
      } catch {
        // ignore
      }
    }
  }

  // ── HTML ────────────────────────────────────────────────────────────

  private getHtml(): string {
    return /*html*/ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MutaGenAI Wizard</title>
<style>
:root {
  --bg: var(--vscode-editor-background);
  --fg: var(--vscode-editor-foreground);
  --accent: var(--vscode-textLink-foreground);
  --inputBg: var(--vscode-input-background);
  --inputBorder: var(--vscode-input-border, #444);
  --inputFg: var(--vscode-input-foreground);
  --btnBg: var(--vscode-button-background);
  --btnFg: var(--vscode-button-foreground);
  --btnHover: var(--vscode-button-hoverBackground);
  --border: var(--vscode-panel-border, #333);
  --descFg: var(--vscode-descriptionForeground, #999);
  --errorFg: var(--vscode-errorForeground, #f44);
  --successFg: #4ec9b0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--vscode-font-family); color: var(--fg); background: var(--bg); padding: 20px 32px; line-height: 1.6; }

h1 { font-size: 1.4em; margin-bottom: 4px; }
h1 span { color: var(--accent); }
.subtitle { color: var(--descFg); margin-bottom: 24px; font-size: 0.9em; }

/* Progress bar */
.progress { display: flex; gap: 4px; margin-bottom: 28px; }
.progress .step {
  flex: 1; height: 4px; border-radius: 2px;
  background: var(--border); transition: background 0.3s;
}
.progress .step.done { background: var(--successFg); }
.progress .step.active { background: var(--accent); }

/* Steps */
.step-container { display: none; }
.step-container.active { display: block; }
.step-title { font-size: 1.1em; font-weight: 600; margin-bottom: 4px; }
.step-desc { color: var(--descFg); margin-bottom: 16px; font-size: 0.9em; }

/* Form elements */
label { display: block; font-weight: 500; margin-bottom: 4px; margin-top: 14px; font-size: 0.9em; }
select, input[type="text"], input[type="number"], textarea {
  width: 100%; padding: 6px 10px; border: 1px solid var(--inputBorder);
  border-radius: 4px; background: var(--inputBg); color: var(--inputFg);
  font-family: inherit; font-size: 0.9em;
}
textarea { min-height: 80px; resize: vertical; }
input[type="number"] { width: 120px; }

/* Radio/option cards */
.option-cards { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.option-card {
  border: 1px solid var(--inputBorder); border-radius: 6px; padding: 10px 14px;
  cursor: pointer; transition: border-color 0.2s, background 0.2s;
}
.option-card:hover { border-color: var(--accent); }
.option-card.selected { border-color: var(--accent); background: rgba(0,127,255,0.08); }
.option-card .label { font-weight: 600; }
.option-card .desc { color: var(--descFg); font-size: 0.85em; }

/* Checkboxes */
.check-group { margin-top: 8px; }
.check-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.check-item input { margin: 0; }

/* Tags / chips */
.chip-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip {
  background: var(--inputBg); border: 1px solid var(--inputBorder); border-radius: 12px;
  padding: 2px 10px; font-size: 0.85em; display: flex; align-items: center; gap: 4px;
}
.chip .remove { cursor: pointer; opacity: 0.6; }
.chip .remove:hover { opacity: 1; }

/* Navigation */
.nav { display: flex; justify-content: space-between; margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--border); }
.btn {
  padding: 8px 20px; border: none; border-radius: 4px; cursor: pointer;
  font-size: 0.9em; font-weight: 500;
}
.btn-primary { background: var(--btnBg); color: var(--btnFg); }
.btn-primary:hover { background: var(--btnHover); }
.btn-secondary { background: transparent; color: var(--fg); border: 1px solid var(--inputBorder); }
.btn-secondary:hover { border-color: var(--accent); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* Summary */
.summary-grid { display: grid; grid-template-columns: 160px 1fr; gap: 6px 12px; font-size: 0.9em; }
.summary-grid .key { color: var(--descFg); }
</style>
</head>
<body>

<h1>🧬 <span>MutaGenAI</span> — Wizard</h1>
<p class="subtitle">Generate a ready-to-run prompt evolution script for your agent.</p>

<div class="progress" id="progressBar"></div>

<!-- Step 1: Problem Type -->
<div class="step-container" data-step="0">
  <div class="step-title">Step 1 — Problem Type</div>
  <div class="step-desc">What kind of task will the evolved prompt perform?</div>
  <div class="option-cards" data-field="problemType">
    <div class="option-card" data-value="tool_routing">
      <div class="label">Tool Routing</div>
      <div class="desc">Map user queries to tool/function calls (JSON output)</div>
    </div>
    <div class="option-card selected" data-value="classification">
      <div class="label">Classification</div>
      <div class="desc">Classify input text into one of several categories</div>
    </div>
    <div class="option-card" data-value="generation">
      <div class="label">Generation</div>
      <div class="desc">Generate free-form text, summaries, or structured output</div>
    </div>
    <div class="option-card" data-value="code">
      <div class="label">Code</div>
      <div class="desc">Generate or transform code</div>
    </div>
    <div class="option-card" data-value="conversation">
      <div class="label">Conversation</div>
      <div class="desc">Multi-turn dialogue or chatbot agent</div>
    </div>
  </div>
</div>

<!-- Step 2: Task Description -->
<div class="step-container" data-step="1">
  <div class="step-title">Step 2 — Task Description</div>
  <div class="step-desc">Describe what your agent does. Be specific — this drives mutation generation and LLM-as-Judge rubrics.</div>
  <label for="taskDesc">Task description</label>
  <textarea id="taskDesc" placeholder="e.g. You are an API-calling assistant that maps natural-language queries to tool calls in the format [ToolName(param=value)]."></textarea>
</div>

<!-- Step 3: Ground Truth -->
<div class="step-container" data-step="2">
  <div class="step-title">Step 3 — Ground Truth</div>
  <div class="step-desc">Do you have labelled evaluation data — input/output pairs where you know the correct answer?</div>
  <div class="option-cards" data-field="groundTruth">
    <div class="option-card" data-value="yes">
      <div class="label">Yes</div>
      <div class="desc">I have a dataset of inputs with expected outputs</div>
    </div>
    <div class="option-card" data-value="partial">
      <div class="label">Partial</div>
      <div class="desc">I have some labels but not a complete dataset</div>
    </div>
    <div class="option-card selected" data-value="no">
      <div class="label">No</div>
      <div class="desc">No labels — I need label-free evaluation strategies</div>
    </div>
  </div>
  <div id="evalFileSection" style="display:none; margin-top:16px;">
    <label>Eval data file (JSON: [{"input": "...", "expected": "..."}])</label>
    <div style="display:flex;gap:8px;">
      <input type="text" id="evalFile" placeholder="Path to eval data JSON" style="flex:1"/>
      <button class="btn btn-secondary" onclick="pickFile('evalFile')">Browse…</button>
    </div>
  </div>
</div>

<!-- Step 4: Test Inputs -->
<div class="step-container" data-step="3">
  <div class="step-title">Step 4 — Test Inputs</div>
  <div class="step-desc">Provide unlabelled test inputs your agent should handle. Used during evolution to generate outputs for scoring.</div>
  <label>Load from file or enter below</label>
  <div style="display:flex;gap:8px;margin-bottom:12px;">
    <input type="text" id="testInputFile" placeholder="Path to text file (one input per line)" style="flex:1"/>
    <button class="btn btn-secondary" onclick="pickFile('testInputFile')">Browse…</button>
  </div>
  <label>Or enter test inputs (one per line)</label>
  <textarea id="testInputs" placeholder="What is the weather in Paris?&#10;Book a flight to London&#10;Show my calendar for tomorrow" rows="5"></textarea>
</div>

<!-- Step 5: Scoring Strategy -->
<div class="step-container" data-step="4">
  <div class="step-title">Step 5 — Scoring Strategy</div>
  <div class="step-desc">Select one or more strategies to evaluate prompt candidates. Composite is recommended for best results.</div>
  <div class="check-group" id="strategyChecks">
    <div class="check-item"><input type="checkbox" value="llm_judge" id="s_llm"/><label for="s_llm"><strong>LLM-as-Judge</strong> — A second LLM rates output against a rubric</label></div>
    <div class="check-item"><input type="checkbox" value="self_consistency" id="s_sc"/><label for="s_sc"><strong>Self-Consistency</strong> — Consistent outputs across runs score higher</label></div>
    <div class="check-item"><input type="checkbox" value="proxy_metrics" id="s_pm"/><label for="s_pm"><strong>Proxy Metrics</strong> — Structural checks: valid JSON, format, length</label></div>
    <div class="check-item"><input type="checkbox" value="tool_success" id="s_ts"/><label for="s_ts"><strong>Tool-Use Success</strong> — Execute tool calls, score by return code</label></div>
    <div class="check-item"><input type="checkbox" value="preference" id="s_pref"/><label for="s_pref"><strong>Preference Pairs</strong> — Compare against good/bad output examples</label></div>
    <div class="check-item"><input type="checkbox" value="human" id="s_human"/><label for="s_human"><strong>Human-as-Judge</strong> — You rate outputs interactively</label></div>
    <div class="check-item"><input type="checkbox" value="composite" id="s_comp" checked/><label for="s_comp"><strong>Composite</strong> — Weighted mix (recommended)</label></div>
  </div>
  <div id="rubricSection" style="margin-top:14px;">
    <label for="rubric">LLM Judge rubric (optional — auto-generated from task description if blank)</label>
    <textarea id="rubric" rows="3" placeholder="Rate how accurately the output maps the user query to the correct tool call..."></textarea>
  </div>
</div>

<!-- Step 6: Domain Mutations -->
<div class="step-container" data-step="5">
  <div class="step-title">Step 6 — Domain Mutations</div>
  <div class="step-desc">Custom rewrite instructions applied to prompts each generation. Leave blank to auto-generate from your task description.</div>
  <label>Add mutation instructions (one per line)</label>
  <textarea id="mutations" rows="5" placeholder="Add chain-of-thought reasoning before the tool call&#10;Enforce strict JSON output format&#10;Add error recovery for malformed queries"></textarea>
</div>

<!-- Step 7: Human Evaluation -->
<div class="step-container" data-step="6">
  <div class="step-title">Step 7 — Human Evaluation</div>
  <div class="step-desc">Optionally add human-in-the-loop evaluation during or after evolution.</div>
  <div class="option-cards" data-field="humanEval">
    <div class="option-card" data-value="always">
      <div class="label">Always</div>
      <div class="desc">Human rates every generation (highest quality, most effort)</div>
    </div>
    <div class="option-card selected" data-value="final">
      <div class="label">Final only</div>
      <div class="desc">Human picks the winner from the top-K after evolution finishes</div>
    </div>
    <div class="option-card" data-value="no">
      <div class="label">No</div>
      <div class="desc">Fully automated — no human involvement</div>
    </div>
  </div>
</div>

<!-- Step 8: Seed Templates -->
<div class="step-container" data-step="7">
  <div class="step-title">Step 8 — Seed Templates</div>
  <div class="step-desc">Starting prompts for evolution. Using your existing best prompt gives evolution a head start.</div>
  <label>Seed prompts (one per block, separate with blank lines)</label>
  <textarea id="seeds" rows="6" placeholder="You are an AI assistant that routes user queries to the correct tool.&#10;&#10;Act as a precise workflow engine. Emit the execution plan concisely."></textarea>
  <p style="color:var(--descFg);font-size:0.85em;margin-top:6px;">Leave empty to auto-generate seeds from your task description.</p>
</div>

<!-- Step 9: Backend -->
<div class="step-container" data-step="8">
  <div class="step-title">Step 9 — LLM Backend</div>
  <div class="step-desc">Which LLM backend will power the evolution?</div>
  <div class="option-cards" data-field="backend">
    <div class="option-card selected" data-value="ollama">
      <div class="label">Ollama</div>
      <div class="desc">Local model via Ollama (free, private, needs ollama running)</div>
    </div>
    <div class="option-card" data-value="openai">
      <div class="label">OpenAI</div>
      <div class="desc">OpenAI API (GPT-4o-mini etc, needs OPENAI_API_KEY)</div>
    </div>
    <div class="option-card" data-value="azure_openai">
      <div class="label">Azure OpenAI</div>
      <div class="desc">Azure OpenAI (enterprise, needs endpoint + deployment)</div>
    </div>
  </div>
  <label for="modelName" style="margin-top:14px;">Model name</label>
  <input type="text" id="modelName" value="llama3.2" placeholder="llama3.2, gpt-4o-mini, etc."/>
</div>

<!-- Step 10: Configuration -->
<div class="step-container" data-step="9">
  <div class="step-title">Step 10 — Evolution Configuration</div>
  <div class="step-desc">Choose a preset or customise the evolutionary parameters.</div>
  <div class="option-cards" data-field="preset">
    <div class="option-card selected" data-value="standard">
      <div class="label">Standard</div>
      <div class="desc">5 generations, 6 population, 2 islands (~5 min)</div>
    </div>
    <div class="option-card" data-value="deep">
      <div class="label">Deep</div>
      <div class="desc">10 generations, 8 population, 3 islands (~15 min)</div>
    </div>
    <div class="option-card" data-value="custom">
      <div class="label">Custom</div>
      <div class="desc">Set your own parameters</div>
    </div>
  </div>
  <div id="customConfig" style="display:none;margin-top:14px;">
    <label for="iterations">Generations</label>
    <input type="number" id="iterations" value="5" min="1" max="100"/>
    <label for="popSize">Population size</label>
    <input type="number" id="popSize" value="6" min="2" max="50"/>
    <label for="numIslands">Islands</label>
    <input type="number" id="numIslands" value="2" min="1" max="10"/>
  </div>
</div>

<!-- Summary / Review -->
<div class="step-container" data-step="10">
  <div class="step-title">Review &amp; Generate</div>
  <div class="step-desc">Review your configuration, then generate the evolution script or run evolution directly with the real-time dashboard.</div>
  <div class="summary-grid" id="summaryGrid"></div>
  <div style="margin-top:24px;display:flex;gap:12px;">
    <button class="btn btn-primary" id="btnGenerate" onclick="generate()" style="flex:1;">📄 Generate Script</button>
    <button class="btn btn-primary" id="btnRunNow" onclick="runNow()" style="flex:1;background:var(--successFg,#4ec9b0);color:#000;">🧬 Run Now with Dashboard</button>
  </div>
</div>

<!-- Navigation -->
<div class="nav">
  <button class="btn btn-secondary" id="btnBack" onclick="prevStep()" disabled>← Back</button>
  <div>
    <span id="stepLabel" style="color:var(--descFg);font-size:0.85em;margin-right:12px;"></span>
    <button class="btn btn-primary" id="btnNext" onclick="nextStep()">Next →</button>
  </div>
</div>

<script>
const vscode = acquireVsCodeApi();
const TOTAL_STEPS = 11; // 0..10
let currentStep = 0;

// Wizard state
const state = {
  problemType: 'tool_routing',
  taskDescription: '',
  groundTruth: 'no',
  evalFile: '',
  testInputFile: '',
  testInputs: [],
  strategies: ['composite'],
  rubric: '',
  mutations: [],
  humanEval: 'final',
  seeds: [],
  backend: 'ollama',
  model: 'llama3.2',
  preset: 'standard',
  iterations: 5,
  populationSize: 6,
  numIslands: 2,
};

// ── Init ──────────────────────────────────────────────────────────────
(function init() {
  // Build progress bar
  const bar = document.getElementById('progressBar');
  for (let i = 0; i < TOTAL_STEPS; i++) {
    const el = document.createElement('div');
    el.className = 'step';
    bar.appendChild(el);
  }

  // Wire up option cards
  document.querySelectorAll('.option-cards').forEach(group => {
    const field = group.getAttribute('data-field');
    group.querySelectorAll('.option-card').forEach(card => {
      card.addEventListener('click', () => {
        group.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        state[field] = card.getAttribute('data-value');
        onFieldChange(field);
      });
    });
  });

  showStep(0);
})();

// ── Navigation ────────────────────────────────────────────────────────
function showStep(n) {
  currentStep = n;
  document.querySelectorAll('.step-container').forEach(el => {
    el.classList.toggle('active', parseInt(el.getAttribute('data-step')) === n);
  });

  // Progress bar
  document.querySelectorAll('.progress .step').forEach((el, i) => {
    el.className = 'step' + (i < n ? ' done' : i === n ? ' active' : '');
  });

  // Nav buttons
  document.getElementById('btnBack').disabled = n === 0;
  const btnNext = document.getElementById('btnNext');
  if (n === TOTAL_STEPS - 1) {
    // On the review step, hide the Next button (dedicated buttons are shown instead)
    btnNext.style.display = 'none';
  } else {
    btnNext.style.display = '';
    btnNext.textContent = 'Next →';
    btnNext.style.fontWeight = '500';
  }

  document.getElementById('stepLabel').textContent =
    n < TOTAL_STEPS - 1 ? 'Step ' + (n + 1) + ' of 10' : 'Ready';

  // Populate summary on last step
  if (n === TOTAL_STEPS - 1) buildSummary();
}

function nextStep() {
  collectCurrentStep();
  showStep(currentStep + 1);
}

function prevStep() {
  collectCurrentStep();
  if (currentStep > 0) showStep(currentStep - 1);
}

// ── Collect values from current step ──────────────────────────────────
function collectCurrentStep() {
  switch (currentStep) {
    case 1:
      state.taskDescription = document.getElementById('taskDesc').value.trim();
      break;
    case 2:
      state.evalFile = document.getElementById('evalFile').value.trim();
      break;
    case 3:
      state.testInputFile = document.getElementById('testInputFile').value.trim();
      state.testInputs = document.getElementById('testInputs').value
        .split('\\n').map(s => s.trim()).filter(Boolean);
      break;
    case 4: {
      const checked = [];
      document.querySelectorAll('#strategyChecks input:checked').forEach(cb => {
        checked.push(cb.value);
      });
      state.strategies = checked.length ? checked : ['composite'];
      state.rubric = document.getElementById('rubric').value.trim();
      break;
    }
    case 5:
      state.mutations = document.getElementById('mutations').value
        .split('\\n').map(s => s.trim()).filter(Boolean);
      break;
    case 7:
      state.seeds = document.getElementById('seeds').value
        .split('\\n\\n').map(s => s.trim()).filter(Boolean);
      break;
    case 8:
      state.model = document.getElementById('modelName').value.trim() || 'llama3.2';
      break;
    case 9: {
      const preset = state.preset;
      if (preset === 'standard') { state.iterations = 5; state.populationSize = 6; state.numIslands = 2; }
      else if (preset === 'deep') { state.iterations = 10; state.populationSize = 8; state.numIslands = 3; }
      else {
        state.iterations = parseInt(document.getElementById('iterations').value) || 5;
        state.populationSize = parseInt(document.getElementById('popSize').value) || 6;
        state.numIslands = parseInt(document.getElementById('numIslands').value) || 2;
      }
      break;
    }
  }
}

// ── Field change handlers ─────────────────────────────────────────────
function onFieldChange(field) {
  if (field === 'groundTruth') {
    document.getElementById('evalFileSection').style.display =
      (state.groundTruth === 'yes' || state.groundTruth === 'partial') ? 'block' : 'none';
  }
  if (field === 'preset') {
    document.getElementById('customConfig').style.display =
      state.preset === 'custom' ? 'block' : 'none';
  }
  if (field === 'backend') {
    const modelInput = document.getElementById('modelName');
    if (state.backend === 'ollama') modelInput.value = 'llama3.2';
    else if (state.backend === 'openai') modelInput.value = 'gpt-4o-mini';
    else modelInput.value = 'gpt-4o-mini';
  }
}

// ── File picker ───────────────────────────────────────────────────────
function pickFile(field) {
  vscode.postMessage({ command: 'pickFile', data: { field } });
}

window.addEventListener('message', e => {
  const msg = e.data;
  if (msg.command === 'filePicked' && msg.field) {
    const el = document.getElementById(msg.field);
    if (el) el.value = msg.path;
  }
});

// ── Summary ───────────────────────────────────────────────────────────
function buildSummary() {
  const grid = document.getElementById('summaryGrid');
  const rows = [
    ['Problem type', state.problemType],
    ['Task', state.taskDescription || '(not set)'],
    ['Ground truth', state.groundTruth],
    ['Strategies', state.strategies.join(', ')],
    ['Human eval', state.humanEval],
    ['Backend', state.backend + ' / ' + state.model],
    ['Config', state.preset === 'custom'
      ? state.iterations + ' gen, ' + state.populationSize + ' pop, ' + state.numIslands + ' islands'
      : state.preset],
    ['Seeds', state.seeds.length ? state.seeds.length + ' provided' : 'Auto-generated'],
    ['Mutations', state.mutations.length ? state.mutations.length + ' custom' : 'Auto-generated'],
  ];
  grid.innerHTML = rows.map(([k, v]) =>
    '<div class="key">' + k + '</div><div>' + v + '</div>'
  ).join('');
}

// ── Generate ──────────────────────────────────────────────────────────
function generate() {
  try {
    collectCurrentStep();
    const btn = document.getElementById('btnGenerate');
    btn.textContent = 'Generating…';
    btn.disabled = true;
    vscode.postMessage({ command: 'generate', data: state });
    // Re-enable after a delay (generation happens async)
    setTimeout(() => { btn.textContent = '📄 Generate Script'; btn.disabled = false; }, 5000);
  } catch (err) {
    console.error('[MutaGenAI Wizard webview] generate error:', err);
    document.getElementById('btnGenerate').textContent = 'Error — see console';
  }
}

// ── Run Now — launch evolution with real-time dashboard ───────────────
function runNow() {
  try {
    collectCurrentStep();
    const btn = document.getElementById('btnRunNow');
    btn.textContent = 'Launching…';
    btn.disabled = true;
    vscode.postMessage({ command: 'runNow', data: state });
  } catch (err) {
    console.error('[MutaGenAI Wizard webview] runNow error:', err);
    const btn = document.getElementById('btnRunNow');
    btn.textContent = 'Error — see console';
    btn.disabled = false;
  }
}
</script>
</body>
</html>`;
  }
}
