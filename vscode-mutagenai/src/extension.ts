/**
 * MutaGenAI VS Code Extension — main entry point.
 *
 * Registers commands, tree views, and orchestrates the sidecar process
 * for evolutionary prompt optimisation with real-time visualisation.
 */

import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { Sidecar } from "./sidecar";
import { EvolvePanel } from "./evolvePanel";
import { WizardPanel } from "./wizardPanel";
import {
  SeedTemplateTreeProvider,
  RunsTreeProvider,
  RunRecord,
} from "./treeProviders";
import { EvolutionConfig, SidecarEvent, DoneEvent } from "./types";

let sidecar: Sidecar | undefined;
let statusBarItem: vscode.StatusBarItem;
let templateTree: SeedTemplateTreeProvider;
let runsTree: RunsTreeProvider;
const outputChannel = vscode.window.createOutputChannel("MutaGenAI");

function log(msg: string): void {
  const ts = new Date().toISOString();
  outputChannel.appendLine(`[${ts}] ${msg}`);
}

export function activate(context: vscode.ExtensionContext): void {
  // ── Status bar ──────────────────────────────────────────────────────
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100
  );
  statusBarItem.text = "$(beaker) MutaGenAI";
  statusBarItem.tooltip = "MutaGenAI — Evolutionary Prompt Optimiser";
  statusBarItem.command = "mutagenai.showDashboard";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // ── Tree views ──────────────────────────────────────────────────────
  templateTree = new SeedTemplateTreeProvider();
  runsTree = new RunsTreeProvider();

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      "mutagenai.templates",
      templateTree
    ),
    vscode.window.registerTreeDataProvider("mutagenai.runs", runsTree)
  );

  // ── Commands ────────────────────────────────────────────────────────

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "mutagenai.evolvePrompt",
      () => evolvePromptCommand(context)
    ),
    vscode.commands.registerCommand(
      "mutagenai.stopEvolution",
      stopEvolutionCommand
    ),
    vscode.commands.registerCommand(
      "mutagenai.showDashboard",
      () => showDashboardCommand(context)
    ),
    vscode.commands.registerCommand(
      "mutagenai.openSeedTemplate",
      openSeedTemplateCommand
    ),
    vscode.commands.registerCommand(
      "mutagenai.runWizard",
      () => WizardPanel.createOrShow(context.extensionUri)
    ),
    vscode.commands.registerCommand(
      "mutagenai.evolveFromWizard",
      (wizardState: Record<string, unknown>) =>
        evolveFromWizardCommand(context, wizardState)
    ),
    // DIAGNOSTIC: standalone webview script test
    vscode.commands.registerCommand("mutagenai.testWebview", () => {
      log(">>> testWebview command invoked");
      outputChannel.show(true);
      const tp = vscode.window.createWebviewPanel(
        "mutagenai.scriptTest",
        "Script Test",
        vscode.ViewColumn.Beside,
        { enableScripts: true }
      );
      tp.webview.html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head><body>
<div id="t" style="font-size:32px;padding:40px;color:red;font-weight:bold;">SCRIPT OFF</div>
<script>
  document.getElementById("t").textContent = "SCRIPT ON - " + new Date().toISOString();
  document.getElementById("t").style.color = "green";
</script>
</body></html>`;
      log("testWebview panel created, enableScripts=" + tp.webview.options.enableScripts);
    })
  );

  // Refresh templates when workspace changes
  const watcher = vscode.workspace.createFileSystemWatcher(
    "**/seed_templates/*.json"
  );
  watcher.onDidChange(() => templateTree.refresh());
  watcher.onDidCreate(() => templateTree.refresh());
  watcher.onDidDelete(() => templateTree.refresh());
  context.subscriptions.push(watcher);
}

export function deactivate(): void {
  sidecar?.kill();
}

// ── Evolve Prompt Command ─────────────────────────────────────────────

async function evolvePromptCommand(
  context: vscode.ExtensionContext
): Promise<void> {
  // ── Gather configuration via quick picks ─────────────────────────
  const cfg = vscode.workspace.getConfiguration("mutagenai");

  // Pick seed template
  const templateNames = findSeedTemplates();
  let selectedTemplate: string | undefined;
  let seeds: string[] = [];

  if (templateNames.length > 0) {
    const pick = await vscode.window.showQuickPick(
      [
        { label: "$(file-code) Use a seed template", value: "template" },
        {
          label: "$(edit) Use current editor content",
          value: "editor",
        },
        { label: "$(symbol-string) Enter seeds manually", value: "manual" },
      ],
      {
        placeHolder: "How should the initial population be seeded?",
        title: "MutaGenAI — Seed Selection",
      }
    );

    if (!pick) {
      return;
    }

    if (pick.value === "template") {
      const items = templateNames.map((t) => ({
        label: t.name,
        description: `${t.seedCount} seeds`,
        detail: t.path,
        value: t.name,
      }));
      const sel = await vscode.window.showQuickPick(items, {
        placeHolder: "Select a seed template",
        title: "MutaGenAI — Seed Template",
      });
      if (!sel) {
        return;
      }
      selectedTemplate = sel.value;
    } else if (pick.value === "editor") {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor.");
        return;
      }
      const text = editor.document.getText();
      if (!text.trim()) {
        vscode.window.showErrorMessage("Editor is empty.");
        return;
      }
      seeds = [text.trim()];
    } else {
      const raw = await vscode.window.showInputBox({
        prompt: "Enter seed prompts separated by |||",
        placeHolder: "You are an AI assistant ||| Act as a helpful agent",
        title: "MutaGenAI — Manual Seeds",
      });
      if (!raw) {
        return;
      }
      seeds = raw.split("|||").map((s) => s.trim()).filter(Boolean);
    }
  } else {
    // No templates found — use editor content
    const editor = vscode.window.activeTextEditor;
    if (editor && editor.document.getText().trim()) {
      seeds = [editor.document.getText().trim()];
    }
  }

  // ── Build evolution config ───────────────────────────────────────
  const config: EvolutionConfig = {
    backend: cfg.get<string>("backend", "ollama"),
    model: cfg.get<string>("model", "llama3.2"),
    iterations: cfg.get<number>("iterations", 5),
    populationSize: cfg.get<number>("populationSize", 6),
    numIslands: cfg.get<number>("numIslands", 2),
    eliteSize: Math.max(2, Math.floor(cfg.get<number>("populationSize", 6) / 2)),
    mutationRate: 0.6,
    crossoverRate: 0.3,
    earlyStopThreshold: cfg.get<number>("earlyStopThreshold", 100),
    earlyStopPatience: cfg.get<number>("earlyStopPatience", 3),
    seed: 42,
    seeds: seeds.length > 0 ? seeds : undefined,
    seedTemplate: selectedTemplate,
    seedTemplatesDir: cfg.get<string>("seedTemplatesDir", ""),
  };

  await runEvolution(context, config);
}

// ── Evolve from Wizard Command ────────────────────────────────────────

async function evolveFromWizardCommand(
  context: vscode.ExtensionContext,
  wizardState: Record<string, unknown>
): Promise<void> {
  console.log("[MutaGenAI] evolveFromWizardCommand called with state:", JSON.stringify(wizardState, null, 2));
  // Map wizard state to EvolutionConfig
  const iterations =
    (wizardState.iterations as number) ||
    (wizardState.preset === "deep" ? 10 : 5);
  const populationSize =
    (wizardState.populationSize as number) ||
    (wizardState.preset === "deep" ? 8 : 6);
  const numIslands =
    (wizardState.numIslands as number) ||
    (wizardState.preset === "deep" ? 3 : 2);

  const seeds = Array.isArray(wizardState.seeds) && wizardState.seeds.length > 0
    ? (wizardState.seeds as string[])
    : undefined;

  // The wizard always runs in no-eval mode (no ground-truth labels)
  const config: EvolutionConfig = {
    backend: (wizardState.backend as string) || "ollama",
    model: (wizardState.model as string) || "llama3.2",
    iterations,
    populationSize,
    numIslands,
    eliteSize: Math.max(2, Math.floor(populationSize / 2)),
    mutationRate: 0.6,
    crossoverRate: 0.3,
    earlyStopThreshold: 100,
    earlyStopPatience: 3,
    seed: 42,
    seeds,
    // No-eval specific fields
    mode: "noeval",
    taskDescription: (wizardState.taskDescription as string) || "",
    testInputs: Array.isArray(wizardState.testInputs)
      ? (wizardState.testInputs as string[])
      : [],
    problemType: (wizardState.problemType as string) || "tool_routing",
    rubric: (wizardState.rubric as string) || "",
    mutations: Array.isArray(wizardState.mutations)
      ? (wizardState.mutations as string[])
      : [],
    strategies: Array.isArray(wizardState.strategies)
      ? (wizardState.strategies as string[])
      : ["composite"],
    adaptiveMutations: true,
    llmMutationRate: 0.3,
    refineAfterSplice: true,
  };

  await runEvolution(context, config);
}

// ── Shared Evolution Runner ───────────────────────────────────────────

async function runEvolution(
  context: vscode.ExtensionContext,
  config: EvolutionConfig
): Promise<void> {
  log(">>> runEvolution ENTERED");
  outputChannel.show(true);
  if (sidecar?.running) {
    const choice = await vscode.window.showWarningMessage(
      "An evolution run is already in progress.",
      "Stop & Start New",
      "Cancel"
    );
    if (choice !== "Stop & Start New") {
      return;
    }
    sidecar.stop();
    await new Promise((r) => setTimeout(r, 500));
    sidecar.kill();
  }

  // ── Open the dashboard panel ─────────────────────────────────────
  EvolvePanel.setLogger(log);
  const panel = EvolvePanel.createOrShow(context.extensionUri);
  log("Dashboard panel created/shown");

  // ── Start the sidecar ────────────────────────────────────────────
  sidecar = new Sidecar();
  log(`Starting sidecar with config: ${JSON.stringify(config).substring(0, 300)}`);

  const runId = Date.now().toString(36);
  const run: RunRecord = {
    id: runId,
    timestamp: new Date().toLocaleTimeString(),
    bestScore: 0,
    iterations: config.iterations,
    status: "running",
    model: config.model,
  };
  runsTree.addRun(run);

  // Status bar animation
  statusBarItem.text = "$(sync~spin) MutaGenAI — Evolving…";
  statusBarItem.backgroundColor = new vscode.ThemeColor(
    "statusBarItem.warningBackground"
  );

  // Listen for webview stop button
  panel.onMessage((msg) => {
    if (msg.command === "stop") {
      sidecar?.stop();
    }
  });

  // Forward sidecar events to webview
  let eventCount = 0;
  sidecar.on("event", (event: SidecarEvent) => {
    eventCount++;
    log(`Event #${eventCount} type=${event.type} ${JSON.stringify(event).substring(0, 200)}`);
    panel.postEvent(event);

    // Update status bar and runs tree
    switch (event.type) {
      case "generation":
        statusBarItem.text = `$(sync~spin) Gen ${event.generation}/${event.totalGenerations} — ${event.bestScore.toFixed(1)}%`;
        runsTree.updateRun(runId, { bestScore: event.bestScore });
        break;

      case "stopped":
        statusBarItem.text = `$(debug-stop) MutaGenAI — Stopped at ${run.bestScore.toFixed(1)}%`;
        statusBarItem.backgroundColor = undefined;
        runsTree.updateRun(runId, { status: "stopped" });
        break;

      case "done": {
        const done = event as DoneEvent;
        statusBarItem.text = `$(check) MutaGenAI — ${done.bestScore.toFixed(1)}%`;
        statusBarItem.backgroundColor = undefined;
        runsTree.updateRun(runId, {
          status: "done",
          bestScore: done.bestScore,
        });

        // Offer to open the evolved prompt in a new editor
        vscode.window
          .showInformationMessage(
            `Evolution complete! Best score: ${done.bestScore.toFixed(1)}% in ${done.wallTime.toFixed(1)}s`,
            "Open Evolved Prompt",
            "Copy to Clipboard"
          )
          .then((choice) => {
            if (choice === "Open Evolved Prompt") {
              openPromptInEditor(done.bestPrompt);
            } else if (choice === "Copy to Clipboard") {
              vscode.env.clipboard.writeText(done.bestPrompt);
              vscode.window.showInformationMessage(
                "Evolved prompt copied to clipboard."
              );
            }
          });
        break;
      }

      case "error":
        statusBarItem.text = "$(error) MutaGenAI — Error";
        statusBarItem.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.errorBackground"
        );
        runsTree.updateRun(runId, { status: "error" });
        vscode.window.showErrorMessage(
          `MutaGenAI error: ${event.message}`
        );
        break;
    }
  });

  // Run
  try {
    outputChannel.show(true); // Show output channel so user can see diagnostics
    const code = await sidecar.start(config);
    log(`Sidecar exited with code=${code}, total events received=${eventCount}`);
    if (eventCount === 0) {
      vscode.window.showWarningMessage(
        "MutaGenAI: Sidecar process exited without sending any events. Check the MutaGenAI output channel for details."
      );
    }
    if (code !== 0 && run.status === "running") {
      runsTree.updateRun(runId, { status: "error" });
      statusBarItem.text = "$(error) MutaGenAI — Exit " + code;
      statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground"
      );
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    vscode.window.showErrorMessage(`Failed to start evolution: ${msg}`);
    runsTree.updateRun(runId, { status: "error" });
    statusBarItem.text = "$(error) MutaGenAI";
    statusBarItem.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.errorBackground"
    );
  }
}

// ── Stop Command ──────────────────────────────────────────────────────

function stopEvolutionCommand(): void {
  if (sidecar?.running) {
    sidecar.stop();
    vscode.window.showInformationMessage("Stopping evolution…");
  } else {
    vscode.window.showInformationMessage("No evolution is running.");
  }
}

// ── Show Dashboard Command ────────────────────────────────────────────

function showDashboardCommand(
  context: vscode.ExtensionContext
): void {
  EvolvePanel.createOrShow(context.extensionUri);
}

// ── Open Seed Template Command ────────────────────────────────────────

async function openSeedTemplateCommand(): Promise<void> {
  const templates = findSeedTemplates();
  if (templates.length === 0) {
    vscode.window.showInformationMessage("No seed templates found.");
    return;
  }

  const pick = await vscode.window.showQuickPick(
    templates.map((t) => ({
      label: t.name,
      description: `${t.seedCount} seeds`,
      detail: t.path,
    })),
    { placeHolder: "Select a seed template to open" }
  );

  if (pick?.detail) {
    const doc = await vscode.workspace.openTextDocument(pick.detail);
    vscode.window.showTextDocument(doc);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────

interface TemplateInfo {
  name: string;
  path: string;
  seedCount: number;
}

function findSeedTemplates(): TemplateInfo[] {
  const results: TemplateInfo[] = [];
  const dirs: string[] = [];

  const custom = vscode.workspace
    .getConfiguration("mutagenai")
    .get<string>("seedTemplatesDir", "");
  if (custom) {
    dirs.push(custom);
  }

  for (const wf of vscode.workspace.workspaceFolders || []) {
    dirs.push(path.join(wf.uri.fsPath, "seed_templates"));
  }

  for (const dir of dirs) {
    if (!fs.existsSync(dir)) {
      continue;
    }
    for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".json"))) {
      try {
        const fullPath = path.join(dir, file);
        const data = JSON.parse(fs.readFileSync(fullPath, "utf-8"));
        results.push({
          name: data.name || path.basename(file, ".json"),
          path: fullPath,
          seedCount: (data.seeds || []).length,
        });
      } catch {
        // Skip malformed files
      }
    }
  }

  return results;
}

async function openPromptInEditor(prompt: string): Promise<void> {
  const doc = await vscode.workspace.openTextDocument({
    content: prompt,
    language: "markdown",
  });
  vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
}
