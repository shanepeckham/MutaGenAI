"use strict";
/**
 * MutaGenAI VS Code Extension — main entry point.
 *
 * Registers commands, tree views, and orchestrates the sidecar process
 * for evolutionary prompt optimisation with real-time visualisation.
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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const sidecar_1 = require("./sidecar");
const evolvePanel_1 = require("./evolvePanel");
const wizardPanel_1 = require("./wizardPanel");
const treeProviders_1 = require("./treeProviders");
let sidecar;
let statusBarItem;
let templateTree;
let runsTree;
const outputChannel = vscode.window.createOutputChannel("MutaGenAI");
function log(msg) {
    const ts = new Date().toISOString();
    outputChannel.appendLine(`[${ts}] ${msg}`);
}
function activate(context) {
    // ── Status bar ──────────────────────────────────────────────────────
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(beaker) MutaGenAI";
    statusBarItem.tooltip = "MutaGenAI — Evolutionary Prompt Optimiser";
    statusBarItem.command = "mutagenai.showDashboard";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
    // ── Tree views ──────────────────────────────────────────────────────
    templateTree = new treeProviders_1.SeedTemplateTreeProvider();
    runsTree = new treeProviders_1.RunsTreeProvider();
    context.subscriptions.push(vscode.window.registerTreeDataProvider("mutagenai.templates", templateTree), vscode.window.registerTreeDataProvider("mutagenai.runs", runsTree));
    // ── Commands ────────────────────────────────────────────────────────
    context.subscriptions.push(vscode.commands.registerCommand("mutagenai.evolvePrompt", () => evolvePromptCommand(context)), vscode.commands.registerCommand("mutagenai.stopEvolution", stopEvolutionCommand), vscode.commands.registerCommand("mutagenai.showDashboard", () => showDashboardCommand(context)), vscode.commands.registerCommand("mutagenai.openSeedTemplate", openSeedTemplateCommand), vscode.commands.registerCommand("mutagenai.runWizard", () => wizardPanel_1.WizardPanel.createOrShow(context.extensionUri)), vscode.commands.registerCommand("mutagenai.evolveFromWizard", (wizardState) => evolveFromWizardCommand(context, wizardState)), 
    // DIAGNOSTIC: standalone webview script test
    vscode.commands.registerCommand("mutagenai.testWebview", () => {
        log(">>> testWebview command invoked");
        outputChannel.show(true);
        const tp = vscode.window.createWebviewPanel("mutagenai.scriptTest", "Script Test", vscode.ViewColumn.Beside, { enableScripts: true });
        tp.webview.html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head><body>
<div id="t" style="font-size:32px;padding:40px;color:red;font-weight:bold;">SCRIPT OFF</div>
<script>
  document.getElementById("t").textContent = "SCRIPT ON - " + new Date().toISOString();
  document.getElementById("t").style.color = "green";
</script>
</body></html>`;
        log("testWebview panel created, enableScripts=" + tp.webview.options.enableScripts);
    }));
    // Refresh templates when workspace changes
    const watcher = vscode.workspace.createFileSystemWatcher("**/seed_templates/*.json");
    watcher.onDidChange(() => templateTree.refresh());
    watcher.onDidCreate(() => templateTree.refresh());
    watcher.onDidDelete(() => templateTree.refresh());
    context.subscriptions.push(watcher);
}
function deactivate() {
    sidecar?.kill();
}
// ── Evolve Prompt Command ─────────────────────────────────────────────
async function evolvePromptCommand(context) {
    // ── Gather configuration via quick picks ─────────────────────────
    const cfg = vscode.workspace.getConfiguration("mutagenai");
    // Pick seed template
    const templateNames = findSeedTemplates();
    let selectedTemplate;
    let seeds = [];
    if (templateNames.length > 0) {
        const pick = await vscode.window.showQuickPick([
            { label: "$(file-code) Use a seed template", value: "template" },
            {
                label: "$(edit) Use current editor content",
                value: "editor",
            },
            { label: "$(symbol-string) Enter seeds manually", value: "manual" },
        ], {
            placeHolder: "How should the initial population be seeded?",
            title: "MutaGenAI — Seed Selection",
        });
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
        }
        else if (pick.value === "editor") {
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
        }
        else {
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
    }
    else {
        // No templates found — use editor content
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.getText().trim()) {
            seeds = [editor.document.getText().trim()];
        }
    }
    // ── Build evolution config ───────────────────────────────────────
    const config = {
        backend: cfg.get("backend", "ollama"),
        model: cfg.get("model", "llama3.2"),
        iterations: cfg.get("iterations", 5),
        populationSize: cfg.get("populationSize", 6),
        numIslands: cfg.get("numIslands", 2),
        eliteSize: Math.max(2, Math.floor(cfg.get("populationSize", 6) / 2)),
        mutationRate: 0.6,
        crossoverRate: 0.3,
        earlyStopThreshold: cfg.get("earlyStopThreshold", 100),
        earlyStopPatience: cfg.get("earlyStopPatience", 3),
        seed: 42,
        seeds: seeds.length > 0 ? seeds : undefined,
        seedTemplate: selectedTemplate,
        seedTemplatesDir: cfg.get("seedTemplatesDir", ""),
    };
    await runEvolution(context, config);
}
// ── Evolve from Wizard Command ────────────────────────────────────────
async function evolveFromWizardCommand(context, wizardState) {
    console.log("[MutaGenAI] evolveFromWizardCommand called with state:", JSON.stringify(wizardState, null, 2));
    // Map wizard state to EvolutionConfig
    const iterations = wizardState.iterations ||
        (wizardState.preset === "deep" ? 10 : 5);
    const populationSize = wizardState.populationSize ||
        (wizardState.preset === "deep" ? 8 : 6);
    const numIslands = wizardState.numIslands ||
        (wizardState.preset === "deep" ? 3 : 2);
    const seeds = Array.isArray(wizardState.seeds) && wizardState.seeds.length > 0
        ? wizardState.seeds
        : undefined;
    // The wizard always runs in no-eval mode (no ground-truth labels)
    const config = {
        backend: wizardState.backend || "ollama",
        model: wizardState.model || "llama3.2",
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
        taskDescription: wizardState.taskDescription || "",
        testInputs: Array.isArray(wizardState.testInputs)
            ? wizardState.testInputs
            : [],
        problemType: wizardState.problemType || "tool_routing",
        rubric: wizardState.rubric || "",
        mutations: Array.isArray(wizardState.mutations)
            ? wizardState.mutations
            : [],
        strategies: Array.isArray(wizardState.strategies)
            ? wizardState.strategies
            : ["composite"],
        adaptiveMutations: true,
        llmMutationRate: 0.3,
        refineAfterSplice: true,
    };
    await runEvolution(context, config);
}
// ── Shared Evolution Runner ───────────────────────────────────────────
async function runEvolution(context, config) {
    log(">>> runEvolution ENTERED");
    outputChannel.show(true);
    if (sidecar?.running) {
        const choice = await vscode.window.showWarningMessage("An evolution run is already in progress.", "Stop & Start New", "Cancel");
        if (choice !== "Stop & Start New") {
            return;
        }
        sidecar.stop();
        await new Promise((r) => setTimeout(r, 500));
        sidecar.kill();
    }
    // ── Open the dashboard panel ─────────────────────────────────────
    evolvePanel_1.EvolvePanel.setLogger(log);
    const panel = evolvePanel_1.EvolvePanel.createOrShow(context.extensionUri);
    log("Dashboard panel created/shown");
    // ── Start the sidecar ────────────────────────────────────────────
    sidecar = new sidecar_1.Sidecar();
    log(`Starting sidecar with config: ${JSON.stringify(config).substring(0, 300)}`);
    const runId = Date.now().toString(36);
    const run = {
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
    statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
    // Listen for webview stop button
    panel.onMessage((msg) => {
        if (msg.command === "stop") {
            sidecar?.stop();
        }
    });
    // Forward sidecar events to webview
    let eventCount = 0;
    sidecar.on("event", (event) => {
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
                const done = event;
                statusBarItem.text = `$(check) MutaGenAI — ${done.bestScore.toFixed(1)}%`;
                statusBarItem.backgroundColor = undefined;
                runsTree.updateRun(runId, {
                    status: "done",
                    bestScore: done.bestScore,
                });
                // Offer to open the evolved prompt in a new editor
                vscode.window
                    .showInformationMessage(`Evolution complete! Best score: ${done.bestScore.toFixed(1)}% in ${done.wallTime.toFixed(1)}s`, "Open Evolved Prompt", "Copy to Clipboard")
                    .then((choice) => {
                    if (choice === "Open Evolved Prompt") {
                        openPromptInEditor(done.bestPrompt);
                    }
                    else if (choice === "Copy to Clipboard") {
                        vscode.env.clipboard.writeText(done.bestPrompt);
                        vscode.window.showInformationMessage("Evolved prompt copied to clipboard.");
                    }
                });
                break;
            }
            case "error":
                statusBarItem.text = "$(error) MutaGenAI — Error";
                statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
                runsTree.updateRun(runId, { status: "error" });
                vscode.window.showErrorMessage(`MutaGenAI error: ${event.message}`);
                break;
        }
    });
    // Run
    try {
        outputChannel.show(true); // Show output channel so user can see diagnostics
        const code = await sidecar.start(config);
        log(`Sidecar exited with code=${code}, total events received=${eventCount}`);
        if (eventCount === 0) {
            vscode.window.showWarningMessage("MutaGenAI: Sidecar process exited without sending any events. Check the MutaGenAI output channel for details.");
        }
        if (code !== 0 && run.status === "running") {
            runsTree.updateRun(runId, { status: "error" });
            statusBarItem.text = "$(error) MutaGenAI — Exit " + code;
            statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
        }
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        vscode.window.showErrorMessage(`Failed to start evolution: ${msg}`);
        runsTree.updateRun(runId, { status: "error" });
        statusBarItem.text = "$(error) MutaGenAI";
        statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
    }
}
// ── Stop Command ──────────────────────────────────────────────────────
function stopEvolutionCommand() {
    if (sidecar?.running) {
        sidecar.stop();
        vscode.window.showInformationMessage("Stopping evolution…");
    }
    else {
        vscode.window.showInformationMessage("No evolution is running.");
    }
}
// ── Show Dashboard Command ────────────────────────────────────────────
function showDashboardCommand(context) {
    evolvePanel_1.EvolvePanel.createOrShow(context.extensionUri);
}
// ── Open Seed Template Command ────────────────────────────────────────
async function openSeedTemplateCommand() {
    const templates = findSeedTemplates();
    if (templates.length === 0) {
        vscode.window.showInformationMessage("No seed templates found.");
        return;
    }
    const pick = await vscode.window.showQuickPick(templates.map((t) => ({
        label: t.name,
        description: `${t.seedCount} seeds`,
        detail: t.path,
    })), { placeHolder: "Select a seed template to open" });
    if (pick?.detail) {
        const doc = await vscode.workspace.openTextDocument(pick.detail);
        vscode.window.showTextDocument(doc);
    }
}
function findSeedTemplates() {
    const results = [];
    const dirs = [];
    const custom = vscode.workspace
        .getConfiguration("mutagenai")
        .get("seedTemplatesDir", "");
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
            }
            catch {
                // Skip malformed files
            }
        }
    }
    return results;
}
async function openPromptInEditor(prompt) {
    const doc = await vscode.workspace.openTextDocument({
        content: prompt,
        language: "markdown",
    });
    vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
}
//# sourceMappingURL=extension.js.map