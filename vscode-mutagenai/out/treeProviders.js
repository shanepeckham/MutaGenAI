"use strict";
/**
 * Tree view provider for MutaGenAI seed templates and recent runs.
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
exports.RunsTreeProvider = exports.SeedTemplateItem = exports.SeedTemplateTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
// ── Seed Templates Tree ─────────────────────────────────────────────────
class SeedTemplateTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
    }
    refresh() {
        this._onDidChangeTreeData.fire(undefined);
    }
    getTreeItem(element) {
        return element;
    }
    async getChildren(element) {
        if (element) {
            // Show individual seeds inside a template file
            return element.seeds.map((seed, i) => new SeedTemplateItem(`Seed ${i + 1}`, seed.substring(0, 80) + (seed.length > 80 ? "…" : ""), vscode.TreeItemCollapsibleState.None, [], element.filePath, "seed"));
        }
        // Top level: find seed template JSON files
        const dirs = this.getSeedTemplateDirs();
        const items = [];
        for (const dir of dirs) {
            if (!fs.existsSync(dir)) {
                continue;
            }
            const files = fs
                .readdirSync(dir)
                .filter((f) => f.endsWith(".json"))
                .sort();
            for (const file of files) {
                const fullPath = path.join(dir, file);
                try {
                    const raw = fs.readFileSync(fullPath, "utf-8");
                    const data = JSON.parse(raw);
                    const seeds = data.seeds || [];
                    const name = data.name || path.basename(file, ".json");
                    const desc = data.description || `${seeds.length} seeds`;
                    items.push(new SeedTemplateItem(name, desc, seeds.length > 0
                        ? vscode.TreeItemCollapsibleState.Collapsed
                        : vscode.TreeItemCollapsibleState.None, seeds, fullPath, "template"));
                }
                catch {
                    // Skip malformed JSON
                }
            }
        }
        if (items.length === 0) {
            return [
                new SeedTemplateItem("No seed templates found", "Add .json files to seed_templates/", vscode.TreeItemCollapsibleState.None, [], undefined, "empty"),
            ];
        }
        return items;
    }
    getSeedTemplateDirs() {
        const dirs = [];
        // Custom dir from settings
        const custom = vscode.workspace
            .getConfiguration("mutagenai")
            .get("seedTemplatesDir", "");
        if (custom) {
            dirs.push(custom);
        }
        // Workspace seed_templates/ directories
        const workspaceFolders = vscode.workspace.workspaceFolders || [];
        for (const wf of workspaceFolders) {
            dirs.push(path.join(wf.uri.fsPath, "seed_templates"));
        }
        return dirs;
    }
}
exports.SeedTemplateTreeProvider = SeedTemplateTreeProvider;
class SeedTemplateItem extends vscode.TreeItem {
    constructor(label, description, collapsibleState, seeds, filePath, kind) {
        super(label, collapsibleState);
        this.label = label;
        this.description = description;
        this.collapsibleState = collapsibleState;
        this.seeds = seeds;
        this.filePath = filePath;
        this.kind = kind;
        this.tooltip = description;
        if (kind === "template" && filePath) {
            this.iconPath = new vscode.ThemeIcon("file-code");
            this.contextValue = "seedTemplate";
            this.command = {
                command: "vscode.open",
                title: "Open Template",
                arguments: [vscode.Uri.file(filePath)],
            };
        }
        else if (kind === "seed") {
            this.iconPath = new vscode.ThemeIcon("symbol-string");
        }
        else {
            this.iconPath = new vscode.ThemeIcon("info");
        }
    }
}
exports.SeedTemplateItem = SeedTemplateItem;
class RunsTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.runs = [];
    }
    refresh() {
        this._onDidChangeTreeData.fire(undefined);
    }
    addRun(run) {
        this.runs.unshift(run);
        // Keep last 20 runs
        if (this.runs.length > 20) {
            this.runs = this.runs.slice(0, 20);
        }
        this.refresh();
    }
    updateRun(id, update) {
        const run = this.runs.find((r) => r.id === id);
        if (run) {
            Object.assign(run, update);
            this.refresh();
        }
    }
    getTreeItem(element) {
        return element;
    }
    async getChildren() {
        if (this.runs.length === 0) {
            return [
                new RunItem("No runs yet", "", "info", vscode.TreeItemCollapsibleState.None),
            ];
        }
        return this.runs.map((r) => {
            const icon = r.status === "running"
                ? "sync~spin"
                : r.status === "done"
                    ? "check"
                    : r.status === "stopped"
                        ? "debug-stop"
                        : "error";
            return new RunItem(`${r.bestScore.toFixed(1)}% — ${r.model}`, `${r.timestamp} · ${r.iterations} gens · ${r.status}`, icon, vscode.TreeItemCollapsibleState.None);
        });
    }
}
exports.RunsTreeProvider = RunsTreeProvider;
class RunItem extends vscode.TreeItem {
    constructor(label, description, icon, collapsibleState) {
        super(label, collapsibleState);
        this.description = description;
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}
//# sourceMappingURL=treeProviders.js.map