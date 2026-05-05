/**
 * Tree view provider for MutaGenAI seed templates and recent runs.
 */

import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";

// ── Seed Templates Tree ─────────────────────────────────────────────────

export class SeedTemplateTreeProvider
  implements vscode.TreeDataProvider<SeedTemplateItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    SeedTemplateItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: SeedTemplateItem): vscode.TreeItem {
    return element;
  }

  async getChildren(
    element?: SeedTemplateItem
  ): Promise<SeedTemplateItem[]> {
    if (element) {
      // Show individual seeds inside a template file
      return element.seeds.map(
        (seed, i) =>
          new SeedTemplateItem(
            `Seed ${i + 1}`,
            seed.substring(0, 80) + (seed.length > 80 ? "…" : ""),
            vscode.TreeItemCollapsibleState.None,
            [],
            element.filePath,
            "seed"
          )
      );
    }

    // Top level: find seed template JSON files
    const dirs = this.getSeedTemplateDirs();
    const items: SeedTemplateItem[] = [];

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
          const seeds: string[] = data.seeds || [];
          const name = data.name || path.basename(file, ".json");
          const desc = data.description || `${seeds.length} seeds`;

          items.push(
            new SeedTemplateItem(
              name,
              desc,
              seeds.length > 0
                ? vscode.TreeItemCollapsibleState.Collapsed
                : vscode.TreeItemCollapsibleState.None,
              seeds,
              fullPath,
              "template"
            )
          );
        } catch {
          // Skip malformed JSON
        }
      }
    }

    if (items.length === 0) {
      return [
        new SeedTemplateItem(
          "No seed templates found",
          "Add .json files to seed_templates/",
          vscode.TreeItemCollapsibleState.None,
          [],
          undefined,
          "empty"
        ),
      ];
    }

    return items;
  }

  private getSeedTemplateDirs(): string[] {
    const dirs: string[] = [];

    // Custom dir from settings
    const custom = vscode.workspace
      .getConfiguration("mutagenai")
      .get<string>("seedTemplatesDir", "");
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

export class SeedTemplateItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly description: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly seeds: string[],
    public readonly filePath: string | undefined,
    public readonly kind: "template" | "seed" | "empty"
  ) {
    super(label, collapsibleState);
    this.tooltip = description;

    if (kind === "template" && filePath) {
      this.iconPath = new vscode.ThemeIcon("file-code");
      this.contextValue = "seedTemplate";
      this.command = {
        command: "vscode.open",
        title: "Open Template",
        arguments: [vscode.Uri.file(filePath)],
      };
    } else if (kind === "seed") {
      this.iconPath = new vscode.ThemeIcon("symbol-string");
    } else {
      this.iconPath = new vscode.ThemeIcon("info");
    }
  }
}

// ── Runs Tree ───────────────────────────────────────────────────────────

export interface RunRecord {
  id: string;
  timestamp: string;
  bestScore: number;
  iterations: number;
  status: "running" | "done" | "stopped" | "error";
  model: string;
}

export class RunsTreeProvider
  implements vscode.TreeDataProvider<RunItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    RunItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private runs: RunRecord[] = [];

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  addRun(run: RunRecord): void {
    this.runs.unshift(run);
    // Keep last 20 runs
    if (this.runs.length > 20) {
      this.runs = this.runs.slice(0, 20);
    }
    this.refresh();
  }

  updateRun(id: string, update: Partial<RunRecord>): void {
    const run = this.runs.find((r) => r.id === id);
    if (run) {
      Object.assign(run, update);
      this.refresh();
    }
  }

  getTreeItem(element: RunItem): vscode.TreeItem {
    return element;
  }

  async getChildren(): Promise<RunItem[]> {
    if (this.runs.length === 0) {
      return [
        new RunItem("No runs yet", "", "info", vscode.TreeItemCollapsibleState.None),
      ];
    }

    return this.runs.map((r) => {
      const icon =
        r.status === "running"
          ? "sync~spin"
          : r.status === "done"
          ? "check"
          : r.status === "stopped"
          ? "debug-stop"
          : "error";

      return new RunItem(
        `${r.bestScore.toFixed(1)}% — ${r.model}`,
        `${r.timestamp} · ${r.iterations} gens · ${r.status}`,
        icon,
        vscode.TreeItemCollapsibleState.None
      );
    });
  }
}

class RunItem extends vscode.TreeItem {
  constructor(
    label: string,
    description: string,
    icon: string,
    collapsibleState: vscode.TreeItemCollapsibleState
  ) {
    super(label, collapsibleState);
    this.description = description;
    this.iconPath = new vscode.ThemeIcon(icon);
  }
}
