/**
 * Tree view provider for MutaGenAI seed templates and recent runs.
 */
import * as vscode from "vscode";
export declare class SeedTemplateTreeProvider implements vscode.TreeDataProvider<SeedTemplateItem> {
    private _onDidChangeTreeData;
    readonly onDidChangeTreeData: vscode.Event<SeedTemplateItem | undefined>;
    refresh(): void;
    getTreeItem(element: SeedTemplateItem): vscode.TreeItem;
    getChildren(element?: SeedTemplateItem): Promise<SeedTemplateItem[]>;
    private getSeedTemplateDirs;
}
export declare class SeedTemplateItem extends vscode.TreeItem {
    readonly label: string;
    readonly description: string;
    readonly collapsibleState: vscode.TreeItemCollapsibleState;
    readonly seeds: string[];
    readonly filePath: string | undefined;
    readonly kind: "template" | "seed" | "empty";
    constructor(label: string, description: string, collapsibleState: vscode.TreeItemCollapsibleState, seeds: string[], filePath: string | undefined, kind: "template" | "seed" | "empty");
}
export interface RunRecord {
    id: string;
    timestamp: string;
    bestScore: number;
    iterations: number;
    status: "running" | "done" | "stopped" | "error";
    model: string;
}
export declare class RunsTreeProvider implements vscode.TreeDataProvider<RunItem> {
    private _onDidChangeTreeData;
    readonly onDidChangeTreeData: vscode.Event<RunItem | undefined>;
    private runs;
    refresh(): void;
    addRun(run: RunRecord): void;
    updateRun(id: string, update: Partial<RunRecord>): void;
    getTreeItem(element: RunItem): vscode.TreeItem;
    getChildren(): Promise<RunItem[]>;
}
declare class RunItem extends vscode.TreeItem {
    constructor(label: string, description: string, icon: string, collapsibleState: vscode.TreeItemCollapsibleState);
}
export {};
//# sourceMappingURL=treeProviders.d.ts.map