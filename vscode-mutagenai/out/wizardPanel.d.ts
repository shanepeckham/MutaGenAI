/**
 * MutaGenAI Wizard — webview-based project bootstrapper.
 *
 * Mirrors the 10-step CLI wizard (MutaGenAI/wizard.py) with a
 * multi-step form inside a VS Code webview panel.  On completion it
 * generates a ready-to-run Python script, identical to the CLI output.
 */
import * as vscode from "vscode";
export declare class WizardPanel {
    private readonly extensionUri;
    static readonly viewType = "mutagenai.wizard";
    private static instance;
    private readonly panel;
    private disposables;
    static createOrShow(extensionUri: vscode.Uri): WizardPanel;
    private constructor();
    private handleMessage;
    private generateScript;
    private getHtml;
}
//# sourceMappingURL=wizardPanel.d.ts.map