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
import * as vscode from "vscode";
import { SidecarEvent } from "./types";
export declare class EvolvePanel {
    static readonly viewType = "mutagenai.evolvePanel";
    private static currentPanel;
    private static logFn;
    private readonly panel;
    private readonly extensionUri;
    private disposables;
    private webviewReady;
    private eventBuffer;
    private fallbackTimer;
    /** Wire up a logger (call before createOrShow). */
    static setLogger(fn: (msg: string) => void): void;
    private _log;
    private constructor();
    /**
     * Create or reveal the panel.
     */
    static createOrShow(extensionUri: vscode.Uri): EvolvePanel;
    /**
     * Forward a sidecar event to the webview.
     * Buffers events until the webview signals it is ready.
     */
    postEvent(event: SidecarEvent): void;
    /** Flush all buffered events to the webview. */
    private _flushBuffer;
    private dispose;
    /**
     * Register a handler for messages FROM the webview (e.g. stop button).
     */
    onMessage(handler: (msg: {
        command: string;
    }) => void): void;
    private _getNonce;
    private getHtml;
    private getMinimalHtml;
}
//# sourceMappingURL=evolvePanel.d.ts.map