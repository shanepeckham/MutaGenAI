/**
 * Sidecar — manages the Python child process for evolution runs.
 *
 * - Spawns `evolve_runner.py` with the configured Python interpreter.
 * - Sends a JSON config on stdin.
 * - Reads JSON-line events from stdout.
 * - Sends `{"type": "stop"}` on stdin for early stopping.
 */

import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { EventEmitter } from "events";
import { EvolutionConfig, SidecarEvent } from "./types";

export class Sidecar extends EventEmitter {
  private proc: cp.ChildProcess | null = null;
  private buffer = "";
  private _running = false;

  get running(): boolean {
    return this._running;
  }

  /**
   * Start an evolution run.
   *
   * @param config  Evolution configuration to send to the sidecar.
   * @returns A promise that resolves when the process exits.
   */
  start(config: EvolutionConfig): Promise<number> {
    if (this._running) {
      throw new Error("An evolution run is already in progress.");
    }

    let pythonPath = vscode.workspace
      .getConfiguration("mutagenai")
      .get<string>("pythonPath", "");

    const sidecarScript = path.join(
      __dirname,
      "..",
      "sidecar",
      "evolve_runner.py"
    );

    this._running = true;
    this.buffer = "";

    // ---- Resolve workspace root first (needed for Python detection) ----
    let workspaceCwd =
      vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "";

    if (!workspaceCwd || workspaceCwd === "/") {
      // Walk up from the sidecar script to find the project root
      let dir = path.dirname(sidecarScript);
      for (let i = 0; i < 10 && dir !== path.dirname(dir); i++) {
        if (fs.existsSync(path.join(dir, "MutaGenAI", "__init__.py"))) {
          workspaceCwd = dir;
          break;
        }
        dir = path.dirname(dir);
      }
      if (!workspaceCwd || workspaceCwd === "/") {
        workspaceCwd = process.cwd();
      }
    }

    // ---- Resolve Python interpreter ----
    if (!pythonPath) {
      // Check workspaceCwd (already resolved above)
      const venvCandidate = path.join(workspaceCwd, ".venv", "bin", "python");
      if (fs.existsSync(venvCandidate)) {
        pythonPath = venvCandidate;
      }
      // Also check workspace folders in case they differ
      if (!pythonPath) {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
          for (const folder of workspaceFolders) {
            const venvPython = path.join(folder.uri.fsPath, ".venv", "bin", "python");
            if (fs.existsSync(venvPython)) {
              pythonPath = venvPython;
              break;
            }
          }
        }
      }
      if (!pythonPath) {
        pythonPath = "python3";
      }
    }

    console.log(`[MutaGenAI] Sidecar python=${pythonPath} cwd=${workspaceCwd} script=${sidecarScript}`);
    console.log(`[MutaGenAI] Sidecar script exists: ${fs.existsSync(sidecarScript)}`);
    console.log(`[MutaGenAI] Sidecar python exists: ${pythonPath !== "python3" ? fs.existsSync(pythonPath) : "fallback"}`);
    console.log(`[MutaGenAI] workspaceFolders: ${JSON.stringify(vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath))}`);

    return new Promise<number>((resolve, reject) => {
      this.proc = cp.spawn(pythonPath, [sidecarScript, "--workspace", workspaceCwd], {
        stdio: ["pipe", "pipe", "pipe"],
        cwd: workspaceCwd,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          PYTHONPATH: workspaceCwd + (process.env.PYTHONPATH ? ":" + process.env.PYTHONPATH : ""),
        },
      });

      // Send config on stdin
      this.proc.stdin!.write(JSON.stringify(config) + "\n");
      console.log("[MutaGenAI] Config written to sidecar stdin");

      // Stream stdout line by line
      this.proc.stdout!.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf-8");
        console.log(`[MutaGenAI] stdout chunk (${text.length} bytes): ${text.substring(0, 150)}`);
        this.buffer += text;
        const lines = this.buffer.split("\n");
        // Keep incomplete last line in buffer
        this.buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) {
            continue;
          }
          try {
            const event: SidecarEvent = JSON.parse(line);
            this.emit("event", event);
          } catch {
            // Non-JSON output → treat as log
            this.emit("event", {
              type: "log",
              level: "info",
              message: line,
            } as SidecarEvent);
          }
        }
      });

      // Capture stderr for diagnostics
      this.proc.stderr!.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf-8").trim();
        if (text) {
          this.emit("event", {
            type: "log",
            level: "warn",
            message: text,
          } as SidecarEvent);
        }
      });

      this.proc.on("close", (code) => {
        console.log(`[MutaGenAI] Sidecar process closed with code=${code}`);
        this._running = false;
        this.proc = null;
        resolve(code ?? 0);
      });

      this.proc.on("error", (err) => {
        console.log(`[MutaGenAI] Sidecar process error: ${err.message}`);
        this._running = false;
        this.proc = null;
        reject(err);
      });
    });
  }

  /**
   * Send a stop signal to the running evolution.
   */
  stop(): void {
    if (this.proc?.stdin?.writable) {
      this.proc.stdin.write(JSON.stringify({ type: "stop" }) + "\n");
    }
  }

  /**
   * Kill the process immediately.
   */
  kill(): void {
    if (this.proc) {
      this.proc.kill("SIGTERM");
      this._running = false;
      this.proc = null;
    }
  }
}
