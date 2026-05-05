/**
 * Sidecar — manages the Python child process for evolution runs.
 *
 * - Spawns `evolve_runner.py` with the configured Python interpreter.
 * - Sends a JSON config on stdin.
 * - Reads JSON-line events from stdout.
 * - Sends `{"type": "stop"}` on stdin for early stopping.
 */
import { EventEmitter } from "events";
import { EvolutionConfig } from "./types";
export declare class Sidecar extends EventEmitter {
    private proc;
    private buffer;
    private _running;
    get running(): boolean;
    /**
     * Start an evolution run.
     *
     * @param config  Evolution configuration to send to the sidecar.
     * @returns A promise that resolves when the process exits.
     */
    start(config: EvolutionConfig): Promise<number>;
    /**
     * Send a stop signal to the running evolution.
     */
    stop(): void;
    /**
     * Kill the process immediately.
     */
    kill(): void;
}
//# sourceMappingURL=sidecar.d.ts.map