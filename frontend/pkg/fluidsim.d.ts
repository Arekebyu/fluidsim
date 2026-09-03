/* tslint:disable */
/* eslint-disable */

export enum ModelPreset {
    Untrained = 0,
    Epoch5 = 1,
    Epoch25 = 2,
    Custom = 3,
}

export class WebSimulator {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Add an interactive vortex perturbation at normalized coordinate (cx, cy)
     */
    add_vortex(cx_norm: number, cy_norm: number, strength: number, radius_norm: number): void;
    get_active_preset(): number;
    /**
     * Compute relative L2 error between FNO prediction and Ground-Truth Physics
     */
    get_relative_l2_error(): number;
    /**
     * Exposes a direct pointer to the RGBA render buffer in WebAssembly memory
     */
    get_render_buffer_ptr(): number;
    get_render_height(): number;
    get_render_width(mode: number): number;
    /**
     * Load custom binary weights from user upload
     */
    load_custom_weights(weights: Float32Array): void;
    constructor(width: number, height: number, viscosity: number, preset: number);
    /**
     * Render simulation fields into the RGBA render buffer:
     * mode 0 = Side-by-Side (Ground Truth | FNO)
     * mode 1 = Ground Truth only
     * mode 2 = FNO only
     * mode 3 = Absolute Difference Heatmap
     */
    render_to_rgba(mode: number, v_max: number): void;
    /**
     * Regenerate divergence-free initial fluid conditions using Gaussian Random Fields
     */
    reset_with_grf(alpha: number, tau: number, target_std: number, seed: bigint): void;
    /**
     * Change the active FNO model checkpoint (Untrained, Epoch 5, Epoch 25)
     */
    set_model_preset(preset: number): void;
    /**
     * Set resolution dynamically (Demonstrating Mesh Invariance / Zero-Shot Super-Resolution)
     */
    set_resolution(new_width: number, new_height: number, seed: bigint): void;
    /**
     * Advance both the Ground Truth solver and the FNO surrogate simultaneously
     */
    step_both(dt: number): void;
    dt: number;
    height: number;
    viscosity: number;
    width: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_get_websimulator_dt: (a: number) => number;
    readonly __wbg_get_websimulator_height: (a: number) => number;
    readonly __wbg_get_websimulator_viscosity: (a: number) => number;
    readonly __wbg_get_websimulator_width: (a: number) => number;
    readonly __wbg_set_websimulator_dt: (a: number, b: number) => void;
    readonly __wbg_set_websimulator_height: (a: number, b: number) => void;
    readonly __wbg_set_websimulator_viscosity: (a: number, b: number) => void;
    readonly __wbg_set_websimulator_width: (a: number, b: number) => void;
    readonly __wbg_websimulator_free: (a: number, b: number) => void;
    readonly websimulator_add_vortex: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly websimulator_get_active_preset: (a: number) => number;
    readonly websimulator_get_relative_l2_error: (a: number) => number;
    readonly websimulator_get_render_buffer_ptr: (a: number) => number;
    readonly websimulator_get_render_height: (a: number) => number;
    readonly websimulator_get_render_width: (a: number, b: number) => number;
    readonly websimulator_load_custom_weights: (a: number, b: number, c: number) => void;
    readonly websimulator_new: (a: number, b: number, c: number, d: number) => number;
    readonly websimulator_render_to_rgba: (a: number, b: number, c: number) => void;
    readonly websimulator_reset_with_grf: (a: number, b: number, c: number, d: number, e: bigint) => void;
    readonly websimulator_set_model_preset: (a: number, b: number) => void;
    readonly websimulator_set_resolution: (a: number, b: number, c: number, d: bigint) => void;
    readonly websimulator_step_both: (a: number, b: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
