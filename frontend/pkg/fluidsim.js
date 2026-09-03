/* @ts-self-types="./fluidsim.d.ts" */

/**
 * @enum {0 | 1 | 2 | 3}
 */
export const ModelPreset = Object.freeze({
    Untrained: 0, "0": "Untrained",
    Epoch5: 1, "1": "Epoch5",
    Epoch25: 2, "2": "Epoch25",
    Custom: 3, "3": "Custom",
});

export class WebSimulator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WebSimulatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_websimulator_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get dt() {
        const ret = wasm.__wbg_get_websimulator_dt(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get height() {
        const ret = wasm.__wbg_get_websimulator_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get viscosity() {
        const ret = wasm.__wbg_get_websimulator_viscosity(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get width() {
        const ret = wasm.__wbg_get_websimulator_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set dt(arg0) {
        wasm.__wbg_set_websimulator_dt(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set height(arg0) {
        wasm.__wbg_set_websimulator_height(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set viscosity(arg0) {
        wasm.__wbg_set_websimulator_viscosity(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set width(arg0) {
        wasm.__wbg_set_websimulator_width(this.__wbg_ptr, arg0);
    }
    /**
     * Add an interactive vortex perturbation at normalized coordinate (cx, cy)
     * @param {number} cx_norm
     * @param {number} cy_norm
     * @param {number} strength
     * @param {number} radius_norm
     */
    add_vortex(cx_norm, cy_norm, strength, radius_norm) {
        wasm.websimulator_add_vortex(this.__wbg_ptr, cx_norm, cy_norm, strength, radius_norm);
    }
    /**
     * @returns {number}
     */
    get_active_preset() {
        const ret = wasm.websimulator_get_active_preset(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Compute relative L2 error between FNO prediction and Ground-Truth Physics
     * @returns {number}
     */
    get_relative_l2_error() {
        const ret = wasm.websimulator_get_relative_l2_error(this.__wbg_ptr);
        return ret;
    }
    /**
     * Exposes a direct pointer to the RGBA render buffer in WebAssembly memory
     * @returns {number}
     */
    get_render_buffer_ptr() {
        const ret = wasm.websimulator_get_render_buffer_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_render_height() {
        const ret = wasm.websimulator_get_render_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} mode
     * @returns {number}
     */
    get_render_width(mode) {
        const ret = wasm.websimulator_get_render_width(this.__wbg_ptr, mode);
        return ret >>> 0;
    }
    /**
     * Load custom binary weights from user upload
     * @param {Float32Array} weights
     */
    load_custom_weights(weights) {
        const ptr0 = passArrayF32ToWasm0(weights, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        wasm.websimulator_load_custom_weights(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} width
     * @param {number} height
     * @param {number} viscosity
     * @param {number} preset
     */
    constructor(width, height, viscosity, preset) {
        const ret = wasm.websimulator_new(width, height, viscosity, preset);
        this.__wbg_ptr = ret;
        WebSimulatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Render simulation fields into the RGBA render buffer:
     * mode 0 = Side-by-Side (Ground Truth | FNO)
     * mode 1 = Ground Truth only
     * mode 2 = FNO only
     * mode 3 = Absolute Difference Heatmap
     * @param {number} mode
     * @param {number} v_max
     */
    render_to_rgba(mode, v_max) {
        wasm.websimulator_render_to_rgba(this.__wbg_ptr, mode, v_max);
    }
    /**
     * Regenerate divergence-free initial fluid conditions using Gaussian Random Fields
     * @param {number} alpha
     * @param {number} tau
     * @param {number} target_std
     * @param {bigint} seed
     */
    reset_with_grf(alpha, tau, target_std, seed) {
        wasm.websimulator_reset_with_grf(this.__wbg_ptr, alpha, tau, target_std, seed);
    }
    /**
     * Change the active FNO model checkpoint (Untrained, Epoch 5, Epoch 25)
     * @param {number} preset
     */
    set_model_preset(preset) {
        wasm.websimulator_set_model_preset(this.__wbg_ptr, preset);
    }
    /**
     * Set resolution dynamically (Demonstrating Mesh Invariance / Zero-Shot Super-Resolution)
     * @param {number} new_width
     * @param {number} new_height
     * @param {bigint} seed
     */
    set_resolution(new_width, new_height, seed) {
        wasm.websimulator_set_resolution(this.__wbg_ptr, new_width, new_height, seed);
    }
    /**
     * Advance both the Ground Truth solver and the FNO surrogate simultaneously
     * @param {number} dt
     */
    step_both(dt) {
        wasm.websimulator_step_both(this.__wbg_ptr, dt);
    }
}
if (Symbol.dispose) WebSimulator.prototype[Symbol.dispose] = WebSimulator.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_bb96b2010945f0bc: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
    };
    return {
        __proto__: null,
        "./fluidsim_bg.js": import0,
    };
}

const WebSimulatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_websimulator_free(ptr, 1));

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('fluidsim_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
