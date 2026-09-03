import init, { WebSimulator } from './pkg/fluidsim.js';

let wasmModule = null;
let sim = null;

// Simulation State
let isPlaying = true;
let currentRes = 64;
let currentPreset = 2; // Epoch 25
let displayMode = 0; // 0 = Side-by-Side
let vMax = 3.0;
let dt = 0.010;

// Performance Tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();
let lastFrameTime = performance.now();

// DOM Elements
const canvas = document.getElementById('fluid-canvas');
const ctx = canvas.getContext('2d');
const btnPlay = document.getElementById('btn-play');
const btnStep = document.getElementById('btn-step');
const btnReset = document.getElementById('btn-reset');
const selectCheckpoint = document.getElementById('checkpoint-select');
const checkpointTag = document.getElementById('checkpoint-tag');
const selectMode = document.getElementById('display-mode');
const sliderVmax = document.getElementById('vmax-slider');
const valVmax = document.getElementById('vmax-val');
const sliderDt = document.getElementById('dt-slider');
const valDt = document.getElementById('dt-val');
const resButtons = document.querySelectorAll('#resolution-buttons .btn-toggle');
const headerLabels = document.getElementById('canvas-header-labels');

// Metric Elements
const metricFps = document.getElementById('metric-fps');
const metricLatency = document.getElementById('metric-latency');
const metricError = document.getElementById('metric-error');
const metricRes = document.getElementById('metric-res');
const metricParams = document.getElementById('metric-params');

async function start() {
    wasmModule = await init();
    console.log("[WASM] Neural Fluid Simulator module initialized.");

    // Initialize WebSimulator
    sim = new WebSimulator(currentRes, currentRes, 0.005, currentPreset);
    updateCanvasDimensions();
    setupEventListeners();
    requestAnimationFrame(renderLoop);
}

function updateCanvasDimensions() {
    const w = sim.get_render_width(displayMode);
    const h = sim.get_render_height();
    canvas.width = w;
    canvas.height = h;

    const container = canvas.parentElement;
    if (displayMode === 0) {
        container.style.aspectRatio = "2 / 1";
        headerLabels.style.display = "flex";
    } else {
        container.style.aspectRatio = "1 / 1";
        headerLabels.style.display = "none";
    }
}

function setupEventListeners() {
    // Play / Pause
    btnPlay.addEventListener('click', () => {
        isPlaying = !isPlaying;
        btnPlay.querySelector('span').textContent = isPlaying ? 'Pause' : 'Play';
        btnPlay.classList.toggle('btn-primary', isPlaying);
        btnPlay.classList.toggle('btn-secondary', !isPlaying);
    });

    // Step Once
    btnStep.addEventListener('click', () => {
        if (!isPlaying) {
            sim.step_both(dt);
            drawFrame();
        }
    });

    // Reset with Gaussian Random Field
    btnReset.addEventListener('click', () => {
        const seed = BigInt(Math.floor(Math.random() * 1000000));
        sim.reset_with_grf(2.5, 3.0, 1.0, seed);
    });

    // Checkpoint Selection
    selectCheckpoint.addEventListener('change', (e) => {
        currentPreset = parseInt(e.target.value);
        sim.set_model_preset(currentPreset);
        const tags = ["Untrained (Epoch 0)", "Intermediate (Epoch 5)", "Converged (Epoch 25)"];
        checkpointTag.textContent = tags[currentPreset] || "Custom";
    });

    // Resolution Selection (Mesh Invariance)
    resButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            resButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentRes = parseInt(btn.dataset.res);
            const seed = BigInt(Math.floor(Math.random() * 1000000));
            sim.set_resolution(currentRes, currentRes, seed);
            metricRes.textContent = `${currentRes} x ${currentRes}`;
            updateCanvasDimensions();
        });
    });

    // Display Mode Selection
    selectMode.addEventListener('change', (e) => {
        displayMode = parseInt(e.target.value);
        updateCanvasDimensions();
    });

    // Colormap scale
    sliderVmax.addEventListener('input', (e) => {
        vMax = parseFloat(e.target.value);
        valVmax.textContent = vMax.toFixed(1);
    });

    // Timestep scale
    sliderDt.addEventListener('input', (e) => {
        dt = parseFloat(e.target.value);
        valDt.textContent = dt.toFixed(3);
    });

    // Mouse Interaction: Click & Drag to stir vortices
    let isMouseDown = false;

    function handleInteraction(clientX, clientY, strength) {
        const rect = canvas.getBoundingClientRect();
        let normX = (clientX - rect.left) / rect.width;
        let normY = (clientY - rect.top) / rect.height;

        if (displayMode === 0) {
            // For side-by-side mode, map interaction across both halves
            if (normX > 0.5) {
                normX = (normX - 0.5) * 2.0;
            } else {
                normX = normX * 2.0;
            }
        }

        normX = Math.max(0.0, Math.min(1.0, normX));
        normY = Math.max(0.0, Math.min(1.0, normY));

        sim.add_vortex(normX, normY, strength, 0.08);
    }

    canvas.addEventListener('mousedown', (e) => {
        isMouseDown = true;
        handleInteraction(e.clientX, e.clientY, 3.5);
    });

    window.addEventListener('mousemove', (e) => {
        if (isMouseDown) {
            handleInteraction(e.clientX, e.clientY, 2.5);
        }
    });

    window.addEventListener('mouseup', () => {
        isMouseDown = false;
    });

    // Touch Support for Mobile
    canvas.addEventListener('touchstart', (e) => {
        if (e.touches.length > 0) {
            isMouseDown = true;
            handleInteraction(e.touches[0].clientX, e.touches[0].clientY, 3.5);
            e.preventDefault();
        }
    });

    canvas.addEventListener('touchmove', (e) => {
        if (isMouseDown && e.touches.length > 0) {
            handleInteraction(e.touches[0].clientX, e.touches[0].clientY, 2.5);
            e.preventDefault();
        }
    });

    canvas.addEventListener('touchend', () => {
        isMouseDown = false;
    });
}

function drawFrame() {
    sim.render_to_rgba(displayMode, vMax);

    const ptr = sim.get_render_buffer_ptr();
    const w = sim.get_render_width(displayMode);
    const h = sim.get_render_height();
    const len = w * h * 4;

    const bytes = new Uint8ClampedArray(wasmModule.memory.buffer, ptr, len);
    const imgData = new ImageData(bytes, w, h);
    ctx.putImageData(imgData, 0, 0);

    // Update Relative L2 Error
    const error = sim.get_relative_l2_error() * 100.0;
    metricError.textContent = `${error.toFixed(2)}%`;
}

function renderLoop(now) {
    frameCount++;

    // Update FPS once per second
    if (now - lastFpsUpdate >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
        metricFps.textContent = `${fps} FPS`;
        frameCount = 0;
        lastFpsUpdate = now;
    }

    // Step Physics & Neural Operator
    if (isPlaying) {
        const t0 = performance.now();
        sim.step_both(dt);
        const latency = performance.now() - t0;
        metricLatency.textContent = `${latency.toFixed(2)} ms`;
    }

    drawFrame();
    requestAnimationFrame(renderLoop);
}

start().catch(err => {
    console.error("[Error] Failed to initialize fluid simulation:", err);
});
