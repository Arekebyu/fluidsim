import init, { WasmSolver } from "./pkg/fluidsim.js";

const canvas = document.getElementById("window") as HTMLCanvasElement;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

async function run() {
    // 1. Initialize WASM module and retrieve memory export
    const wasm = await init();

    // 2. Configure initial grid conditions
    const xRes = 100;
    const yRes = 100;
    const dx = 1.0 / xRes;
    const dy = 1.0 / yRes;

    // Create initial velocities flat array of size xRes * yRes * 2
    const initialVelocities = new Float64Array(xRes * yRes * 2);
    for (let i = 0; i < yRes; i++) {
        for (let j = 0; j < xRes; j++) {
            const x = j * dx;
            const y = i * dy;
            const idx = (i * xRes + j) * 2;
            // sin(x), sin(y) mirroring main.rs
            initialVelocities[idx] = Math.sin(10 * x);
            initialVelocities[idx + 1] = Math.sin(10 * y);
        }
    }

    // 3. Create WasmSolver instance and initialize it
    const solver = new WasmSolver();
    solver.init(1.0, 1.0, xRes, yRes, 0.00001, initialVelocities);

    const ptr = solver.get_velocities_ptr();
    const len = solver.get_velocities_len();

    // 4. Start rendering loop
    function loop() {
        // Step the simulation inside WebAssembly
        solver.step(1);

        // Wrap the WASM memory buffer directly (Zero-Copy)
        const velocityData = new Float64Array(wasm.memory.buffer, ptr, len);

        // Render current velocities into ImageData for maximum performance
        const imgData = ctx.createImageData(canvas.width, canvas.height);
        const data = imgData.data;

        for (let y = 0; y < canvas.height; y++) {
            const gridY = Math.floor((y / canvas.height) * yRes);
            for (let x = 0; x < canvas.width; x++) {
                const gridX = Math.floor((x / canvas.width) * xRes);

                const idx = (gridY * xRes + gridX) * 2;
                const vx = velocityData[idx];
                const vy = velocityData[idx + 1];
                const speed = Math.sqrt(vx * vx + vy * vy);

                // Beautiful color mapping: speed as intensity
                const intensity = Math.min(speed * 200, 255);

                const pixelIdx = (y * canvas.width + x) * 4;
                data[pixelIdx] = intensity;          // Red: high speed = brighter red/pink
                data[pixelIdx + 1] = intensity * 0.3;  // Green: subtle orange/pink hue
                data[pixelIdx + 2] = 255 - intensity;  // Blue: low speed = blue
                data[pixelIdx + 3] = 255;              // Alpha (fully opaque)
            }
        }

        ctx.putImageData(imgData, 0, 0);
        requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);
}

run().catch(console.error);