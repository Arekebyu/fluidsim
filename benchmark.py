import jax
import jax.numpy as jnp
from jax import jit
import time
import os
import subprocess
import numpy as np
import argparse

# Force JAX to use CPU if you want to compare CPU-to-CPU performance,
# or let JAX auto-detect and use GPU if available.
jax.config.update('jax_platform_name', 'cpu')

def compute_advection_fourier(w_fourier, Nx, Ny, dx, dy):
    # Setup wave numbers
    kx = jnp.fft.fftfreq(Nx, d=dx/(2*jnp.pi))
    ky = jnp.fft.fftfreq(Ny, d=dy/(2*jnp.pi))
    KX, KY = jnp.meshgrid(kx, ky)
    K_sq = KX**2 + KY**2
    K_sq = K_sq.at[0, 0].set(1.0) # Avoid division by zero
    
    # 1. Solve for streamfunction
    psi_fourier = w_fourier / K_sq
    psi_fourier = psi_fourier.at[0, 0].set(0.0)
    
    # 2. Reconstruct velocities in Fourier space
    u_fourier = 1j * KY * psi_fourier
    v_fourier = -1j * KX * psi_fourier
    
    # 3. Compute gradients of vorticity
    dwdx_fourier = 1j * KX * w_fourier
    dwdy_fourier = 1j * KY * w_fourier
    
    # 4. Transform to physical space (extract real part)
    u = jnp.fft.ifft2(u_fourier).real
    v = jnp.fft.ifft2(v_fourier).real
    dwdx = jnp.fft.ifft2(dwdx_fourier).real
    dwdy = jnp.fft.ifft2(dwdy_fourier).real
    
    # 5. Advection in physical space
    advection_phys = -(u * dwdx + v * dwdy)
    
    # 6. Transform back and apply 2/3 dealiasing
    advection_fourier = jnp.fft.fft2(advection_phys)
    dealias = (jnp.abs(KX) < Nx/3) & (jnp.abs(KY) < Ny/3)
    
    return advection_fourier * dealias

@jit(static_argnums=(3, 4))  # Compiles this entire function (and helper calls) into an optimized kernel
def step_rk2_if_jax(w_phys, nu, dt, Nx, Ny, dx, dy):
    w_fourier = jnp.fft.fft2(w_phys)
    
    kx = jnp.fft.fftfreq(Nx, d=dx/(2*jnp.pi))
    ky = jnp.fft.fftfreq(Ny, d=dy/(2*jnp.pi))
    KX, KY = jnp.meshgrid(kx, ky)
    K_sq = KX**2 + KY**2
    
    exp_factor = jnp.exp(-nu * K_sq * dt)
    
    n_n = compute_advection_fourier(w_fourier, Nx, Ny, dx, dy)
    w_pred = (w_fourier + dt * n_n) * exp_factor
    
    n_pred = compute_advection_fourier(w_pred, Nx, Ny, dx, dy)
    w_next_fourier = (w_fourier + 0.5 * dt * n_n) * exp_factor + 0.5 * dt * n_pred
    
    return jnp.fft.ifft2(w_next_fourier).real

def main():
    parser = argparse.ArgumentParser(description="Benchmark Rust pseudo-spectral fluid solver vs Python JAX reference.")
    parser.add_argument("--nx", type=int, default=128, help="Grid size in X (power of two)")
    parser.add_argument("--ny", type=int, default=128, help="Grid size in Y (power of two)")
    parser.add_argument("--nu", type=float, default=0.005, help="Viscosity coefficient")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step size")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    
    args = parser.parse_args()
    Nx = args.nx
    Ny = args.ny
    nu = args.nu
    dt = args.dt
    steps = args.steps

    x_bound = 2.0 * np.pi
    y_bound = 2.0 * np.pi
    dx = x_bound / Nx
    dy = y_bound / Ny

    # Clean previous binary runs
    for f in ["initial_vorticity.bin", "rust_vorticity.bin"]:
        if os.path.exists(f):
            os.remove(f)

    # 1. Run Rust Solver with parameters
    print(f"Building and running Rust solver (N={Nx}, Y={Ny}, nu={nu}, dt={dt}, steps={steps})...")
    

    cmd = ["cargo", "run", "--release", "--", str(Nx), str(Ny), str(nu), str(dt), str(steps)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("\033[91m[Error] Rust compilation or execution failed:\033[0m")
        print(result.stderr)
        return

    print(result.stdout.strip())

    if not os.path.exists("initial_vorticity.bin") or not os.path.exists("rust_vorticity.bin"):
        print("\033[91m[Error] Rust output files were not generated.\033[0m")
        return

    # 2. Read initial conditions exported by Rust (convert to JAX device array)
    w_init_np = np.fromfile("initial_vorticity.bin", dtype=np.float32).reshape((Ny, Nx))
    w_init = jnp.array(w_init_np)
    
    w_rust = np.fromfile("rust_vorticity.bin", dtype=np.float32).reshape((Ny, Nx))

    # 3. Run Python JAX Solver
    print(f"Running JAX JIT-compiled Python solver...")
    
    # Warm-up step: Compile the JIT kernel before timing
    w_py = step_rk2_if_jax(w_init, nu, dt, Nx, Ny, dx, dy)
    w_py.block_until_ready() # Wait for compilation and execution to finish

    # Perform benchmark run
    w_py = w_init
    start_time = time.perf_counter()
    for _ in range(steps):
        w_py = step_rk2_if_jax(w_py, nu, dt, Nx, Ny, dx, dy)
    
    # Wait for the asynchronous queue to finish before reading timer
    w_py.block_until_ready()
    end_time = time.perf_counter()
    py_time_ms = (end_time - start_time) * 1000.0

    print(f"Python (JAX JIT) execution time: {py_time_ms:.2f} ms")

    # 4. Correctness Check
    w_py_np = np.array(w_py)
    max_diff = np.max(np.abs(w_rust - w_py_np))
    mean_diff = np.mean(np.abs(w_rust - w_py_np))
    
    print("\n--- Correctness Comparison ---")
    print(f"Maximum absolute difference: {max_diff:.3e}")
    print(f"Mean absolute difference: {mean_diff:.3e}")

    if max_diff < 1e-2:
        print("\033[92m[PASS]\033[0m Rust solver matches the Python JAX reference solver!")
    else:
        print("\033[91m[FAIL]\033[0m Mismatch between Rust and Python solvers. Verify your RK2 stepping logic.")

if __name__ == "__main__":
    main()
