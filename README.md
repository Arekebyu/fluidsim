# Fluid simulation

A fluid dynamics simulation and Fourier Neural Operator written in Rust and compiled into Webassembly for ease of viewing.

This project uses two methods to compare them:
1.  A Numerical Solver of the Navier-Stokes Equation using the Pseudo-Spectral Method
2.  A Fourier Neural Operator in 2D trained on a 32x32 grid which can scaled to any $2^n\times2^n$ grid without training another model.



---

## [Interactive Web Demo](arekebyu.github.io/fluidsim)

The demo has some configurations to demonstrate the performance of the Fourier Neural Operator:
- A Side-by-Side Comparison to compare accuracy
- Scalable meshes (32x32, 64x64, and 128x128) to demonstrate mesh invariance.
- Error Heatmap to compare pointwise difference between the two solutions.

---

## Key Highlights

- Pseudo-Spectral Navier-Stokes Solver
  - Solves the 2D incompressible Navier-Stokes equations in the vorticity-streamfunction formulation: $\frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega$.
  - Exact viscous dissipation treatment using Fourier integrating factors: $\exp(-\nu |\mathbf{k}|^2 \Delta t)$.
  - 2/3 Orszag rule de-aliasing in Fourier space to dissuade spectral blocking.
- 1D and 2D Fast Fourier Transform and its Inverse (Cooley-Tukey FFT specifically)
- An automatic differentiation engine supporting the above FFT
- Fourier Layers and Convolution Layers
  - Lifting Layers, Residuals, and Projection layers are all implemented as Convolutions to allow for Mesh-invariance.
  - Fourier Layers comprises of a Residual branch and a Layer to truncate and linear transform inputs in Frequency space. These two are summed then fed into a ReLU
- Training:
  - Initial Conditions are generated with a Gaussian Random Field to create a divergence-free initial condition
  - Each initial condition is then advanced $k$ steps by the numerical solver, each is stored into a dataset.
  - The FNO performs $n<k$ autoregression steps on each of the $k-n$ values of the dataset. We then compare the dataset and prediction to compute loss with the $L^2 norm$.
  - Weights are then tuned with Adam optimizer.
---

## Training the Fourier Neural Operator

The FNO is currently trained entirely on CPU, and can be done by running the script, demonstrated below:

```bash
cargo run --release --bin train -- \
  --num-x 32 \
  --num-y 32 \
  --viscosity 0.005 \
  --num-data 10 \
  --num-epochs 25 \
  --dt 0.01 \
  --rollout 4 \
  --modes-x 12 \
  --modes-y 12 \
  --layer-channels 4 \
  --num-layers 4 \
  --output-dir models
```

### Key Training Options

| Flag | Default | Description |
|---|---|---|
| `--num-x`, `--nx` | `32` | Spatial grid resolution along the X axis |
| `--num-y`, `--ny` | `32` | Spatial grid resolution along the Y axis |
| `--viscosity`, `--nu` | `0.005` | Kinematic viscosity coefficient ($\nu$) |
| `--num-data` | `10` | Number of distinct fluid initial conditions to generate |
| `--num-epochs` | `25` | Training epochs over the dataset |
| `--dt` | `0.01` | Physical time step between frames |
| `--rollout` | `16` | Number of Autoregressive steps to take to compute loss |
| `--lr` | `0.005` | Adam optimizer learning rate |
| `--modes-x` | `8` | Truncated Fourier modes retained along X ($k_{x,\text{max}}$) |
| `--modes-y` | `8` | Truncated Fourier modes retained along Y ($k_{y,\text{max}}$) |
| `--layer-channels` | `4` | Latent channel dimension $d_v$ across Fourier layers |
| `--num-layers` | `4` | Number of stacked Fourier spectral convolution layers |
| `--output-dir` | `models` | Directory to write serialized binary weights |

Trained weights are serialized into `models/fno_weights.bin` alongside `models/model_metadata.json`.
