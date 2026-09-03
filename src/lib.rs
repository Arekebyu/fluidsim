pub mod calculations;
pub mod fno;
pub mod messages;
pub mod solver;

use wasm_bindgen::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::fno::FNO;
use crate::fno::backprop::graph::{Context, Variable};
use crate::fno::initialization::{ICConfig, generate_initial_conditions};
use crate::solver::{Config, Grid};

// Embedded pre-trained checkpoint model weights (Compile-time Method 1)
static CHECKPOINT_0: &[u8] = include_bytes!("../models/checkpoint_epoch_0.bin");
static CHECKPOINT_5: &[u8] = include_bytes!("../models/checkpoint_epoch_5.bin");
static CHECKPOINT_25: &[u8] = include_bytes!("../models/checkpoint_epoch_25.bin");

fn bytes_to_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelPreset {
    Untrained = 0,
    Epoch5 = 1,
    Epoch25 = 2,
    Custom = 3,
}

#[wasm_bindgen]
pub struct WebSimulator {
    pub width: usize,
    pub height: usize,
    pub viscosity: f32,
    pub dt: f32,

    // Ground-Truth Physical Solver (Navier-Stokes Pseudo-Spectral)
    gt_solver: Grid,

    // Neural Operator Solver (FNO)
    fno: FNO,
    fno_ctx: Context,
    fno_vorticity: Vec<f32>,
    active_weights: Vec<f32>,
    active_preset: ModelPreset,

    // Shared RGBA Pixel Render Buffer for zero-copy Canvas display
    render_buffer: Vec<u8>,
}

#[wasm_bindgen]
impl WebSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, viscosity: f32, preset: usize) -> Self {
        let active_preset = match preset {
            0 => ModelPreset::Untrained,
            1 => ModelPreset::Epoch5,
            _ => ModelPreset::Epoch25,
        };

        let weights = match active_preset {
            ModelPreset::Untrained => bytes_to_floats(CHECKPOINT_0),
            ModelPreset::Epoch5 => bytes_to_floats(CHECKPOINT_5),
            ModelPreset::Epoch25 | ModelPreset::Custom => bytes_to_floats(CHECKPOINT_25),
        };

        let mut fno_ctx = Context::default();
        let fno = FNO::from_weights(&mut fno_ctx, (1, 4, 1), (4, 4), 2, &weights);

        let cfg = Config {
            x_bound: 2.0 * std::f32::consts::PI,
            y_bound: 2.0 * std::f32::consts::PI,
            x_res: width,
            y_res: height,
            viscosity,
        };

        let mut rng = StdRng::seed_from_u64(42);
        let ic = generate_initial_conditions(
            (width, height),
            ICConfig {
                alpha: 2.5,
                tau: 3.0,
                target_std: 1.0,
            },
            &mut rng,
        );

        let fno_vorticity = ic.vorticity.clone();
        let gt_solver = Grid::new(&cfg, ic);

        // Max possible render buffer: Side-by-Side (2 * width) * height * 4 RGBA bytes
        let render_buffer = vec![0u8; (width * 2) * height * 4];

        WebSimulator {
            width,
            height,
            viscosity,
            dt: 0.01,
            gt_solver,
            fno,
            fno_ctx,
            fno_vorticity,
            active_weights: weights,
            active_preset,
            render_buffer,
        }
    }

    /// Change the active FNO model checkpoint (Untrained, Epoch 5, Epoch 25)
    pub fn set_model_preset(&mut self, preset: usize) {
        self.active_preset = match preset {
            0 => ModelPreset::Untrained,
            1 => ModelPreset::Epoch5,
            _ => ModelPreset::Epoch25,
        };

        self.active_weights = match self.active_preset {
            ModelPreset::Untrained => bytes_to_floats(CHECKPOINT_0),
            ModelPreset::Epoch5 => bytes_to_floats(CHECKPOINT_5),
            ModelPreset::Epoch25 | ModelPreset::Custom => bytes_to_floats(CHECKPOINT_25),
        };

        self.fno_ctx = Context::default();
        self.fno = FNO::from_weights(&mut self.fno_ctx, (1, 4, 1), (4, 4), 2, &self.active_weights);
    }

    /// Load custom binary weights from user upload
    pub fn load_custom_weights(&mut self, weights: &[f32]) {
        self.active_preset = ModelPreset::Custom;
        self.active_weights = weights.to_vec();
        self.fno_ctx = Context::default();
        self.fno = FNO::from_weights(&mut self.fno_ctx, (1, 4, 1), (4, 4), 2, &self.active_weights);
    }

    /// Regenerate divergence-free initial fluid conditions using Gaussian Random Fields
    pub fn reset_with_grf(&mut self, alpha: f32, tau: f32, target_std: f32, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let ic = generate_initial_conditions(
            (self.width, self.height),
            ICConfig {
                alpha,
                tau,
                target_std,
            },
            &mut rng,
        );

        let cfg = Config {
            x_bound: 2.0 * std::f32::consts::PI,
            y_bound: 2.0 * std::f32::consts::PI,
            x_res: self.width,
            y_res: self.height,
            viscosity: self.viscosity,
        };

        self.fno_vorticity = ic.vorticity.clone();
        self.gt_solver = Grid::new(&cfg, ic);
    }

    /// Advance both the Ground Truth solver and the FNO surrogate simultaneously
    pub fn step_both(&mut self, dt: f32) {
        self.dt = dt;

        // 1. Advance Ground-Truth Navier-Stokes Physics
        self.gt_solver.step(dt);

        // 2. Advance Fourier Neural Operator
        let input: Vec<Variable> = self
            .fno_vorticity
            .iter()
            .map(|&v| self.fno_ctx.variable(v))
            .collect();

        let pred = self.fno.forward(&mut self.fno_ctx, (self.width, self.height), &input);
        self.fno_vorticity = pred.iter().map(|&v| self.fno_ctx.get_val(v)).collect();

        // 3. Clear graph memory for next frame
        self.fno_ctx = Context::default();
        self.fno = FNO::from_weights(&mut self.fno_ctx, (1, 4, 1), (4, 4), 2, &self.active_weights);
    }

    /// Set resolution dynamically (Demonstrating Mesh Invariance / Zero-Shot Super-Resolution)
    pub fn set_resolution(&mut self, new_width: usize, new_height: usize, seed: u64) {
        self.width = new_width;
        self.height = new_height;
        self.render_buffer = vec![0u8; (new_width * 2) * new_height * 4];

        self.reset_with_grf(2.5, 3.0, 1.0, seed);
    }

    /// Add an interactive vortex perturbation at normalized coordinate (cx, cy)
    pub fn add_vortex(&mut self, cx_norm: f32, cy_norm: f32, strength: f32, radius_norm: f32) {
        let cx = cx_norm * self.width as f32;
        let cy = cy_norm * self.height as f32;
        let r_sq = (radius_norm * self.width.min(self.height) as f32).powi(2).max(1.0);

        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist_sq = dx * dx + dy * dy;
                let delta = strength * (-dist_sq / (2.0 * r_sq)).exp();

                let idx = y * self.width + x;
                self.gt_solver.vorticity[idx] += delta;
                self.fno_vorticity[idx] += delta;
            }
        }
    }

    /// Compute relative L2 error between FNO prediction and Ground-Truth Physics
    pub fn get_relative_l2_error(&self) -> f32 {
        let mut diff_sq_sum = 0.0;
        let mut gt_sq_sum = 0.0;

        for i in 0..self.width * self.height {
            let diff = self.fno_vorticity[i] - self.gt_solver.vorticity[i];
            diff_sq_sum += diff * diff;
            gt_sq_sum += self.gt_solver.vorticity[i] * self.gt_solver.vorticity[i];
        }

        (diff_sq_sum / gt_sq_sum.max(1e-7)).sqrt()
    }

    /// Render simulation fields into the RGBA render buffer:
    /// mode 0 = Side-by-Side (Ground Truth | FNO)
    /// mode 1 = Ground Truth only
    /// mode 2 = FNO only
    /// mode 3 = Absolute Difference Heatmap
    pub fn render_to_rgba(&mut self, mode: usize, v_max: f32) {
        let max_val = v_max.max(1e-4);

        if mode == 0 {
            // Side-by-Side: width_total = width * 2
            let total_w = self.width * 2;
            for y in 0..self.height {
                for x in 0..self.width {
                    // Left half: Ground Truth
                    let idx_gt = y * self.width + x;
                    let val_gt = self.gt_solver.vorticity[idx_gt];
                    let (r1, g1, b1) = vorticity_colormap(val_gt / max_val);

                    let out_idx_left = (y * total_w + x) * 4;
                    self.render_buffer[out_idx_left] = r1;
                    self.render_buffer[out_idx_left + 1] = g1;
                    self.render_buffer[out_idx_left + 2] = b1;
                    self.render_buffer[out_idx_left + 3] = 255;

                    // Right half: FNO Neural Surrogate
                    let val_fno = self.fno_vorticity[idx_gt];
                    let (r2, g2, b2) = vorticity_colormap(val_fno / max_val);

                    let out_idx_right = (y * total_w + (x + self.width)) * 4;
                    self.render_buffer[out_idx_right] = r2;
                    self.render_buffer[out_idx_right + 1] = g2;
                    self.render_buffer[out_idx_right + 2] = b2;
                    self.render_buffer[out_idx_right + 3] = 255;
                }
            }
        } else {
            // Single Viewport (width x height)
            for i in 0..self.width * self.height {
                let (r, g, b) = match mode {
                    1 => vorticity_colormap(self.gt_solver.vorticity[i] / max_val),
                    2 => vorticity_colormap(self.fno_vorticity[i] / max_val),
                    _ => {
                        // Error Heatmap (0 -> black/blue, large -> bright yellow/red)
                        let err = (self.fno_vorticity[i] - self.gt_solver.vorticity[i]).abs() / max_val;
                        error_colormap(err)
                    }
                };

                let out_idx = i * 4;
                self.render_buffer[out_idx] = r;
                self.render_buffer[out_idx + 1] = g;
                self.render_buffer[out_idx + 2] = b;
                self.render_buffer[out_idx + 3] = 255;
            }
        }
    }

    /// Exposes a direct pointer to the RGBA render buffer in WebAssembly memory
    pub fn get_render_buffer_ptr(&self) -> *const u8 {
        self.render_buffer.as_ptr()
    }

    pub fn get_render_width(&self, mode: usize) -> usize {
        if mode == 0 {
            self.width * 2
        } else {
            self.width
        }
    }

    pub fn get_render_height(&self) -> usize {
        self.height
    }

    pub fn get_active_preset(&self) -> usize {
        self.active_preset as usize
    }
}

/// Diverging Cool-Warm Colormap for Vorticity
#[inline]
fn vorticity_colormap(normalized: f32) -> (u8, u8, u8) {
    let t = normalized.clamp(-1.0, 1.0);
    if t < 0.0 {
        // Negative vorticity (Blue to White)
        let f = -t; // 0 (white) to 1 (deep blue)
        let r = ((1.0 - f) * 230.0 + f * 40.0) as u8;
        let g = ((1.0 - f) * 230.0 + f * 80.0) as u8;
        let b = ((1.0 - f) * 230.0 + f * 220.0) as u8;
        (r, g, b)
    } else {
        // Positive vorticity (White to Crimson Red)
        let f = t; // 0 (white) to 1 (vibrant red)
        let r = ((1.0 - f) * 230.0 + f * 220.0) as u8;
        let g = ((1.0 - f) * 230.0 + f * 35.0) as u8;
        let b = ((1.0 - f) * 230.0 + f * 35.0) as u8;
        (r, g, b)
    }
}

/// Sequential Colormap for Absolute Error Heatmaps
#[inline]
fn error_colormap(err: f32) -> (u8, u8, u8) {
    let t = (err * 2.0).clamp(0.0, 1.0);
    let r = (t * 255.0) as u8;
    let g = ((1.0 - (t - 0.5).abs() * 2.0).max(0.0) * 220.0) as u8;
    let b = ((1.0 - t) * 100.0) as u8;
    (r, g, b)
}
