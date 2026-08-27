// This file is currently a black box because I don't actually
// have enough mathematical knowledge when implementing this
// so I just transcribed the code into rust
use serde::{Deserialize, Serialize};

use crate::calculations::fft::{self, ComplexNum};

#[derive(Clone, Serialize, Deserialize)]
pub struct Grid {
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub vorticity: Vec<f32>,

    pub dx: f32,
    pub dy: f32,
    pub x_res: usize,
    pub y_res: usize,
    pub viscosity: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub x_bound: f32,
    pub y_bound: f32,
    pub x_res: usize,
    pub y_res: usize,
    pub viscosity: f32,
}

#[derive(Serialize, Deserialize)]
pub struct InitialConditions {
    pub vorticity: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
}

impl Grid {
    pub fn new(cfg: &Config, initial_conditions: InitialConditions) -> Self {
        let dy = cfg.y_bound / cfg.y_res as f32;
        let dx = cfg.x_bound / cfg.x_res as f32;

        Grid {
            vx: initial_conditions.vx,
            vy: initial_conditions.vy,
            vorticity: initial_conditions.vorticity,
            dx,
            dy,
            x_res: cfg.x_res,
            y_res: cfg.y_res,
            viscosity: cfg.viscosity,
        }
    }

    fn get_k(&self, x: usize, y: usize) -> (f32, f32) {
        let n_x = self.x_res;
        let n_y = self.y_res;

        let k_x = if x < n_x / 2 {
            x as f32
        } else {
            (x as isize - n_x as isize) as f32
        };
        let k_y = if y < n_y / 2 {
            y as f32
        } else {
            (y as isize - n_y as isize) as f32
        };

        (k_x, k_y)
    }

    fn compute_advection_fourier(&self, omega_fourier: &[fft::ComplexNum]) -> Vec<fft::ComplexNum> {
        let n_pixels = self.x_res * self.y_res;

        let mut u = vec![ComplexNum::zero(); n_pixels];
        let mut v = vec![ComplexNum::zero(); n_pixels];
        let mut dwdx = vec![ComplexNum::zero(); n_pixels];
        let mut dwdy = vec![ComplexNum::zero(); n_pixels];
        for y in 0..self.y_res {
            for x in 0..self.x_res {
                let mut psi = fft::ComplexNum::zero();
                let idx = y * self.x_res + x;
                let (kx, ky) = self.get_k(x, y);
                let k_sq = kx * kx + ky * ky;
                if k_sq > 0.0 {
                    psi = omega_fourier[idx].scale(1.0 / k_sq);
                }
                u[idx] = ComplexNum::new(-ky * psi.im, ky * psi.re);
                v[idx] = ComplexNum::new(kx * psi.im, -kx * psi.re);
                dwdx[idx] =
                    ComplexNum::new(-kx * omega_fourier[idx].im, kx * omega_fourier[idx].re);
                dwdy[idx] =
                    ComplexNum::new(-ky * omega_fourier[idx].im, ky * omega_fourier[idx].re);
            }
        }

        fft::ifft_2d(&mut u, self.x_res, self.y_res);
        fft::ifft_2d(&mut v, self.x_res, self.y_res);
        fft::ifft_2d(&mut dwdx, self.x_res, self.y_res);
        fft::ifft_2d(&mut dwdy, self.x_res, self.y_res);

        let mut advection = vec![ComplexNum::zero(); n_pixels];
        for i in 0..n_pixels {
            let val = -(u[i].re * dwdx[i].re + v[i].re * dwdy[i].re);
            advection[i] = ComplexNum::new(val, 0.0);
        }

        fft::fft_2d(&mut advection, self.x_res, self.y_res);

        for y in 0..self.y_res {
            for x in 0..self.x_res {
                let idx = y * self.x_res + x;
                let (kx, ky) = self.get_k(x, y);
                let limit_x = self.x_res as f32 / 3.0;
                let limit_y = self.y_res as f32 / 3.0;
                if kx.abs() > limit_x || ky.abs() > limit_y {
                    advection[idx] = ComplexNum::zero();
                }
            }
        }

        advection
    }
    pub fn step(&mut self, dt: f32) {
        let n = self.y_res * self.x_res;
        let mut w: Vec<fft::ComplexNum> = self
            .vorticity
            .iter()
            .map(|&val| fft::ComplexNum { re: val, im: 0.0 })
            .collect();
        fft::fft_2d(&mut w, self.x_res, self.y_res);

        let mut exp_factor = vec![0.0; n];
        for y in 0..self.y_res {
            for x in 0..self.x_res {
                let (kx, ky) = self.get_k(x, y);
                let k_sq = kx.powi(2) + ky.powi(2);
                exp_factor[x + y * self.x_res] = (-self.viscosity * k_sq * dt).exp()
            }
        }
        let n_n = self.compute_advection_fourier(&w);

        let mut w_pred = vec![ComplexNum::zero(); n];
        for i in 0..n {
            w_pred[i] = w[i].add(n_n[i].scale(dt)).scale(exp_factor[i]);
        }
        let n_pred = self.compute_advection_fourier(&w_pred);

        //Heun's trapezoidal method
        for i in 0..n {
            let term1 = w[i].add(n_n[i].scale(dt * 0.5)).scale(exp_factor[i]);
            let term2 = n_pred[i].scale(dt * 0.5);
            w[i] = term1.add(term2);
        }

        fft::ifft_2d(&mut w, self.x_res, self.y_res);
        for (i, w) in w.into_iter().enumerate() {
            self.vorticity[i] = w.re;
        }
    }
}
