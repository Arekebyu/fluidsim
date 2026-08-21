use crate::{calculations::fft::{self, ComplexNum}, solver::InitialConditions};
use rand::{RngExt, rngs::StdRng, seq::index::sample};

fn sample_gaussian(rng: &mut StdRng) -> ComplexNum {
    //box-muller transformation implementation
    let u1: f32 = rng.random_range(1e-7..1.0);
    let u2: f32 = rng.random_range(0.0..2.0 * std::f32::consts::PI);
    let r = (-2.0 * u1.ln()).sqrt();
    ComplexNum{re: u2.cos(), im: u2.sin()}.scale(r)
}

/// generate divergence free ic 
pub fn generate_initial_conditions(
    width: usize,
    height: usize,
    alpha: f32,
    tau: f32,
    target_std: f32,
    rng: &mut StdRng,
) -> InitialConditions {
    let n_pixels = width * height;

    let mut w_f = vec![ComplexNum::zero(); n_pixels];
    let mut vx_f = vec![ComplexNum::zero(); n_pixels];
    let mut vy_f = vec![ComplexNum::zero(); n_pixels];

    for y in 0..height {
        let ky = if y < height / 2 {
            y as f32
        } else {
            (y as isize - height as isize) as f32
        };

        for x in 0..width {
            let kx = if x < width / 2 {
                x as f32
            } else {
                (x as isize - width as isize) as f32
            };

            let idx = y * width + x;
            let k_sq = kx * kx + ky * ky;

            if k_sq > 0.0 {
                // noise with power spectrum for vorticity
                let amplitude = (k_sq + tau * tau).powf(-alpha / 2.0);
                let w_hat = sample_gaussian(rng).scale(amplitude);
                w_f[idx] = w_hat;

                // streamfunction \psi = \omega / k^2
                let psi = w_hat.scale(1.0 / k_sq);

                //vx = i ky \psi,  vy = -i ks \psi
                vx_f[idx] = ComplexNum::new(-ky * psi.im, ky * psi.re);
                vy_f[idx] = ComplexNum::new(kx * psi.im, -kx * psi.re);
            }
        }
    }

    fft::ifft_2d(&mut w_f, width, height);
    fft::ifft_2d(&mut vx_f, width, height);
    fft::ifft_2d(&mut vy_f, width, height);

    // map to real
    let mut vorticity: Vec<f32> = w_f.into_iter().map(|c| c.re).collect();
    let mut vx: Vec<f32> = vx_f.into_iter().map(|c| c.re).collect();
    let mut vy: Vec<f32> = vy_f.into_iter().map(|c| c.re).collect();

    // normalize vorticity and velocities
    let E_v = vorticity.iter().sum::<f32>() / n_pixels as f32;
    let var = vorticity.iter().map(|&v| (v - E_v).powi(2)).sum::<f32>() / n_pixels as f32;
    let scale = target_std / var.sqrt().max(1e-7);

    for i in 0..n_pixels {
        // normalize
        vorticity[i] = (vorticity[i] - E_v) * scale;
        vx[i] *= scale;
        vy[i] *= scale;
    }

    InitialConditions { vorticity, vx, vy }
}
