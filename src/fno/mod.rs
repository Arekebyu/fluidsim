pub mod backprop;
pub mod fourier_layer;

#[cfg(test)]
pub mod tests;

use fourier_layer::FourierLayer;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::fno::backprop::graph::{Context, Variable};

pub struct FNO {
    pub lifting_w: Vec<Vec<Variable>>, // [layer_channels][in_channels]
    pub lifting_b: Vec<Variable>,      // [layer_channels]

    // Fourier Layers
    pub fourier_layers: Vec<FourierLayer>,

    // Projection Layer parameters (Direct fields)
    pub proj_w1: Vec<Vec<Variable>>, // [layer_channels][layer_channels]
    pub proj_b1: Vec<Variable>,
    pub proj_w2: Vec<Vec<Variable>>, // [out_channels][layer_channels]
    pub proj_b2: Vec<Variable>,

    pub width: usize,
    pub height: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub layer_channels: usize,
    pub modes_x: usize,
    pub modes_y: usize,
}

impl FNO {
    pub fn new(
        ctx: &mut Context,
        seed: u64,
        width: usize,
        height: usize,
        in_channels: usize,
        out_channels: usize,
        layer_channels: usize,
        modes_x: usize,
        modes_y: usize,
        num_layers: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let init_2d = |dim1: usize, dim2: usize, rng: &mut StdRng, ctx: &mut Context| {
            let scale = (1.0 / dim2 as f32).sqrt();
            (0..dim1)
                .map(|_| {
                    (0..dim2)
                        .map(|_| ctx.variable(rng.random_range(-1.0..=1.0) * scale))
                        .collect()
                })
                .collect()
        };

        // Lifting Layer Initialization: shape [layer_channels][in_channels]
        let lifting_w: Vec<Vec<Variable>> = init_2d(layer_channels, in_channels, &mut rng, ctx);
        let lifting_b = vec![ctx.variable(0.0); layer_channels];

        // Fourier Layers Initialization: shape [layer_channels][layer_channels] for hidden representation
        let mut fourier_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let w_weight: Vec<Vec<Variable>> =
                init_2d(layer_channels, layer_channels, &mut rng, ctx);

            let r_weight_re: Vec<Vec<Vec<Vec<Variable>>>> = (0..layer_channels)
                .map(|_| {
                    (0..layer_channels)
                        .map(|_| init_2d(modes_x, modes_y, &mut rng, ctx))
                        .collect()
                })
                .collect();

            let r_weight_im: Vec<Vec<Vec<Vec<Variable>>>> = (0..layer_channels)
                .map(|_| {
                    (0..layer_channels)
                        .map(|_| init_2d(modes_x, modes_y, &mut rng, ctx))
                        .collect()
                })
                .collect();

            let layer = FourierLayer {
                n_pixels: width * height,
                width,
                height,
                in_channels: layer_channels,
                out_channels: layer_channels,
                modes_x,
                modes_y,
                w_weight,
                r_weight_re,
                r_weight_im,
            };
            fourier_layers.push(layer);
        }

        // Projection Layer 1 Initialization: shape [layer_channels][layer_channels]
        let proj_w1 = init_2d(layer_channels, layer_channels, &mut rng, ctx);
        let proj_b1 = vec![ctx.variable(0.0); layer_channels];

        // Projection Layer 2 Initialization: shape [out_channels][layer_channels]
        let proj_w2 = init_2d(out_channels, layer_channels, &mut rng, ctx);
        let proj_b2 = vec![ctx.variable(0.0); out_channels];

        FNO {
            lifting_w,
            lifting_b,
            fourier_layers,
            proj_w1,
            proj_b1,
            proj_w2,
            proj_b2,
            width,
            height,
            in_channels,
            out_channels,
            layer_channels,
            modes_x,
            modes_y,
        }
    }

    pub fn forward(&self, ctx: &mut Context, input: &[Variable]) -> Vec<Variable> {
        let n_pixels = self.width * self.height;

        // Lifting step (pointwise projection from in_channels to layer_channels)
        let mut state = vec![ctx.variable(0.0); n_pixels * self.layer_channels];
        for pixel in 0..n_pixels {
            for co in 0..self.layer_channels {
                let mut sum = self.lifting_b[co];
                for ci in 0..self.in_channels {
                    let in_val = input[pixel * self.in_channels + ci];
                    let prod = ctx.mul(in_val, self.lifting_w[co][ci]);
                    sum = ctx.add(sum, prod);
                }
                state[pixel * self.layer_channels + co] = sum;
            }
        }

        // Pass through Fourier layers sequentially
        for layer in &self.fourier_layers {
            state = layer.forward(ctx, &state);
        }

        // Projection step (two pointwise layers: projection 1 + projection 2)
        let mut output = vec![ctx.variable(0.0); n_pixels * self.out_channels];
        for pixel in 0..n_pixels {
            // MLP Layer 1 (layer_channels -> layer_channels + relu)
            let mut h = vec![ctx.variable(0.0); self.layer_channels];
            for co in 0..self.layer_channels {
                let mut sum = self.proj_b1[co];
                for ci in 0..self.layer_channels {
                    let val = state[pixel * self.layer_channels + ci];
                    let prod = ctx.mul(val, self.proj_w1[co][ci]);
                    sum = ctx.add(sum, prod);
                }
                h[co] = ctx.relu(sum);
            }

            // MLP Layer 2 (layer_channels -> out_channels, linear)
            for co in 0..self.out_channels {
                let mut sum = self.proj_b2[co];
                for ci in 0..self.layer_channels {
                    let prod = ctx.mul(h[ci], self.proj_w2[co][ci]);
                    sum = ctx.add(sum, prod);
                }
                output[pixel * self.out_channels + co] = sum;
            }
        }

        output
    }
}
