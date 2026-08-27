pub mod backprop;
pub mod data;
mod initialization;
pub mod layers;
pub mod train;

#[cfg(test)]
pub mod tests;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::fno::backprop::graph::{Context, Variable};

pub struct FNO {
    pub channels: (usize, usize, usize), // input, intermediate, output
    pub modes: (usize, usize),           // kx_max, ky_max

    pub lift_layer: layers::LinearLayer,
    pub lift_bias: Vec<Variable>,

    pub fourier_layers: Vec<layers::FourierLayer>,
    pub proj_1: layers::LinearLayer,
    pub proj_b1: Vec<Variable>,
    pub proj_2: layers::LinearLayer,
    pub proj_b2: Vec<Variable>,
}

impl FNO {
    pub fn new(
        ctx: &mut Context,
        channels: (usize, usize, usize),
        modes: (usize, usize),
        num_layers: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let lift_layer = layers::LinearLayer::new(ctx, (channels.0, channels.1), &mut rng);
        let lift_bias = vec![ctx.variable(0.0); channels.1];

        let fourier_layers = (0..num_layers)
            .map(|_| layers::FourierLayer::new(ctx, channels.1, modes, &mut rng))
            .collect();

        let proj_1 = layers::LinearLayer::new(ctx, (channels.1, channels.1), &mut rng);
        let proj_b1 = vec![ctx.variable(0.0); channels.1];

        let proj_2 = layers::LinearLayer::new(ctx, (channels.1, channels.2), &mut rng);
        let proj_b2 = vec![ctx.variable(0.0); channels.2];

        FNO {
            channels,
            modes,
            lift_layer,
            lift_bias,
            fourier_layers,
            proj_1,
            proj_b1,
            proj_2,
            proj_b2,
        }
    }

    pub fn forward(&self, ctx: &mut Context, in_dims: (usize, usize), input: &[Variable]) -> Vec<Variable> {
        let n = in_dims.0 * in_dims.1;

        let mut state = self.lift_layer.forward(ctx, in_dims, input);
        for i in 0..n {
            for channel in 0..self.channels.1 {
                let idx = i * self.channels.1 + channel;
                state[idx] = ctx.add(state[idx], self.lift_bias[channel]);
            }
        }

        for f_layer in &self.fourier_layers {
            state = f_layer.forward(ctx, in_dims, &state);
        }

        let mut state = self.proj_1.forward(ctx, in_dims, &state);
        for i in 0..n {
            for channel in 0..self.channels.1 {
                let idx = i * self.channels.1 + channel;
                let val = ctx.add(state[idx], self.proj_b1[channel]);
                state[idx] = ctx.relu(val);
            }
        }

        let mut output = self.proj_2.forward(ctx, in_dims, &state);
        for i in 0..n {
            for channel in 0..self.channels.2 {
                let idx = i * self.channels.2 + channel;
                output[idx] = ctx.add(output[idx], self.proj_b2[channel]);
            }
        }
        output
    }
}
