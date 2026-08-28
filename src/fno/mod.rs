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

    pub fn forward(
        &self,
        ctx: &mut Context,
        in_dims: (usize, usize),
        input: &[Variable],
    ) -> Vec<Variable> {
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

    pub fn collect_weights(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.lift_layer.collect_weights(&mut vars);
        vars.extend(&self.lift_bias);

        for layer in &self.fourier_layers {
            layer.collect_variables(&mut vars);
        }

        self.proj_1.collect_weights(&mut vars);
        vars.extend(&self.proj_b1);
        self.proj_2.collect_weights(&mut vars);
        vars.extend(&self.proj_b2);

        vars
    }

    pub fn from_weights(
        ctx: &mut Context,
        channels: (usize, usize, usize),
        modes: (usize, usize),
        num_layers: usize,
        weights: &[f32],
    ) -> Self {
        let mut cur = weights;

        let lift_size = channels.0 * channels.1;
        let (lift_w, rest) = cur.split_at(lift_size);
        let lift_layer = layers::LinearLayer::from_weights(ctx, (channels.0, channels.1), lift_w);
        cur = rest;

        let (lift_b, rest) = cur.split_at(channels.1);
        let lift_bias = lift_b.iter().map(|&val| ctx.variable(val)).collect();
        cur = rest;

        let fourier_size =
            channels.1 * channels.1 + 2 * channels.1 * channels.1 * modes.0 * modes.1;
        let mut fourier_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let (f_w, rest) = cur.split_at(fourier_size);
            let layer = layers::FourierLayer::from_weights(ctx, channels.1, modes, f_w);
            fourier_layers.push(layer);
            cur = rest;
        }

        let proj1_size = channels.1 * channels.1;
        let (p1_w, rest) = cur.split_at(proj1_size);
        let proj_1 = layers::LinearLayer::from_weights(ctx, (channels.1, channels.1), p1_w);
        cur = rest;

        let (p1_b, rest) = cur.split_at(channels.1);
        let proj_b1 = p1_b.iter().map(|&val| ctx.variable(val)).collect();
        cur = rest;

        let proj2_size = channels.1 * channels.2;
        let (p2_w, rest) = cur.split_at(proj2_size);
        let proj_2 = layers::LinearLayer::from_weights(ctx, (channels.1, channels.2), p2_w);
        cur = rest;

        let (p2_b, rest) = cur.split_at(channels.2);
        let proj_b2 = p2_b.iter().map(|&val| ctx.variable(val)).collect();
        cur = rest;

        assert!(cur.is_empty(), "wrong input parameters?");

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
}
