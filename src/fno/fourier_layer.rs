use std::iter::Chain;
use std::ops::Range;
use either::Either;

use crate::fno::backprop::fft_traced::*;
use crate::fno::{Context, Variable};

pub struct FourierLayer {
    pub in_dims: (usize, usize),                        // width, height
    pub channels: usize,                                // d_v
    pub modes: (usize, usize),                          // num x modes, num y modes
                                         
    pub w: Vec<Vec<Variable>>,                          // W in the diagram
    pub r: Vec<Vec<Vec<Vec<(Variable, Variable)>>>>,    // R in diagram, also \kappa
        // w_weight has dimensions [output neuron][input neuron]
        // r has dimensions [output][input][modes_x][modes_y] and is complex.
    
}

impl FourierLayer {
    pub fn forward(&self, ctx: &mut Context, input: &[Variable]) -> Vec<Variable> {
        let n = self.in_dims.0 * self.in_dims.1;

        // compute residual
        let mut w_val = vec![ctx.variable(0.0); n * self.channels];
        for i in 0..n {
            for out_channel in 0..self.channels {
                let mut val = &mut w_val[i * self.channels + out_channel];
                for in_channel in 0..self.channels {
                    // input has dimensions [x][y][channels] but is flattened
                    let inp = input[i * n + in_channel];
                    let w = self.w[out_channel][in_channel];
                    let prod = ctx.mul(inp, w);
                    *val = ctx.add(*val, prod);
                }
            }
        }

        // fourier part
        // fourier transform f
        let mut f_inp = vec![Vec::with_capacity(n) ; self.channels];
            // vec![vec![ComplexVariable::new(ctx, 0.0, 0.0); n]; self.channels];
        for in_channel in 0..self.channels {
            for i in 0..n {
                f_inp[in_channel][i] = ComplexVariable {
                    re: input[i * self.channels + in_channel],
                    im: ctx.variable(0.0),
                };
            }
            fft_2d(ctx, &mut f_inp[in_channel], self.in_dims.0, self.in_dims.1);
        }

        let mut f_out: Vec<Vec<ComplexVariable>> = vec![Vec::with_capacity(n) ; self.channels];

        let y_range = (0..self.modes.1.min(self.in_dims.1))
            .map(|ky| (ky, ky))
            .chain((1..self.modes.1).map(|ky| (self.in_dims.1 - ky, ky)));

        let x_range = (0..self.modes.0.min(self.in_dims.0))
            .map(|kx| (kx, kx))
            .chain((1..self.modes.0).map(|kx| (self.in_dims.0 - kx, kx)));

        // not very simd friendly, should adapt
        for (row, ky) in y_range {
            for (col, kx) in x_range.clone() {
                let i = row * self.in_dims.0 + col;

                for output_channel in 0..self.channels {
                    let mut sum = ComplexVariable {
                        re: ctx.variable(0.0),
                        im: ctx.variable(0.0),
                    };

                    for input_channel in 0..self.channels {
                        let (re, im) = self.r[output_channel][input_channel][kx][ky];
                        let weight = ComplexVariable {re, im};
                        let prod = weight.mul(f_inp[input_channel][i], ctx);
                        sum = sum.add(prod, ctx);
                    }
                    f_out[output_channel][i] = sum;
                }
            }
        }

        let mut ret = Vec::with_capacity(n * self.channels);
        for output_channel in 0..self.channels {
            ifft_2d(ctx, &mut f_out[output_channel], self.in_dims.0, self.in_dims.1);
            for i in 0..n {
                let idx = i + self.channels + output_channel;
                ret[idx] = f_out[output_channel][i].re;
                ret[idx] = ctx.add(w_val[idx], ret[idx]);
                ret[idx] = ctx.relu(ret[idx]);
            }
        }
        ret
        
    }
}
