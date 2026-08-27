use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;

use crate::fno::backprop::fft_traced::*;
use crate::fno::{Context, Variable};

pub struct LinearLayer {
    pub channels: (usize, usize), // in, out
    pub w: Vec<Vec<Variable>>,    // [out_channel][in_channel]
}

impl LinearLayer {
    pub fn new(
        ctx: &mut Context,
        channels: (usize, usize),
        rng: &mut StdRng,
    ) -> LinearLayer {
        let a = (1.0 / channels.0 as f32).sqrt() * 0.5;
        let between = Uniform::try_from(-a..a).unwrap();
        let w = (0..channels.1)
            .map(|_| {
                (0..channels.0)
                    .map(|_| ctx.variable(between.sample(rng)))
                    .collect()
            })
            .collect();
        LinearLayer { channels, w }
    }

    pub fn forward(
        &self,
        ctx: &mut Context,
        in_dims: (usize, usize),
        input: &[Variable],
    ) -> Vec<Variable> {
        let n = in_dims.0 * in_dims.1;

        let mut output = vec![ctx.variable(0.0); n * self.channels.1];
        for i in 0..n {
            for out_channel in 0..self.channels.1 {
                let mut sum = ctx.variable(0.0);
                for in_channel in 0..self.channels.0 {
                    let inp = input[i * self.channels.0 + in_channel];
                    let w = self.w[out_channel][in_channel];
                    let prod = ctx.mul(inp, w);
                    sum = ctx.add(sum, prod);
                }
                output[i * self.channels.1 + out_channel] = sum;
            }
        }
        output
    }
}

pub struct FourierLayer {
    pub channels: usize,         // d_v
    pub modes: (usize, usize),   // num x modes, num y modes
    pub residual: LinearLayer,
    pub r: Vec<Vec<Vec<Vec<(Variable, Variable)>>>>, // [out_channel][in_channel][modes_x][modes_y]
}

impl FourierLayer {
    pub fn new(
        ctx: &mut Context,
        channels: usize,
        modes: (usize, usize),
        rng: &mut StdRng,
    ) -> Self {
        let residual = LinearLayer::new(ctx, (channels, channels), rng);

        let a = (1.0 / channels as f32).sqrt() * 0.5;
        let between = Uniform::try_from(-a..a).unwrap();
        // it's ok because I still kept the triangle
        let r = (0..channels)
            .map(|_| {
                (0..channels)
                    .map(|_| {
                        (0..modes.0)
                            .map(|_| {
                                (0..modes.1)
                                    .map(|_| {
                                        (
                                            ctx.variable(between.sample(rng)),
                                            ctx.variable(between.sample(rng)),
                                        )
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        FourierLayer {
            channels,
            modes,
            residual,
            r,
        }
    }

    pub fn forward(&self, ctx: &mut Context, in_dims: (usize, usize), input: &[Variable]) -> Vec<Variable> {
        let n = in_dims.0 * in_dims.1;

        // compute residual
        let w_val = self.residual.forward(ctx, in_dims, input);

        // fourier part
        let mut f_inp = vec![vec![ComplexVariable::new(ctx, 0.0, 0.0); n]; self.channels];
        for in_channel in 0..self.channels {
            for i in 0..n {
                f_inp[in_channel][i] = ComplexVariable {
                    re: input[i * self.channels + in_channel],
                    im: ctx.variable(0.0),
                };
            }
            fft_2d(ctx, &mut f_inp[in_channel], in_dims.0, in_dims.1);
        }

        let mut f_out = vec![vec![ComplexVariable::new(ctx, 0.0, 0.0); n]; self.channels];

        let y_range = (0..self.modes.1.min(in_dims.1))
            .map(|ky| (ky, ky))
            .chain((1..self.modes.1.min(in_dims.1)).map(|ky| (in_dims.1 - ky, ky)));

        let x_range = (0..self.modes.0.min(in_dims.0))
            .map(|kx| (kx, kx))
            .chain((1..self.modes.0.min(in_dims.0)).map(|kx| (in_dims.0 - kx, kx)));

        for (row, ky) in y_range {
            for (col, kx) in x_range.clone() {
                let i = row * in_dims.0 + col;

                for output_channel in 0..self.channels {
                    let mut sum = ComplexVariable {
                        re: ctx.variable(0.0),
                        im: ctx.variable(0.0),
                    };

                    for input_channel in 0..self.channels {
                        let (re, im) = self.r[output_channel][input_channel][kx][ky];
                        let weight = ComplexVariable { re, im };
                        let prod = weight.mul(f_inp[input_channel][i], ctx);
                        sum = sum.add(prod, ctx);
                    }
                    f_out[output_channel][i] = sum;
                }
            }
        }

        let mut ret = vec![ctx.variable(0.0); n * self.channels];
        for output_channel in 0..self.channels {
            ifft_2d(
                ctx,
                &mut f_out[output_channel],
                in_dims.0,
                in_dims.1,
            );
            for i in 0..n {
                let idx = i * self.channels + output_channel;
                let spectral_re = f_out[output_channel][i].re;
                let sum = ctx.add(w_val[idx], spectral_re);
                ret[idx] = ctx.relu(sum);
            }
        }
        ret
    }
}
