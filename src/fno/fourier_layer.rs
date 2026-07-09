use crate::fno::backprop::fft_traced::*;
use crate::fno::{Context, Variable};

pub struct FourierLayer {
    pub n_pixels: usize,
    pub out_channels: usize,
    pub in_channels: usize,
    pub width: usize,
    pub height: usize,
    pub modes_x: usize,
    pub modes_y: usize,

    pub w_weight: Vec<Vec<Variable>>, // Spatial bypass weight [out_channels][in_channels]
    pub r_weight_re: Vec<Vec<Vec<Vec<Variable>>>>, // Spectral weight real [out_channels][in_channels][modes_x][modes_y]
    pub r_weight_im: Vec<Vec<Vec<Vec<Variable>>>>, // Spectral weight imag [out_channels][in_channels][modes_x][modes_y]
                                                   // ... and any other configurations (modes, width, height, etc.)
}


impl FourierLayer {
    pub fn forward(&self, ctx: &mut Context, input: &[Variable]) -> Vec<Variable> {

        // branch
        let mut branch = vec![ctx.variable(0.0); self.n_pixels * self.out_channels];

        for pixel in 0..self.n_pixels {
            for co in 0..self.out_channels {
                let mut val = ctx.variable(0.0);
                for ci in 0..self.in_channels {
                    let input_val = input[pixel * self.in_channels + ci];
                    let weight = self.w_weight[co][ci];
                    let prod = ctx.mul(input_val, weight);
                    val = ctx.add(val, prod);
                }
                branch[pixel * self.out_channels + co] = val;
            }
        }

        // Fourier branch

        // Convert input to ComplexVariable and run 2D FFT per input channel
        let mut fourier_inputs =
            vec![vec![ComplexVariable::new(ctx, 0.0, 0.0); self.n_pixels]; self.in_channels];
        for ci in 0..self.in_channels {
            for pixel in 0..self.n_pixels {
                fourier_inputs[ci][pixel] = ComplexVariable {
                    re: input[pixel * self.in_channels + ci],
                    im: ctx.variable(0.0),
                };
            }
            fft_2d(ctx, &mut fourier_inputs[ci], self.width, self.height);
        }

        // Initialize output Fourier grids to zero
        let mut fourier_outputs = vec![
            vec![
                ComplexVariable {
                    re: ctx.variable(0.0),
                    im: ctx.variable(0.0),
                };
                self.n_pixels
            ];
            self.out_channels
        ];

        // Loop over the 2D frequency grid
        for row in 0..self.height {
            let ky_idx = if row < self.modes_y {
                Some(row)
            } else if row > self.height - self.modes_y {
                Some(self.height - row)
            } else {
                None
            };

            for col in 0..self.width {
                let kx_idx = if col < self.modes_x {
                    Some(col)
                } else if col > self.width - self.modes_x {
                    Some(self.width - col)
                } else {
                    None
                };

                // If this mode is kept, multiply by the complex weights
                if let (Some(kx), Some(ky)) = (kx_idx, ky_idx) {
                    let pixel_idx = row * self.width + col;

                    for co in 0..self.out_channels {
                        let mut sum = ComplexVariable {
                            re: ctx.variable(0.0),
                            im: ctx.variable(0.0),
                        };

                        for ci in 0..self.in_channels {
                            // Get complex weight R_co_ci[kx][ky]
                            let weight = ComplexVariable {
                                re: self.r_weight_re[co][ci][kx][ky],
                                im: self.r_weight_im[co][ci][kx][ky],
                            };

                            let x_val = fourier_inputs[ci][pixel_idx];
                            let prod = weight.mul(x_val, ctx); // Complex multiplication
                            sum = sum.add(prod, ctx); // Complex addition
                        }
                        fourier_outputs[co][pixel_idx] = sum;
                    }
                }
            }
        }

        // 4. Run 2D IFFT back to physical space per output channel
        let mut spectral_out = vec![ctx.variable(0.0); self.n_pixels * self.out_channels];
        for co in 0..self.out_channels {
            ifft_2d(ctx, &mut fourier_outputs[co], self.width, self.height);

            for pixel in 0..self.n_pixels {
                // Extract real part of the IFFT output
                spectral_out[pixel * self.out_channels + co] = fourier_outputs[co][pixel].re;
            }
        }
        let mut output = vec![ctx.variable(0.0); self.n_pixels * self.out_channels];
        for i in 0..output.len() {
            let sum = ctx.add(branch[i], spectral_out[i]);
            output[i] = ctx.relu(sum); // Or your activation function of choice
        }
        output
    }
}
