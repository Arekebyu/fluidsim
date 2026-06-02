use crate::math::backprop::graph::*;

/// A custom complex number representation for spectral computations.
#[derive(Clone, Copy)]
pub struct ComplexVariable {
    pub re: Variable,
    pub im: Variable,
}

impl ComplexVariable {
    pub fn new(ctx: &mut Context, re: f32, im: f32) -> Self {
        let re = ctx.variable(re);
        let im = ctx.variable(im);

        ComplexVariable { re, im }
    }

    pub fn add(self, other: Self, ctx: &mut Context) -> Self {
        let re = ctx.add(self.re, other.re);
        let im = ctx.add(self.im, other.im);
        ComplexVariable { re, im }
    }

    pub fn sub(self, other: Self, ctx: &mut Context) -> Self {
        let re = ctx.sub(self.re, other.re);
        let im = ctx.sub(self.im, other.im);
        ComplexVariable { re, im }
    }

    pub fn mul(self, other: Self, ctx: &mut Context) -> Self {
        let r1 = ctx.mul(self.re, other.re);
        let r2 = ctx.mul(self.im, other.im);
        let re = ctx.sub(r1, r2);
        let i1 = ctx.mul(self.re, other.im);
        let i2 = ctx.mul(self.im, other.re);
        let im = ctx.add(i1, i2);
        ComplexVariable { re, im }
    }
}

// --- Cooley-Tukey Radix-2 1D FFT ---

pub fn fft_1d(ctx: &mut Context, data: &mut [ComplexVariable]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT size must be a power of two");

    // butterfly
    let mut j = 0;
    for i in 0..n {
        if i < j {
            data.swap(i, j)
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j ^= m
    }

    // transform
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f32::consts::PI / len as f32;
        let w_step = ComplexVariable {
            re: ctx.variable(f32::cos(angle)),
            im: ctx.variable(f32::sin(angle)),
        };

        for i in (0..n).step_by(len) {
            let mut w = ComplexVariable {
                re: ctx.variable(1.0),
                im: ctx.variable(0.0),
            };
            for j in 0..half {
                let u = data[i + j];
                let t = w.mul(data[i + j + half], ctx);
                data[i + j] = u.add(t, ctx);
                data[i + j + half] = u.sub(t, ctx);
                w = w.mul(w_step, ctx);
            }
        }
        len *= 2;
    }
}

/// Computes the 1D Inverse Fast Fourier Transform in-place.
///
/// Hint: You can reuse fft_1d!
/// Conjugating the inputs, running the forward FFT, and then conjugating
/// the outputs and dividing by N is mathematically equivalent to the IFFT.
pub fn ifft_1d(ctx: &mut Context, data: &mut [ComplexVariable]) {
    let n = data.len();
    for x in data.iter_mut() {
        let zero = ctx.variable(0.0);
        let conj_im = ctx.sub(zero, x.im);
        *x = ComplexVariable {
            re: x.re,
            im: conj_im,
        };
    }
    fft_1d(ctx, data);

    let s = ctx.variable(1.0 / n as f32);
    for x in data.iter_mut() {
        let zero = ctx.variable(0.0);
        let conj_im = ctx.sub(zero, x.im);
        *x = ComplexVariable {
            re: ctx.mul(x.re, s),
            im: ctx.mul(conj_im, s),
        };
    }
}

// --- 2D FFT via Row-Column Decomposition ---

pub fn row_col_apply(
    ctx: &mut Context,
    data: &mut [ComplexVariable],
    width: usize,
    height: usize,
    mut f: impl FnMut(&mut Context, &mut [ComplexVariable]),
) {
    for row in 0..height {
        let row_index = row * width;
        f(ctx, &mut data[row_index..row_index + width]);
    }

    let mut columns = vec![
        ComplexVariable {
            re: ctx.variable(0.0,),
            im: ctx.variable(0.0)
        };
        height
    ];
    for col in 0..width {
        for row in 0..height {
            columns[row] = data[col + row * width];
        }
        f(ctx, &mut columns);
        for row in 0..height {
            data[col + row * width] = columns[row];
        }
    }
}

/// Computes the 2D FFT in-place on a row-major flat vector.
///
/// Hint:
/// 1. Loop through each row and run fft_1d.
/// 2. Since columns are not contiguous, extract each column into a temporary
///    vector, run fft_1d on it, and copy the results back.
pub fn fft_2d(ctx: &mut Context, data: &mut [ComplexVariable], width: usize, height: usize) {
    assert_eq!(data.len(), width * height);
    assert!(width.is_power_of_two());
    assert!(height.is_power_of_two());
    row_col_apply(ctx, data, width, height, fft_1d);
}

/// Computes the 2D IFFT in-place on a row-major flat vector.
pub fn ifft_2d(ctx: &mut Context, data: &mut [ComplexVariable], width: usize, height: usize) {
    assert_eq!(data.len(), width * height);
    assert!(width.is_power_of_two());
    assert!(height.is_power_of_two());
    row_col_apply(ctx, data, width, height, ifft_1d);
}
