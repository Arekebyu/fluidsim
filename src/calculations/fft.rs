#[derive(Clone, Copy)]
pub struct ComplexNum {
    pub re: f32,
    pub im: f32,
}

impl ComplexNum {
    pub const fn new(re: f32, im: f32) -> Self {
        ComplexNum { re, im }
    }

    pub const fn scale(self, scale: f32) -> Self {
        ComplexNum {
            re: self.re * scale,
            im: self.im * scale,
        }
    }

    pub const fn zero() -> Self {
        ComplexNum { re: 0.0, im: 0.0 }
    }

    pub const fn add(self, other: Self) -> Self {
        ComplexNum {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub const fn sub(self, other: Self) -> Self {
        ComplexNum {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub const fn mul(self, other: Self) -> Self {
        ComplexNum {
            re: self.re * other.re - self.im * other.im,
            im: self.im * other.re + self.re * other.im,
        }
    }
}

pub fn fft_1d(data: &mut [ComplexNum]) {
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
        let w_step = ComplexNum {
            re: f32::cos(angle),
            im: f32::sin(angle),
        };

        for i in (0..n).step_by(len) {
            let mut w = ComplexNum { re: 1.0, im: 0.0 };
            for j in 0..half {
                let u = data[i + j];
                let t = w.mul(data[i + j + half]);
                data[i + j] = u.add(t);
                data[i + j + half] = u.sub(t);
                w = w.mul(w_step);
            }
        }
        len *= 2;
    }
}

pub fn ifft_1d(data: &mut [ComplexNum]) {
    let n = data.len();
    for x in data.iter_mut() {
        *x = ComplexNum {
            re: x.re,
            im: -x.im,
        };
    }
    fft_1d(data);

    let s = 1.0 / n as f32;
    for x in data.iter_mut() {
        *x = x.scale(s);
    }
}

// --- 2D FFT via Row-Column Decomposition ---

fn row_col_apply(
    data: &mut [ComplexNum],
    width: usize,
    height: usize,
    mut f: impl FnMut(&mut [ComplexNum]),
) {
    for row in 0..height {
        let row_index = row * width;
        f(&mut data[row_index..row_index + width]);
    }

    let mut columns = vec![ComplexNum::zero(); height];
    for col in 0..width {
        for row in 0..height {
            columns[row] = data[col + row * width];
        }
        f(&mut columns);
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
pub fn fft_2d(data: &mut [ComplexNum], width: usize, height: usize) {
    assert_eq!(data.len(), width * height);
    assert!(width.is_power_of_two());
    assert!(height.is_power_of_two());
    row_col_apply(data, width, height, fft_1d);
}

/// Computes the 2D IFFT in-place on a row-major flat vector.
pub fn ifft_2d(data: &mut [ComplexNum], width: usize, height: usize) {
    assert_eq!(data.len(), width * height);
    assert!(width.is_power_of_two());
    assert!(height.is_power_of_two());
    row_col_apply(data, width, height, ifft_1d);
}
