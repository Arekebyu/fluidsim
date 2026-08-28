use super::fft_traced::{ComplexVariable as Complex, fft_1d, fft_2d, ifft_1d, ifft_2d};
use super::graph::{Context, Variable};

#[derive(Debug, Clone, Copy)]
struct FloatComplex {
    re: f32,
    im: f32,
}

impl FloatComplex {
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

impl std::ops::Add for FloatComplex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Mul for FloatComplex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

fn slow_dft(input: &[FloatComplex]) -> Vec<FloatComplex> {
    let n = input.len();
    let mut output = vec![FloatComplex::zero(); n];
    for k in 0..n {
        let mut sum = FloatComplex::zero();
        for j in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
            let twiddle = FloatComplex::new(angle.cos(), angle.sin());
            sum = sum + input[j] * twiddle;
        }
        output[k] = sum;
    }
    output
}

// --- Unit Tests ---

#[test]
fn test_complex_operators() {
    let mut ctx = Context::default();
    let a = Complex {
        re: ctx.variable(1.0),
        im: ctx.variable(2.0),
    };
    let b = Complex {
        re: ctx.variable(3.0),
        im: ctx.variable(4.0),
    };

    // Test Addition
    let add_res = a.add(b, &mut ctx);
    assert!((ctx.get_val(add_res.re) - 4.0).abs() < 1e-5);
    assert!((ctx.get_val(add_res.im) - 6.0).abs() < 1e-5);

    // Test Subtraction
    let sub_res = a.sub(b, &mut ctx);
    assert!((ctx.get_val(sub_res.re) - -2.0).abs() < 1e-5);
    assert!((ctx.get_val(sub_res.im) - -2.0).abs() < 1e-5);

    // Test Multiplication
    let mul_res = a.mul(b, &mut ctx);
    assert!((ctx.get_val(mul_res.re) - -5.0).abs() < 1e-5);
    assert!((ctx.get_val(mul_res.im) - 10.0).abs() < 1e-5);
}

#[test]
fn test_fft_vs_dft() {
    let mut ctx = Context::default();
    let float_input = vec![
        FloatComplex::new(1.0, 0.0),
        FloatComplex::new(2.0, -1.0),
        FloatComplex::new(0.0, 3.0),
        FloatComplex::new(-1.5, 1.5),
    ];

    let dft_res = slow_dft(&float_input);

    let mut fft_input = float_input
        .iter()
        .map(|fc| Complex {
            re: ctx.variable(fc.re),
            im: ctx.variable(fc.im),
        })
        .collect::<Vec<_>>();

    fft_1d(&mut ctx, &mut fft_input);

    for i in 0..float_input.len() {
        let val_re = ctx.get_val(fft_input[i].re);
        let val_im = ctx.get_val(fft_input[i].im);
        assert!(
            (val_re - dft_res[i].re).abs() < 1e-3,
            "Mismatch at index {} re",
            i
        );
        assert!(
            (val_im - dft_res[i].im).abs() < 1e-3,
            "Mismatch at index {} im",
            i
        );
    }
}

#[test]
fn test_fft_invertibility_1d() {
    let mut ctx = Context::default();
    let original_floats = vec![1.0, 2.0, -3.5, 0.0, 0.0, -1.25, 4.0, 3.0];

    let mut data = original_floats
        .chunks(2)
        .map(|chunk| Complex {
            re: ctx.variable(chunk[0]),
            im: ctx.variable(chunk[1]),
        })
        .collect::<Vec<_>>();

    let original = data.clone();

    fft_1d(&mut ctx, &mut data);
    ifft_1d(&mut ctx, &mut data);

    for i in 0..data.len() {
        let original_re = ctx.get_val(original[i].re);
        let original_im = ctx.get_val(original[i].im);
        let reconstructed_re = ctx.get_val(data[i].re);
        let reconstructed_im = ctx.get_val(data[i].im);

        assert!(
            (reconstructed_re - original_re).abs() < 1e-3,
            "Inversion re failed at {}",
            i
        );
        assert!(
            (reconstructed_im - original_im).abs() < 1e-3,
            "Inversion im failed at {}",
            i
        );
    }
}

#[test]
fn test_fft_invertibility_2d() {
    let mut ctx = Context::default();
    let width = 4;
    let height = 4;

    let mut data = vec![
        Complex {
            re: ctx.variable(0.0),
            im: ctx.variable(0.0)
        };
        width * height
    ];
    for r in 0..height {
        for c in 0..width {
            let x = c as f32 / width as f32;
            let y = r as f32 / height as f32;
            let re_val = (2.0 * std::f32::consts::PI * (x + 2.0 * y)).sin();
            data[r * width + c] = Complex {
                re: ctx.variable(re_val),
                im: ctx.variable(0.0),
            };
        }
    }

    let original = data.clone();

    fft_2d(&mut ctx, &mut data, width, height);
    ifft_2d(&mut ctx, &mut data, width, height);

    for i in 0..data.len() {
        let original_re = ctx.get_val(original[i].re);
        let original_im = ctx.get_val(original[i].im);
        let reconstructed_re = ctx.get_val(data[i].re);
        let reconstructed_im = ctx.get_val(data[i].im);

        assert!(
            (reconstructed_re - original_re).abs() < 1e-3,
            "2D inversion re failed at {}",
            i
        );
        assert!(
            (reconstructed_im - original_im).abs() < 1e-3,
            "2D inversion im failed at {}",
            i
        );
    }
}

// --- Backpropagation Graph Test ---

#[test]
fn test_linear_regression_training() {
    let mut ctx = Context::default();

    let x = ctx.variable(2.0);
    let target = ctx.variable(5.0);

    let w = ctx.variable(0.5);
    let b = ctx.variable(0.0);

    let lr = 0.02;

    let mut last_loss = f32::MAX;
    for _ in 0..10 {
        let wx = ctx.mul(w, x);
        let pred = ctx.add(wx, b);

        let diff = ctx.sub(pred, target);
        let loss = ctx.mul(diff, diff);

        let loss_val = ctx.get_val(loss);
        assert!(loss_val <= last_loss);
        last_loss = loss_val;

        ctx.zero_grad();
        ctx.backward(loss);

        let w_val = ctx.get_val(w);
        let w_grad = ctx.get_grad(w);
        ctx.update_val(w, w_val - lr * w_grad);

        let b_val = ctx.get_val(b);
        let b_grad = ctx.get_grad(b);
        ctx.update_val(b, b_val - lr * b_grad);
    }
}

#[test]
fn test_numerical_gradient_checking() {
    let epsilon = 1e-4;

    let eval_loss = |w_val: f32| -> f32 {
        let mut ctx = Context::default();
        let x = ctx.variable(1.5);
        let target = ctx.variable(3.0);
        let w = ctx.variable(w_val);
        let wx = ctx.mul(w, x);
        let diff = ctx.sub(wx, target);
        let loss = ctx.mul(diff, diff);
        ctx.get_val(loss)
    };

    let w_init = 0.8;
    let mut ctx = Context::default();
    let x = ctx.variable(1.5);
    let target = ctx.variable(3.0);
    let w = ctx.variable(w_init);
    let wx = ctx.mul(w, x);
    let diff = ctx.sub(wx, target);
    let loss = ctx.mul(diff, diff);

    ctx.backward(loss);
    let analytical_grad = ctx.get_grad(w);

    let loss_plus = eval_loss(w_init + epsilon);
    let loss_minus = eval_loss(w_init - epsilon);
    let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

    assert!((analytical_grad - numerical_grad).abs() < 5e-3);
}
