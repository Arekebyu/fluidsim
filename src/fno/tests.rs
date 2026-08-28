use crate::fno::backprop::graph::Context;

#[test]
fn test_fno_forward_and_training() {
    let mut ctx = Context::default();
    let seed = 42u64;

    let width = 4;
    let height = 4;
    let in_channels = 1;
    let out_channels = 1;
    let layer_channels = 2;
    let modes_x = 2;
    let modes_y = 2;
    let num_layers = 1;

    let fno = super::FNO::new(
        &mut ctx,
        (in_channels, layer_channels, out_channels),
        (modes_x, modes_y),
        num_layers,
        seed,
    );

    // Inputs: size 4*4*1 = 16 variables
    let input = (0..16)
        .map(|i| ctx.variable(i as f32 / 16.0))
        .collect::<Vec<_>>();

    // Target: size 4*4*1 = 16 variables
    let target = (0..16)
        .map(|i| ctx.variable((i as f32 / 16.0).sin()))
        .collect::<Vec<_>>();

    // Evolve forward
    let pred = fno.forward(&mut ctx, (width, height), &input);
    assert_eq!(pred.len(), 16);

    // Compute MSE Loss
    let mut total_loss = ctx.variable(0.0);
    for i in 0..16 {
        let diff = ctx.sub(pred[i], target[i]);
        let sq_diff = ctx.mul(diff, diff);
        total_loss = ctx.add(total_loss, sq_diff);
    }

    // Run backward
    ctx.backward(total_loss);

    // Verify gradients propagate across all parameter tensors
    let lift_has_grad = fno
        .lift_layer
        .w
        .iter()
        .any(|row| row.iter().any(|&v| ctx.get_grad(v) != 0.0));
    assert!(lift_has_grad, "Lifting gradients should be non-zero");

    let w_has_grad = fno.fourier_layers[0]
        .residual
        .w
        .iter()
        .any(|row| row.iter().any(|&v| ctx.get_grad(v) != 0.0));
    assert!(w_has_grad, "Fourier spatial gradients should be non-zero");

    let r_has_grad = fno.fourier_layers[0].r.iter().any(|c_out| {
        c_out.iter().any(|c_in| {
            c_in.iter().any(|kx| {
                kx.iter()
                    .any(|&(re, im)| ctx.get_grad(re) != 0.0 || ctx.get_grad(im) != 0.0)
            })
        })
    });
    assert!(r_has_grad, "Fourier spectral gradients should be non-zero");
}

#[test]
fn test_fno_from_weights_consistency() {
    let mut ctx1 = Context::default();
    let mut ctx2 = Context::default();
    let seed = 123u64;

    let width = 4;
    let height = 4;
    let in_channels = 1;
    let out_channels = 1;
    let layer_channels = 2;
    let modes_x = 2;
    let modes_y = 2;
    let num_layers = 2;

    let fno1 = super::FNO::new(
        &mut ctx1,
        (in_channels, layer_channels, out_channels),
        (modes_x, modes_y),
        num_layers,
        seed,
    );

    let weights: Vec<f32> = fno1
        .collect_weights()
        .iter()
        .map(|&v| ctx1.get_val(v))
        .collect();

    let fno2 = super::FNO::from_weights(
        &mut ctx2,
        (in_channels, layer_channels, out_channels),
        (modes_x, modes_y),
        num_layers,
        &weights,
    );

    let weights2: Vec<f32> = fno2
        .collect_weights()
        .iter()
        .map(|&v| ctx2.get_val(v))
        .collect();

    assert_eq!(
        weights, weights2,
        "Weights must be reconstructed identically"
    );

    let input1: Vec<_> = (0..16).map(|i| ctx1.variable(i as f32 * 0.1)).collect();
    let input2: Vec<_> = (0..16).map(|i| ctx2.variable(i as f32 * 0.1)).collect();

    let pred1 = fno1.forward(&mut ctx1, (width, height), &input1);
    let pred2 = fno2.forward(&mut ctx2, (width, height), &input2);

    for i in 0..16 {
        let val1 = ctx1.get_val(pred1[i]);
        let val2 = ctx2.get_val(pred2[i]);
        assert!(
            (val1 - val2).abs() < 1e-6,
            "Forward pass output must match between FNO::new and FNO::from_weights"
        );
    }
}
