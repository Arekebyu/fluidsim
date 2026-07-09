use crate::fno::backprop::graph::Context;

#[test]
fn test_fno_forward_and_training() {
    let mut ctx = Context::new();
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
        seed,
        width,
        height,
        in_channels,
        out_channels,
        layer_channels,
        modes_x,
        modes_y,
        num_layers,
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
    let pred = fno.forward(&mut ctx, &input);
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

    // Verify gradients propagate to all weights
    let lift_grad = ctx.get_grad(fno.lifting_w[0][0]);
    assert!(lift_grad != 0.0, "Lifting gradients should be non-zero");

    let w_grad = ctx.get_grad(fno.fourier_layers[0].w_weight[0][0]);
    assert!(
        w_grad != 0.0,
        "Fourier spatial gradients should be non-zero"
    );

    let r_grad = ctx.get_grad(fno.fourier_layers[0].r_weight_re[0][0][0][0]);
    assert!(
        r_grad != 0.0,
        "Fourier spectral gradients should be non-zero"
    );
}
