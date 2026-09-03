use crate::fno::{
    FNO,
    backprop::graph::{Context, Variable},
    data::TrajectoryDataset,
};

// who is adam and why is he optimizing my model by hand
pub struct Adam {
    lr: f32, // \alpha
    b1: f32, // momentum decay
    b2: f32,
    step: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(num_params: usize, lr: f32) -> Self {
        Adam {
            lr,
            b1: 0.9,
            b2: 0.999,
            step: 0,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;

        let b1_t = self.b1.powi(self.step as i32);
        let b2_t = self.b2.powi(self.step as i32);

        for (i, param) in params.iter_mut().enumerate() {
            let g = grads[i];
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g;
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g;

            let m = self.m[i] / (1.0 - b1_t);
            let v = self.v[i] / (1.0 - b2_t);

            *param -= self.lr * m / (v.sqrt() + 1e-8);
        }
    }
}

pub struct Hyperparameters {
    pub epochs: usize,
    pub lr: f32,
    pub seed: u64,
}

pub fn train_fno(
    dataset: &TrajectoryDataset,
    rollout_steps: usize,
    Hyperparameters { epochs, lr, seed }: Hyperparameters,
    channels: (usize, usize, usize),
    modes: (usize, usize),
    num_layers: usize,
) -> Vec<f32> {
    let dims = dataset.dims;
    let n = dims.0 * dims.1;

    let mut ctx = Context::default();
    let fno_init = FNO::new(&mut ctx, channels, modes, num_layers, seed);
    let var_pointers = fno_init.collect_weights();
    let mut weights: Vec<f32> = var_pointers.iter().map(|&v| ctx.get_val(v)).collect();

    let mut optimizer = Adam::new(weights.len(), lr);

    for epoch in 0..epochs {
        let mut total_epoch_loss = 0.0;

        for trajectory in &dataset.trajectories {
            let mut ctx = Context::default();

            let fno = FNO::from_weights(&mut ctx, channels, modes, num_layers, &weights);
            let vars = fno.collect_weights();

            let mut current_state: Vec<Variable> = trajectory.frames[0]
                .iter()
                .map(|&val| ctx.variable(val))
                .collect();

            let mut total_loss = ctx.variable(0.0);
            for step in 1..=rollout_steps {
                let pred = fno.forward(&mut ctx, dims, &current_state);

                for i in 0..n {
                    let target = ctx.variable(trajectory.frames[step][i]);
                    let diff = ctx.sub(pred[i], target);
                    let sq = ctx.mul(diff, diff);
                    total_loss = ctx.add(total_loss, sq);
                }
                current_state = pred; 
            }

            let scale = ctx.variable(1.0 / (rollout_steps * n) as f32);
            let mean_loss = ctx.mul(total_loss, scale);
            total_epoch_loss += ctx.get_val(mean_loss);

            ctx.backward(mean_loss);

            let grads: Vec<f32> = vars.iter().map(|&v| ctx.get_grad(v)).collect();
            optimizer.step(&mut weights, &grads);
        }

        println!(
            "Epoch {} / {}: Loss = {:.6}",
            epoch + 1,
            epochs,
            total_epoch_loss / dataset.trajectories.len() as f32
        );
    }

    weights
}
