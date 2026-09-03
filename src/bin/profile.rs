use core::f32;
use std::hint::black_box;
use fluidsim::fno::training::{Hyperparameters, train_fno};
use rand::{SeedableRng, rngs::StdRng};

use fluidsim::fno::FNO;
use fluidsim::fno::data::generate_dataset;
use fluidsim::fno::initialization::ICConfig;
use fluidsim::solver::Config;

fn main() {
    let cfg = Config {
        x_bound: 2.0 * f32::consts::PI,
        y_bound: 2.0 * f32::consts::PI,
        x_res: 32,
        y_res: 32,
        viscosity: 0.05,
    };

    let ic_cfg = ICConfig {
        alpha: 2.5,
        tau: 3.0,
        target_std: 1.0,
    };

    let mut rng = StdRng::seed_from_u64(2);

    let dataset = black_box(generate_dataset(10, 30, 0.1, cfg, ic_cfg, &mut rng));

    train_fno(&dataset, 10, Hyperparameters{epochs:5, lr:0.005, seed:42}, (1,4,1), (4,4), 2);

}
