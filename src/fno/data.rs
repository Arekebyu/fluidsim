use crate::fno::initialization::{self, ICConfig, generate_initial_conditions};
use crate::{Config, Grid};
use rand::rngs::StdRng;

#[derive(Clone, Debug)]
pub struct Trajectory {
    pub frames: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct TrajectoryDataset {
    pub trajectories: Vec<Trajectory>,
    pub dims: (usize, usize),
    pub dt: f32,
}

pub fn generate_dataset(
    num_trajectories: usize,
    num_steps: usize,
    dt: f32,
    cfg: Config,
    ICConfig {
        alpha,
        tau,
        target_std,
    }: ICConfig,
    rng: &mut StdRng,
) -> TrajectoryDataset {
    let mut trajectories = Vec::with_capacity(num_trajectories);
    let width = cfg.x_res;
    let height = cfg.y_res;

    for _ in 0..num_trajectories {
        let ic = generate_initial_conditions(
            (width, height),
            initialization::ICConfig {
                alpha,
                tau,
                target_std,
            },
            rng,
        );
        let mut solver = Grid::new(&cfg, ic);

        let mut frames = Vec::with_capacity(num_steps + 1);
        frames.push(solver.vorticity.clone()); // Frame 0: w_0

        for _ in 0..num_steps {
            solver.step(dt);
            frames.push(solver.vorticity.clone());
        }

        trajectories.push(Trajectory { frames });
    }

    TrajectoryDataset {
        trajectories,
        dims: (width, height),
        dt,
    }
}
