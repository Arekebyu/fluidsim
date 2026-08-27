use crate::fno::{FNO, backprop::graph::Context, data};

pub struct Hyperparameters {}

pub fn train(
    ctx: &mut Context,
    model: FNO,
    data: data::TrajectoryDataset,
    params: Hyperparameters,
) {
}
