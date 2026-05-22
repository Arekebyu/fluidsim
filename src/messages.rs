use crate::solver::{Config, Grid, InitialConditions};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum Message {
    Config(Config, InitialConditions),
    Step(f64),
}

#[derive(Serialize, Deserialize)]
pub enum Response {
    Ready,
    Error(String),
    Grid(Grid),
}
