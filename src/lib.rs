use futures::prelude::*;
use std::convert::Infallible;
use wasm_bindgen::prelude::*;

use crate::messages::{Message, Response};
use crate::solver::{Config, Grid, InitialConditions};

pub mod calculations;
pub mod messages;
pub mod solver;

// pub async fn run(
//     mut incoming: impl Stream<Item = Message> + Unpin,
//     mut outgoing: impl Sink<Response, Error = Infallible> + Unpin,
// ) {
//     outgoing.send(Response::Ready).await.unwrap();
//     let mut grid: Option<Grid> = None;
//     while let Some(msg) = incoming.next().await {
//         match msg {
//             Message::Config(cfg, ic) => {
//                 grid = Some(Grid::new(cfg, ic));
//             }
//             Message::Step(dt) => {
//                 if let Some(g) = grid.as_mut() {
//                     *g = g.step_euler(dt);
//                     outgoing.send(Response::Grid(g.clone())).await.unwrap();
//                 } else {
//                     outgoing
//                         .send(Response::Error("No grid set".to_string()))
//                         .await
//                         .unwrap();
//                 }
//             }
//         }
//     }
// }
//
// #[wasm_bindgen]
// pub struct WasmSolver {
//     grid: Option<Grid>,
// }
//
// #[wasm_bindgen]
// impl WasmSolver {
//     #[wasm_bindgen(constructor)]
//     pub fn new() -> Self {
//         WasmSolver { grid: None }
//     }
//
//     pub fn init(
//         &mut self,
//         width: f64,
//         height: f64,
//         x_resolution: usize,
//         y_resolution: usize,
//         viscosity: f64,
//         initial_velocities: Vec<f64>,
//     ) {
//         let cfg = Config {
//             width,
//             height,
//             x_resolution,
//             y_resolution,
//             viscosity,
//         };
//         self.grid = Some(Grid::new(cfg, InitialConditions(initial_velocities)));
//     }
//
//     pub fn step(&mut self, dt: f64) {
//         if let Some(g) = self.grid.as_mut() {
//             *g = g.step_euler(dt);
//         }
//     }
//
//     pub fn get_velocities_ptr(&self) -> *const f64 {
//         match &self.grid {
//             Some(g) => g.velocities.as_ptr(),
//             None => std::ptr::null(),
//         }
//     }
//
//     pub fn get_velocities_len(&self) -> usize {
//         match &self.grid {
//             Some(g) => g.velocities.len(),
//             None => 0,
//         }
//     }
// }
