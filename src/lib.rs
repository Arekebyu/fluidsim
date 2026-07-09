pub mod calculations;
pub mod fno;
pub mod messages;
pub mod solver;
use crate::messages::{Message, Response};
use crate::solver::Config;
use crate::solver::Grid;
use futures::prelude::*;
use std::convert::Infallible;
use wasm_bindgen::prelude::*;

pub async fn run(
    mut incoming: impl Stream<Item = Message> + Unpin,
    mut outgoing: impl Sink<Response, Error = Infallible> + Unpin,
) {
    outgoing.send(Response::Ready).await.unwrap();
    let mut grid: Option<Grid> = None;
    while let Some(msg) = incoming.next().await {
        match msg {
            Message::Config(cfg, ic) => {
                grid = Some(Grid::new(cfg, ic));
            }
            Message::Step(dt) => {}
        }
    }
}
