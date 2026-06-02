mod solver;
use solver::{Config, Grid, InitialConditions};

fn main() {
    // let cfg = Config {
    //     height: 1.0,
    //     width: 1.0,
    //     x_resolution: 100,
    //     y_resolution: 100,
    //     viscosity: 0.00001,
    // };
    // let initial_conditions_fn = |x: f64, y: f64| ((f64::sin(x), f64::sin(y)), 0.0);
    //
    // let dy = cfg.height / cfg.y_resolution as f64;
    // let dx = cfg.width / cfg.x_resolution as f64;
    // let mut velocities = vec![0.0; cfg.x_resolution * cfg.y_resolution * 2];
    // for i in 0..cfg.y_resolution {
    //     for j in 0..cfg.x_resolution {
    //         let x = j as f64 * dx;
    //         let y = i as f64 * dy;
    //         let (v, _density) = initial_conditions_fn(x, y);
    //         let idx = (i * cfg.x_resolution + j) * 2;
    //         velocities[idx] = v.0;
    //         velocities[idx + 1] = v.1;
    //     }
    // }
    //
    // let mut grid = Grid::new(cfg, InitialConditions(velocities));
    // for _ in 0..100 {
    //     grid = grid.step_euler(0.001);
    // }
}
