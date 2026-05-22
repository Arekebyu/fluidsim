use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Grid {
    pub velocities: Vec<f64>, // even terms are vx, odd terms are vy, index should be doubled.
    pub viscosity: f64,
    pub dx: f64,
    pub dy: f64,
    pub x_resolution: usize,
    pub y_resolution: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub height: f64,
    pub width: f64,
    pub x_resolution: usize,
    pub y_resolution: usize,
    pub viscosity: f64,
}

#[derive(Serialize, Deserialize)]
pub struct InitialConditions(pub Vec<f64>);

impl Grid {
    pub fn new(cfg: Config, initial_conditions: InitialConditions) -> Self {
        let dy = cfg.height / cfg.y_resolution as f64;
        let dx = cfg.width / cfg.x_resolution as f64;

        Grid {
            velocities: initial_conditions.0,
            viscosity: cfg.viscosity,
            dx,
            dy,
            x_resolution: cfg.x_resolution,
            y_resolution: cfg.y_resolution,
        }
    }

    pub fn step_euler(&self, dt: f64) -> Self {
        let mut new_velocities = self.velocities.clone();
        let x_res = self.x_resolution;
        let y_res = self.y_resolution;

        let idx = |i: usize, j: usize| -> usize { (i * x_res + j) * 2 };

        // Only update inner cells to avoid boundary out-of-bounds index overflows
        for i in 1..(y_res - 1) {
            for j in 1..(x_res - 1) {
                let cell_idx = idx(i, j);
                let vx = self.velocities[cell_idx];
                let vy = self.velocities[cell_idx + 1];

                // Right neighbor: (i, j + 1)
                let rx = self.velocities[idx(i, j + 1)];
                // Left neighbor: (i, j - 1)
                let lx = self.velocities[idx(i, j - 1)];
                let ax_laplace = (rx - 2.0 * vx + lx) / self.dx.powi(2);

                // Down neighbor: (i + 1, j)
                let dy_val = self.velocities[idx(i + 1, j) + 1];
                // Up neighbor: (i - 1, j)
                let uy_val = self.velocities[idx(i - 1, j) + 1];
                let ay_laplace = (dy_val - 2.0 * vy + uy_val) / self.dy.powi(2);

                new_velocities[cell_idx] = vx + self.viscosity * ax_laplace * dt;
                new_velocities[cell_idx + 1] = vy + self.viscosity * ay_laplace * dt;
            }
        }

        Grid {
            velocities: new_velocities,
            viscosity: self.viscosity,
            dx: self.dx,
            dy: self.dy,
            x_resolution: x_res,
            y_resolution: y_res,
        }
    }
}
