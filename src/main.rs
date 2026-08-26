use fluidsim::solver::{Config, Grid, InitialConditions};
use std::env;
use std::io::Write;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    let width = args[1].parse().unwrap_or(64);
    let height = args[2].parse().unwrap_or(64);
    let viscosity = args[3].parse().unwrap_or(0.01);
    let dt = args[4].parse().unwrap_or(0.01);
    let steps = args[5].parse().unwrap_or(100);

    print!("{} {} {} {} {}", width, height, viscosity, dt, steps);

    let cfg = Config {
        x_bound: 2.0 * std::f32::consts::PI,
        y_bound: 2.0 * std::f32::consts::PI,
        x_res: width,
        y_res: height,
        viscosity,
    };

    let dx = 2.0 * std::f32::consts::PI / width as f32;
    let dy = 2.0 * std::f32::consts::PI / height as f32;
    let mut initial_vorticity = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let px = x as f32 * dx;
            let py = y as f32 * dy;
            initial_vorticity[y * width + x] = -2.0 * px.sin() * py.sin();
        }
    }

    let initial_conditions = InitialConditions {
        vorticity: initial_vorticity.clone(),
        vx: vec![0.0; width * height],
        vy: vec![0.0; width * height],
    };

    let mut solver = Grid::new(&cfg, initial_conditions);

    save("initial_vorticity.bin", &initial_vorticity)?;

    println!(
        "Running Rust solver for {} steps on a {}x{} grid...",
        steps, width, height
    );

    let start = Instant::now();
    for _ in 0..steps {
        solver.step(dt);
    }
    let duration = start.elapsed();
    println!(
        "Rust execution time: {:.2} ms",
        duration.as_secs_f64() * 1000.0
    );

    save("rust_vorticity.bin", &solver.vorticity)?;

    Ok(())
}

/// Helper function to save a float slice to a binary file (little-endian)
fn save(path: &str, data: &[f32]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}
