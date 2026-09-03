use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use fluidsim::fno::data::generate_dataset;
use fluidsim::fno::initialization::ICConfig;
use fluidsim::fno::training::{train_fno, Hyperparameters};
use fluidsim::solver::Config;

fn print_usage() {
    println!(
        "Usage: cargo run --release --bin train -- [OPTIONS] or [POSITIONAL ARGS]\n\n\
        Required/Standard Parameters:\n  \
        --x <float>           Domain x bound (default: tau)\n  \
        --y <float>           Domain y bound (default: tau)\n  \
        --num-x <int>         Grid resolution in X (default: 32)\n  \
        --num-y <int>         Grid resolution in Y (default: 32)\n  \
        --viscosity <float>   Fluid kinematic viscosity (default: 0.005)\n  \
        --num-data <int>      Number of trajectories to generate (default: 10)\n  \
        --num-epochs <int>    Number of training epochs (default: 25)\n\n\
        Optional Hyperparameters:\n  \
        --dt <float>          Simulation timestep (default: 0.01)\n  \
        --steps <int>         Number of timesteps per trajectory (default: 32)\n  \
        --rollout <int>       Autoregressive rollout depth (default: 16)\n  \
        --lr <float>          Adam learning rate (default: 0.005)\n  \
        --modes-x <int>       Fourier modes in X (default: 8)\n  \
        --modes-y <int>       Fourier modes in Y (default: 8)\n  \
        --layer-channels <int>Latent representation channel width (default: 4)\n  \
        --num-layers <int>    Number of Fourier layers (default: 4)\n  \
        --seed <int>          PRNG seed (default: 42)\n  \
        --output-dir <path>   Directory to save model weights (default: models)\n\n\
        Positional Syntax:\n  \
        cargo run --release --bin train -- <x> <y> <num_x> <num_y> <viscosity> <num_data> <num_epochs>"
    );
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    // Default configuration values
    let mut x: f32 = 2.0 * std::f32::consts::PI;
    let mut y: f32 = 2.0 * std::f32::consts::PI;
    let mut num_x: usize = 32;
    let mut num_y: usize = 32;
    let mut viscosity: f32 = 0.005;
    let mut num_data: usize = 10;
    let mut num_epochs: usize = 25;

    let mut dt: f32 = 0.01;
    let mut num_steps: usize = 32;
    let mut rollout_steps: usize = 16;
    let mut lr: f32 = 0.005;
    let mut modes_x: usize = 8;
    let mut modes_y: usize = 8;
    let mut layer_channels: usize = 4;
    let mut num_layers: usize = 4;
    let mut seed: u64 = 42;
    let mut output_dir = "models".to_string();

    let mut i = 1;
    let mut positional_idx = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--x" => {
                if i + 1 < args.len() {
                    x = args[i + 1].parse().unwrap_or(x);
                    i += 1;
                }
            }
            "--y" => {
                if i + 1 < args.len() {
                    y = args[i + 1].parse().unwrap_or(y);
                    i += 1;
                }
            }
            "--num-x" | "--nx" => {
                if i + 1 < args.len() {
                    num_x = args[i + 1].parse().unwrap_or(num_x);
                    i += 1;
                }
            }
            "--num-y" | "--ny" => {
                if i + 1 < args.len() {
                    num_y = args[i + 1].parse().unwrap_or(num_y);
                    i += 1;
                }
            }
            "--viscosity" | "--nu" => {
                if i + 1 < args.len() {
                    viscosity = args[i + 1].parse().unwrap_or(viscosity);
                    i += 1;
                }
            }
            "--num-data" | "--data" => {
                if i + 1 < args.len() {
                    num_data = args[i + 1].parse().unwrap_or(num_data);
                    i += 1;
                }
            }
            "--num-epochs" | "--epochs" => {
                if i + 1 < args.len() {
                    num_epochs = args[i + 1].parse().unwrap_or(num_epochs);
                    i += 1;
                }
            }
            "--dt" => {
                if i + 1 < args.len() {
                    dt = args[i + 1].parse().unwrap_or(dt);
                    i += 1;
                }
            }
            "--steps" => {
                if i + 1 < args.len() {
                    num_steps = args[i + 1].parse().unwrap_or(num_steps);
                    i += 1;
                }
            }
            "--rollout" => {
                if i + 1 < args.len() {
                    rollout_steps = args[i + 1].parse().unwrap_or(rollout_steps);
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    lr = args[i + 1].parse().unwrap_or(lr);
                    i += 1;
                }
            }
            "--modes-x" => {
                if i + 1 < args.len() {
                    modes_x = args[i + 1].parse().unwrap_or(modes_x);
                    i += 1;
                }
            }
            "--modes-y" => {
                if i + 1 < args.len() {
                    modes_y = args[i + 1].parse().unwrap_or(modes_y);
                    i += 1;
                }
            }
            "--layer-channels" | "--width" => {
                if i + 1 < args.len() {
                    layer_channels = args[i + 1].parse().unwrap_or(layer_channels);
                    i += 1;
                }
            }
            "--num-layers" => {
                if i + 1 < args.len() {
                    num_layers = args[i + 1].parse().unwrap_or(num_layers);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse().unwrap_or(seed);
                    i += 1;
                }
            }
            "--output-dir" => {
                if i + 1 < args.len() {
                    output_dir = args[i + 1].clone();
                    i += 1;
                }
            }
            val => {
                if !val.starts_with('-') {
                    match positional_idx {
                        0 => x = val.parse().unwrap_or(x),
                        1 => y = val.parse().unwrap_or(y),
                        2 => num_x = val.parse().unwrap_or(num_x),
                        3 => num_y = val.parse().unwrap_or(num_y),
                        4 => viscosity = val.parse().unwrap_or(viscosity),
                        5 => num_data = val.parse().unwrap_or(num_data),
                        6 => num_epochs = val.parse().unwrap_or(num_epochs),
                        _ => {}
                    }
                    positional_idx += 1;
                }
            }
        }
        i += 1;
    }

    assert!(num_x.is_power_of_two(), "num_x must be a power of two");
    assert!(num_y.is_power_of_two(), "num_y must be a power of two");
    assert!(rollout_steps <= num_steps, "rollout_steps cannot exceed num_steps per trajectory");

    println!("============================================================");
    println!("               Fourier Neural Operator (FNO)                ");
    println!("============================================================");
    println!("Domain Bounds:        x = {:.4}, y = {:.4}", x, y);
    println!("Grid Resolution:      {} x {}", num_x, num_y);
    println!("Physical Viscosity:   nu = {:.5}", viscosity);
    println!("Dataset Size:         {} trajectories ({} steps each, dt={})", num_data, num_steps, dt);
    println!("Training Epochs:      {}", num_epochs);
    println!("Autoregressive Depth: {} rollout steps", rollout_steps);
    println!("Model Architecture:   Channels: (1, {}, 1), Modes: ({}, {}), Layers: {}", layer_channels, modes_x, modes_y, num_layers);
    println!("Learning Rate:        {}", lr);
    println!("============================================================\n");

    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Generate Dataset
    println!("[1/3] Generating fluid dataset via pseudo-spectral Navier-Stokes solver...");
    let start_data = Instant::now();
    let cfg = Config {
        x_bound: x,
        y_bound: y,
        x_res: num_x,
        y_res: num_y,
        viscosity,
    };
    let ic_cfg = ICConfig {
        alpha: 2.5,
        tau: 3.0,
        target_std: 1.0,
    };

    let dataset = generate_dataset(num_data, num_steps, dt, cfg, ic_cfg, &mut rng);
    let data_duration = start_data.elapsed();
    println!("Dataset generation completed in {:.2?} ({} total frame samples generated).\n", data_duration, num_data * (num_steps + 1));

    // 2. Train FNO
    println!("[2/3] Training Fourier Neural Operator on CPU Automatic Differentiation Graph...");
    let start_train = Instant::now();
    let hyperparams = Hyperparameters {
        epochs: num_epochs,
        lr,
        seed,
    };

    let trained_weights = train_fno(
        &dataset,
        rollout_steps,
        hyperparams,
        (1, layer_channels, 1),
        (modes_x, modes_y),
        num_layers,
    );
    let train_duration = start_train.elapsed();
    println!("Training completed in {:.2?}.\n", train_duration);

    // 3. Serialize and Save Weights
    println!("[3/3] Serializing trained model weights to folder '{}'...", output_dir);
    fs::create_dir_all(&output_dir)?;

    let weights_path = format!("{}/fno_weights.bin", output_dir);
    let mut file = File::create(&weights_path)?;
    for &val in &trained_weights {
        file.write_all(&val.to_le_bytes())?;
    }

    let metadata_path = format!("{}/model_metadata.json", output_dir);
    let metadata = format!(
        "{{\n  \
        \"domain\": [{:.6}, {:.6}],\n  \
        \"resolution\": [{}, {}],\n  \
        \"viscosity\": {:.6},\n  \
        \"num_data\": {},\n  \
        \"num_epochs\": {},\n  \
        \"dt\": {:.6},\n  \
        \"rollout_steps\": {},\n  \
        \"channels\": [1, {}, 1],\n  \
        \"modes\": [{}, {}],\n  \
        \"num_layers\": {},\n  \
        \"num_weights\": {}\n\
        }}\n",
        x, y, num_x, num_y, viscosity, num_data, num_epochs, dt, rollout_steps, layer_channels, modes_x, modes_y, num_layers, trained_weights.len()
    );
    fs::write(&metadata_path, metadata)?;

    println!("\x1b[92m[Success]\x1b[0m Model successfully trained and saved!");
    println!("  -> Weights binary:  {}", weights_path);
    println!("  -> Metadata config: {}", metadata_path);
    println!("  -> Total Weights:   {} parameters", trained_weights.len());

    Ok(())
}
