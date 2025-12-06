use clap::Parser;
use faer::prelude::*;
use std::path::PathBuf;
use transformer::{models::Model, ModelType, io};
use rand::Rng;

#[derive(Parser, Debug)]
#[command(name = "mnist-test")]
#[command(about = "Test a trained MNIST model on test images", long_about = None)]
struct Args {
    /// Path to the trained model file
    #[arg(short, long, default_value = "mnist_model.bin")]
    model_path: PathBuf,

    /// Path to MNIST dataset directory
    #[arg(short, long, default_value = "datasets/mnist")]
    dataset_path: PathBuf,

    /// Specific test image index to use (if not provided, uses random)
    #[arg(short, long)]
    image_index: Option<usize>,
}

fn main() {
    let args = Args::parse();

    println!("Loading MNIST test dataset from {:?}...", args.dataset_path);

    // Load MNIST test dataset
    let mnist::Mnist {
        tst_img,
        tst_lbl,
        ..
    } = mnist::MnistBuilder::new()
        .base_path(&args.dataset_path.to_string_lossy())
        .label_format_one_hot()
        .finalize();

    let num_test = tst_img.len() / 784;
    println!("Test samples available: {}", num_test);

    // Select image index
    let image_idx = match args.image_index {
        Some(idx) => {
            if idx >= num_test {
                eprintln!("Error: Image index {} is out of range (max: {})", idx, num_test - 1);
                std::process::exit(1);
            }
            println!("Using specified image index: {}", idx);
            idx
        }
        None => {
            let idx = rand::rng().random_range(0..num_test);
            println!("Using random image index: {}", idx);
            idx
        }
    };

    // Load the trained model
    println!("Loading model from {:?}...", args.model_path);
    let mut network: Vec<ModelType> = io::load(&args.model_path)
        .expect("Failed to load model. Make sure you've trained a model first!");

    // Set to inference mode
    network.set_inference(true);

    // Get the test image and label
    let image_start = image_idx * 784;
    let label_start = image_idx * 10;

    // Find true label
    let mut true_label = 0;
    for i in 0..10 {
        if tst_lbl[label_start + i] == 1 {
            true_label = i;
            break;
        }
    }

    // Prepare input (784, 1) - single image
    let input = Mat::from_fn(784, 1, |i, _| {
        tst_img[image_start + i] as f32 / 255.0
    });

    // Run inference
    let output = network.forward(&input);

    // Find predicted class and get all probabilities
    let mut predictions: Vec<(usize, f32)> = (0..10)
        .map(|i| (i, output[(i, 0)]))
        .collect();

    // Sort by probability (descending)
    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let predicted_label = predictions[0].0;
    let confidence = predictions[0].1 * 100.0;

    // Display the image
    println!("\n{}", "=".repeat(56));
    println!("                    MNIST TEST IMAGE");
    println!("{}", "=".repeat(56));
    print_image(&tst_img[image_start..image_start + 784]);
    println!("{}", "=".repeat(56));

    // Display results
    println!("\nRESULTS:");
    println!("  True Label:      {}", true_label);
    println!("  Predicted Label: {}", predicted_label);
    println!("  Confidence:      {:02.2}%", confidence);
    println!("  Status:          {}",
        if predicted_label == true_label {
            "CORRECT"
        } else {
            "INCORRECT"
        });

    // Display top 3 predictions
    println!("\nTop 3 Predictions:");
    for (i, (label, prob)) in predictions.iter().take(3).enumerate() {
        println!("  {}. Digit {}: {:>5.2}%", i + 1, label, prob * 100.0);
    }
    println!();
}

/// Prints a 28x28 MNIST image to the terminal using ASCII characters
fn print_image(pixels: &[u8]) {
    // ASCII characters from darkest to lightest
    let ascii_chars = " .:;+=xX$&#@";
    let scale = ascii_chars.len() - 1;

    println!();
    for row in 0..28 {
        print!("  ");
        for col in 0..28 {
            let pixel_value = pixels[row * 28 + col];
            let char_idx = ((pixel_value as usize) * scale) / 255;
            let ch = ascii_chars.chars().nth(char_idx).unwrap();
            // Print each character twice for better aspect ratio
            print!("{}{}", ch, ch);
        }
        println!();
    }
    println!();
}
