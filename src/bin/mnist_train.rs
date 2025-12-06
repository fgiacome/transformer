use clap::Parser;
use faer::prelude::*;
use std::path::PathBuf;
use transformer::{
    models::Model, Feedforward, SigmoidActivation, SoftmaxActivation, ModelType,
    CrossEntropyLoss, Loss, Optimizer, SGD, io,
};

#[derive(Parser, Debug)]
#[command(name = "mnist-train")]
#[command(about = "Train an MLP on the MNIST dataset", long_about = None)]
struct Args {
    /// Path to MNIST dataset directory
    #[arg(short, long, default_value = "datasets/mnist")]
    dataset_path: PathBuf,

    /// Number of epochs to train
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,

    /// Learning rate
    #[arg(short, long, default_value_t = 0.1)]
    learning_rate: f32,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Hidden layer size
    #[arg(long, default_value_t = 128)]
    hidden_size: usize,

    /// Path to save the trained model
    #[arg(short, long, default_value = "mnist_model.bin")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    println!("Loading MNIST dataset from {:?}...", args.dataset_path);

    // Load MNIST dataset
    let mnist::Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = mnist::MnistBuilder::new()
        .base_path(&args.dataset_path.to_string_lossy())
        .label_format_one_hot()
        .finalize();

    let num_train = trn_img.len() / 784;
    let num_test = tst_img.len() / 784;

    println!("Training samples: {}", num_train);
    println!("Test samples: {}", num_test);

    // Build network: 784 -> hidden_size -> 10
    println!(
        "Building MLP: 784 -> {} -> 10",
        args.hidden_size
    );

    let mut network: Vec<ModelType> = vec![
        ModelType::Feedforward(Feedforward::new(784, args.hidden_size)),
        ModelType::Sigmoid(SigmoidActivation::new()),
        ModelType::Feedforward(Feedforward::new(args.hidden_size, 10)),
        ModelType::Softmax(SoftmaxActivation::new()),
    ];

    let mut optimizer = SGD::new(args.learning_rate);
    let loss_fn = CrossEntropyLoss::new();

    println!(
        "Training for {} epochs with batch size {} and learning rate {}",
        args.epochs, args.batch_size, args.learning_rate
    );

    // Training loop
    for epoch in 0..args.epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // Iterate over batches
        for batch_start in (0..num_train).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(num_train);
            let actual_batch_size = batch_end - batch_start;

            // Prepare batch data: (784, batch_size)
            let input = Mat::from_fn(784, actual_batch_size, |i, j| {
                let sample_idx = batch_start + j;
                trn_img[sample_idx * 784 + i] as f32 / 255.0 // Normalize to [0, 1]
            });

            // Prepare batch labels: (10, batch_size) one-hot encoded
            let target = Mat::from_fn(10, actual_batch_size, |i, j| {
                let sample_idx = batch_start + j;
                trn_lbl[sample_idx * 10 + i] as f32
            });

            // Forward pass
            let output = network.forward(&input);

            // Compute loss
            let loss = loss_fn.compute(&output, &target);
            total_loss += loss;
            num_batches += 1;

            // Backward pass
            let loss_grad = loss_fn.gradient(&output, &target);
            network.zero_grad();
            network.gradient(&loss_grad);

            // Update weights
            optimizer.step(&mut network);
        }

        let avg_loss = total_loss / num_batches as f32;

        // Evaluate on test set
        let test_accuracy = evaluate(&mut network, &tst_img, &tst_lbl, num_test);

        println!(
            "Epoch {}/{}: Loss = {:.4}, Test Accuracy = {:.2}%",
            epoch + 1,
            args.epochs,
            avg_loss,
            test_accuracy * 100.0
        );
    }

    let final_accuracy = evaluate(&mut network, &tst_img, &tst_lbl, num_test);
    println!("Training complete! Final test accuracy: {:.2}%", final_accuracy * 100.0);

    // Save the trained model
    println!("Saving model to {:?}...", args.output);
    io::save(&network, &args.output).expect("Failed to save model");
    println!("Model saved successfully!");
}

fn evaluate(
    network: &mut Vec<ModelType>,
    test_img: &[u8],
    test_lbl: &[u8],
    num_test: usize,
) -> f32 {
    let mut correct = 0;

    // Evaluate in batches for efficiency
    let batch_size = 100;

    for batch_start in (0..num_test).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_test);
        let actual_batch_size = batch_end - batch_start;

        // Prepare batch
        let input = Mat::from_fn(784, actual_batch_size, |i, j| {
            let sample_idx = batch_start + j;
            test_img[sample_idx * 784 + i] as f32 / 255.0
        });

        // Forward pass
        let output = network.forward(&input);

        // Check predictions
        for j in 0..actual_batch_size {
            // Find predicted class (argmax)
            let mut max_idx = 0;
            let mut max_val = output[(0, j)];
            for i in 1..10 {
                if output[(i, j)] > max_val {
                    max_val = output[(i, j)];
                    max_idx = i;
                }
            }

            // Find true class
            let sample_idx = batch_start + j;
            let mut true_class = 0;
            for i in 0..10 {
                if test_lbl[sample_idx * 10 + i] == 1 {
                    true_class = i;
                    break;
                }
            }

            if max_idx == true_class {
                correct += 1;
            }
        }
    }

    correct as f32 / num_test as f32
}
