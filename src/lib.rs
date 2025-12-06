use faer::prelude::*;
use serde::{Deserialize, Serialize};

pub mod io;
pub mod losses;
pub mod models;
pub mod optimizers;

// Re-export commonly used items
pub use losses::{CrossEntropyLoss, Loss};
pub use models::{
    Feedforward, Model, ModelType, RELUActivation, SigmoidActivation, SoftmaxActivation,
};
pub use optimizers::{Optimizer, SGD};

#[derive(Serialize, Deserialize)]
pub struct Parameter {
    pub value: Mat<f32>,
    pub grad: Mat<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use losses::Loss;
    use models::Model;

    #[test]
    fn test_feedforward_single_input() {
        let mut layer = Feedforward::new(3, 2);

        // Single input: (3, 1) - 3 features, 1 sample
        let input = Mat::from_fn(3, 1, |i, _| i as f32);

        let output = layer.forward(&input);

        // Output should be (2, 1) - 2 outputs, 1 sample
        assert_eq!(output.nrows(), 2);
        assert_eq!(output.ncols(), 1);
    }

    #[test]
    fn test_feedforward_batched_input() {
        let mut layer = Feedforward::new(3, 2);

        // Batched input: (3, 4) - 3 features, 4 samples
        let input = Mat::from_fn(3, 4, |i, j| (i + j) as f32);

        let output = layer.forward(&input);

        // Output should be (2, 4) - 2 outputs, 4 samples
        assert_eq!(output.nrows(), 2);
        assert_eq!(output.ncols(), 4);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss::new();

        // Create predictions: (3 classes, 2 samples)
        // Sample 1: [0.7, 0.2, 0.1] - predicts class 0
        // Sample 2: [0.1, 0.1, 0.8] - predicts class 2
        let predicted = Mat::from_fn(3, 2, |i, j| {
            if j == 0 {
                match i {
                    0 => 0.7,
                    1 => 0.2,
                    2 => 0.1,
                    _ => 0.0,
                }
            } else {
                match i {
                    0 => 0.1,
                    1 => 0.1,
                    2 => 0.8,
                    _ => 0.0,
                }
            }
        });

        // One-hot targets: class 0 for sample 1, class 2 for sample 2
        let target = Mat::from_fn(3, 2, |i, j| {
            if (j == 0 && i == 0) || (j == 1 && i == 2) {
                1.0
            } else {
                0.0
            }
        });

        // Compute loss
        let loss = loss_fn.compute(&predicted, &target);

        // Loss should be positive and reasonable
        assert!(loss > 0.0);
        assert!(loss < 1.0); // Good predictions should have low loss

        // Compute gradient
        let grad = loss_fn.gradient(&predicted, &target);

        // Gradient shape should match prediction shape
        assert_eq!(grad.nrows(), predicted.nrows());
        assert_eq!(grad.ncols(), predicted.ncols());
    }

    #[test]
    fn test_softmax_probabilities_sum_to_one() {
        let mut softmax = SoftmaxActivation::new();

        // Input: (3 classes, 2 samples)
        let input = Mat::from_fn(3, 2, |i, j| (i as f32 + j as f32) * 0.5);

        let output = softmax.forward(&input);

        // Check output shape
        assert_eq!(output.nrows(), 3);
        assert_eq!(output.ncols(), 2);

        // Check that probabilities sum to 1 for each sample (column)
        for j in 0..output.ncols() {
            let mut sum = 0.0;
            for i in 0..output.nrows() {
                sum += output[(i, j)];
            }
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities should sum to 1, got {}",
                sum
            );
        }

        // Check all values are between 0 and 1
        for j in 0..output.ncols() {
            for i in 0..output.nrows() {
                let val = output[(i, j)];
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Probability should be in [0,1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_softmax_gradient_shape() {
        let mut softmax = SoftmaxActivation::new();

        // Input: (4 classes, 3 samples)
        let input = Mat::from_fn(4, 3, |i, j| (i as f32 - j as f32) * 0.3);

        // Forward pass
        let _output = softmax.forward(&input);

        // Gradient from loss
        let loss_grad = Mat::from_fn(4, 3, |i, j| (i + j) as f32 * 0.1);

        // Backward pass
        let input_grad = softmax.gradient(&loss_grad);

        // Check gradient shape matches input shape
        assert_eq!(input_grad.nrows(), input.nrows());
        assert_eq!(input_grad.ncols(), input.ncols());
    }

    #[test]
    fn test_model_save_and_load() {
        use std::fs;

        // Create a feedforward layer
        let mut layer = Feedforward::new(10, 5);

        // Get parameters for comparison
        let original_params: Vec<_> = layer
            .parameters_mut()
            .iter()
            .map(|p| p.value.clone())
            .collect();

        // Save to file
        let path = "test_model.bin";
        io::save(&layer, path).unwrap();

        // Load from file
        let mut loaded_layer: Feedforward = io::load(path).unwrap();

        // Get loaded parameters
        let loaded_params: Vec<_> = loaded_layer
            .parameters_mut()
            .iter()
            .map(|p| p.value.clone())
            .collect();

        // Verify we have the same number of parameters
        assert_eq!(loaded_params.len(), original_params.len());

        // Verify each parameter matches
        for (loaded, original) in loaded_params.iter().zip(original_params.iter()) {
            assert_eq!(loaded.nrows(), original.nrows());
            assert_eq!(loaded.ncols(), original.ncols());

            for i in 0..original.nrows() {
                for j in 0..original.ncols() {
                    assert_eq!(
                        loaded[(i, j)],
                        original[(i, j)],
                        "Parameter mismatch at ({}, {})",
                        i,
                        j
                    );
                }
            }
        }

        // Clean up
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_network_serialization() {
        use std::fs;

        // Create a small network using Vec<ModelType>
        let mut network: Vec<ModelType> = vec![
            ModelType::Feedforward(Feedforward::new(10, 5)),
            ModelType::Sigmoid(SigmoidActivation::new()),
            ModelType::Feedforward(Feedforward::new(5, 3)),
        ];

        // Create test input
        let input = Mat::from_fn(10, 2, |i, j| (i + j) as f32 * 0.1);
        let original_output = network.forward(&input);

        // Save network
        let path = "test_network.bin";
        io::save(&network, path).unwrap();

        // Load network
        let mut loaded_network: Vec<ModelType> = io::load(path).unwrap();

        // Test that loaded network produces same output
        let loaded_output = loaded_network.forward(&input);

        assert_eq!(loaded_output.nrows(), original_output.nrows());
        assert_eq!(loaded_output.ncols(), original_output.ncols());

        for i in 0..original_output.nrows() {
            for j in 0..original_output.ncols() {
                assert_eq!(
                    loaded_output[(i, j)],
                    original_output[(i, j)],
                    "Output mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Clean up
        fs::remove_file(path).unwrap();
    }
}
