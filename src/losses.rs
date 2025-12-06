use faer::prelude::*;

pub trait Loss {
    /// Computes the loss value given predictions and targets
    ///
    /// # Arguments
    /// * `predicted` - Model predictions (e.g., after softmax)
    /// * `target` - Ground truth labels (e.g., one-hot encoded)
    ///
    /// # Returns
    /// Scalar loss value (averaged over batch)
    fn compute(&self, predicted: &Mat<f32>, target: &Mat<f32>) -> f32;

    /// Computes the gradient of the loss with respect to predictions
    ///
    /// # Arguments
    /// * `predicted` - Model predictions
    /// * `target` - Ground truth labels
    ///
    /// # Returns
    /// Gradient matrix with same shape as predicted
    fn gradient(&self, predicted: &Mat<f32>, target: &Mat<f32>) -> Mat<f32>;
}

/// Categorical Cross-Entropy Loss
///
/// Used for multi-class classification problems.
/// Expects predictions to be probabilities (e.g., after softmax)
/// and targets to be one-hot encoded.
///
/// Matrix format: Each column is a sample, each row is a class
/// - predictions: (num_classes, batch_size)
/// - targets: (num_classes, batch_size) one-hot encoded
///
/// Loss formula: -sum(target * log(predicted)) / batch_size
/// Gradient (when used with softmax): (predicted - target) / batch_size
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for CrossEntropyLoss {
    fn compute(&self, predicted: &Mat<f32>, target: &Mat<f32>) -> f32 {
        let batch_size = predicted.ncols() as f32;
        let mut total_loss = 0.0;

        // Loss = -sum(target * log(predicted)) / batch_size
        for j in 0..predicted.ncols() {
            for i in 0..predicted.nrows() {
                let t = target[(i, j)];
                let p = predicted[(i, j)];

                // Clip predictions to avoid log(0)
                let p_clipped = p.max(1e-7).min(1.0 - 1e-7);

                total_loss -= t * p_clipped.ln();
            }
        }

        total_loss / batch_size
    }

    fn gradient(&self, predicted: &Mat<f32>, target: &Mat<f32>) -> Mat<f32> {
        let batch_size = predicted.ncols() as f32;

        // When used with softmax output: gradient = (predicted - target) / batch_size
        Mat::from_fn(predicted.nrows(), predicted.ncols(), |i, j| {
            (predicted[(i, j)] - target[(i, j)]) / batch_size
        })
    }
}
