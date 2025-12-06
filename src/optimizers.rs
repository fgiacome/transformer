use faer::prelude::*;
use crate::models::Model;

pub trait Optimizer {
    /// Performs a single optimization step, updating model parameters based on their gradients
    fn step(&mut self, model: &mut dyn Model);
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    /// Creates a new SGD optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate for gradient descent
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, model: &mut dyn Model) {
        for param in model.parameters_mut() {
            param.value -= &param.grad * Scale(self.learning_rate);
        }
    }
}
