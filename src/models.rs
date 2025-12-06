use faer::prelude::*;
use faer::{zip, unzip};
use rand::rng;
use rand_distr::{Distribution, Uniform};
use serde::{Serialize, Deserialize};

use crate::Parameter;

pub trait Model {
    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32>;
    /// Caches the gradient of the error wrt model weights and return
    /// the gradient wrt inputs
    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32>;
    fn set_inference(&mut self, inference: bool);
    fn zero_grad(&mut self);
    /// Returns mutable references to all trainable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;
}

#[derive(Serialize, Deserialize)]
pub struct Feedforward {
    weights: Parameter,
    biases: Parameter,
    inference: bool,
    last_input: Option<Mat<f32>>,
}

impl Feedforward {
    /// Creates a new feedforward layer with Xavier initialization
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `output_dim` - Number of output features
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rng();

        // Xavier initialization: uniform distribution in [-limit, limit]
        // where limit = sqrt(6 / (input_dim + output_dim))
        let limit = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let dist = Uniform::new(-limit, limit).unwrap();

        let weights = Parameter {
            value: Mat::from_fn(output_dim, input_dim, |_, _| dist.sample(&mut rng)),
            grad: Mat::zeros(output_dim, input_dim),
        };

        let biases = Parameter {
            value: Mat::zeros(output_dim, 1),
            grad: Mat::zeros(output_dim, 1),
        };

        Self {
            weights,
            biases,
            inference: false,
            last_input: None,
        }
    }
}

impl Model for Feedforward {
    fn set_inference(&mut self, inference: bool) {
        self.inference = inference;
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        if !self.inference {
            self.last_input = Some(x.clone());
        }

        // Compute weights * x
        let wx = &self.weights.value * x;

        // Broadcast biases across all columns (batch dimension)
        let mut output = Mat::zeros(wx.nrows(), wx.ncols());

        for j in 0..wx.ncols() {
            zip!(&mut output.col_mut(j), &wx.col(j), &self.biases.value.col(0))
                .for_each(|unzip!(out, wx_val, bias)| {
                    *out = *wx_val + *bias;
                });
        }

        output
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        let last_input = self
                .last_input
                .as_ref()
                .expect("Last input is not cached, cannot compute gradient");

        // Gradient for weights: loss * last_input^T
        self.weights.grad += loss * last_input.transpose();

        // Gradient for biases: sum across batch dimension (columns)
        for j in 0..loss.ncols() {
            for i in 0..loss.nrows() {
                self.biases.grad[(i, 0)] += loss[(i, j)];
            }
        }

        // Return gradient w.r.t. input
        &self.weights.value.transpose() * loss
    }

    fn zero_grad(&mut self) {
        self.weights.grad = Mat::zeros(self.weights.value.nrows(), self.weights.value.ncols());
        self.biases.grad = Mat::zeros(self.biases.value.nrows(), self.biases.value.ncols());
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weights, &mut self.biases]
    }
}

#[derive(Serialize, Deserialize)]
pub struct SigmoidActivation {
    inference: bool,
    last_output: Option<Mat<f32>>,
}

impl SigmoidActivation {
    /// Creates a new sigmoid activation layer
    pub fn new() -> Self {
        Self {
            inference: false,
            last_output: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct RELUActivation {
    inference: bool,
    last_input: Option<Mat<f32>>,
}

impl RELUActivation {
    /// Creates a new ReLU activation layer
    pub fn new() -> Self {
        Self {
            inference: false,
            last_input: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct SoftmaxActivation {
    inference: bool,
    last_output: Option<Mat<f32>>,
}

impl SoftmaxActivation {
    /// Creates a new softmax activation layer
    pub fn new() -> Self {
        Self {
            inference: false,
            last_output: None,
        }
    }
}

impl Model for SigmoidActivation {
    fn set_inference(&mut self, inference: bool) {
        self.inference = inference;
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let mut output = Mat::zeros(x.nrows(), x.ncols());

        zip!(&mut output, x).for_each(|unzip!(out, x_val)| {
            *out = 1.0 / (1.0 + (-*x_val).exp());
        });

        if !self.inference {
            self.last_output = Some(output.clone());
        }

        output
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        let sigmoid_output = self
            .last_output
            .as_ref()
            .expect("Last output is not cached, cannot compute gradient");

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // Using cached sigmoid output to avoid recomputation
        let mut grad_input = Mat::zeros(loss.nrows(), loss.ncols());

        zip!(&mut grad_input, loss, sigmoid_output).for_each(|unzip!(grad, loss_val, s)| {
            let sigmoid_grad = *s * (1.0 - *s);
            *grad = *loss_val * sigmoid_grad;
        });

        grad_input
    }

    fn zero_grad(&mut self) { }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![]  // No trainable parameters
    }
}

impl Model for RELUActivation {
    fn set_inference(&mut self, inference: bool) {
        self.inference = inference;
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        if !self.inference {
            self.last_input = Some(x.clone());
        }

        // ReLU(x) = max(0, x)
        let mut output = Mat::zeros(x.nrows(), x.ncols());

        zip!(&mut output, x).for_each(|unzip!(out, x_val)| {
            *out = x_val.max(0.0);
        });

        output
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        let last_input = self
            .last_input
            .as_ref()
            .expect("Last input is not cached, cannot compute gradient");

        // ReLU'(x) = 1 if x > 0, else 0
        let mut grad_input = Mat::zeros(loss.nrows(), loss.ncols());

        zip!(&mut grad_input, loss, last_input).for_each(|unzip!(grad, loss_val, x)| {
            *grad = if *x > 0.0 { *loss_val } else { 0.0 };
        });

        grad_input
    }

    fn zero_grad(&mut self) { }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![]  // No trainable parameters
    }
}

impl Model for SoftmaxActivation {
    fn set_inference(&mut self, inference: bool) {
        self.inference = inference;
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        // Compute softmax for each column (sample) independently
        // softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        // Subtracting max(x) for numerical stability

        let result = Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
            // Find max in this column for numerical stability
            let mut max_val = x[(0, j)];
            for row in 1..x.nrows() {
                max_val = max_val.max(x[(row, j)]);
            }

            // Compute exp(x_i - max)
            let exp_val = (x[(i, j)] - max_val).exp();

            // Compute sum of exp for this column
            let mut sum_exp = 0.0;
            for row in 0..x.nrows() {
                sum_exp += (x[(row, j)] - max_val).exp();
            }

            exp_val / sum_exp
        });

        if !self.inference {
            self.last_output = Some(result.clone());
        }

        result
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        let softmax_output = self
            .last_output
            .as_ref()
            .expect("Last output is not cached, cannot compute gradient");

        // Softmax gradient: grad_input = softmax * (grad_output - dot(grad_output, softmax))
        // For each column (sample), compute independently
        Mat::from_fn(loss.nrows(), loss.ncols(), |i, j| {
            // Compute dot product of grad_output and softmax for this column
            let mut dot_product = 0.0;
            for row in 0..loss.nrows() {
                dot_product += loss[(row, j)] * softmax_output[(row, j)];
            }

            // grad_input[i,j] = softmax[i,j] * (grad_output[i,j] - dot_product)
            softmax_output[(i, j)] * (loss[(i, j)] - dot_product)
        })
    }

    fn zero_grad(&mut self) { }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![]  // No trainable parameters
    }
}

impl Model for Vec<Box<dyn Model>> {
    fn set_inference(&mut self, inference: bool) {
        for model in self.iter_mut() {
            model.set_inference(inference);
        }
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        self.iter_mut().fold(x.clone(), |acc, model| model.forward(&acc))
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        self.iter_mut().rev().fold(loss.clone(), |acc, model| model.gradient(&acc))
    }

    fn zero_grad(&mut self) {
        self.iter_mut().for_each(|model| model.zero_grad());
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        self.iter_mut()
            .flat_map(|model| model.parameters_mut())
            .collect()
    }
}

/// Enum wrapper for all concrete model types to enable serialization
#[derive(Serialize, Deserialize)]
pub enum ModelType {
    Feedforward(Feedforward),
    Sigmoid(SigmoidActivation),
    RELU(RELUActivation),
    Softmax(SoftmaxActivation),
}

impl Model for ModelType {
    fn set_inference(&mut self, inference: bool) {
        match self {
            ModelType::Feedforward(m) => m.set_inference(inference),
            ModelType::Sigmoid(m) => m.set_inference(inference),
            ModelType::RELU(m) => m.set_inference(inference),
            ModelType::Softmax(m) => m.set_inference(inference),
        }
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        match self {
            ModelType::Feedforward(m) => m.forward(x),
            ModelType::Sigmoid(m) => m.forward(x),
            ModelType::RELU(m) => m.forward(x),
            ModelType::Softmax(m) => m.forward(x),
        }
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        match self {
            ModelType::Feedforward(m) => m.gradient(loss),
            ModelType::Sigmoid(m) => m.gradient(loss),
            ModelType::RELU(m) => m.gradient(loss),
            ModelType::Softmax(m) => m.gradient(loss),
        }
    }

    fn zero_grad(&mut self) {
        match self {
            ModelType::Feedforward(m) => m.zero_grad(),
            ModelType::Sigmoid(m) => m.zero_grad(),
            ModelType::RELU(m) => m.zero_grad(),
            ModelType::Softmax(m) => m.zero_grad(),
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        match self {
            ModelType::Feedforward(m) => m.parameters_mut(),
            ModelType::Sigmoid(m) => m.parameters_mut(),
            ModelType::RELU(m) => m.parameters_mut(),
            ModelType::Softmax(m) => m.parameters_mut(),
        }
    }
}

impl Model for Vec<ModelType> {
    fn set_inference(&mut self, inference: bool) {
        for model in self.iter_mut() {
            model.set_inference(inference);
        }
    }

    fn forward(&mut self, x: &Mat<f32>) -> Mat<f32> {
        self.iter_mut().fold(x.clone(), |acc, model| model.forward(&acc))
    }

    fn gradient(&mut self, loss: &Mat<f32>) -> Mat<f32> {
        self.iter_mut().rev().fold(loss.clone(), |acc, model| model.gradient(&acc))
    }

    fn zero_grad(&mut self) {
        self.iter_mut().for_each(|model| model.zero_grad());
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        self.iter_mut()
            .flat_map(|model| model.parameters_mut())
            .collect()
    }
}
