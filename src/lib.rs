


pub mod layer;
pub mod activation;
pub mod sequential_model;
pub mod loss;
pub mod optimizer;
pub mod random;
mod matrix_ops;
pub mod column_based_scaling;
pub mod onnx;

/// Samples, targets and weights are all safe_matmul matrices. Re-exported so
/// users don't need safe_matmul in their own Cargo.toml.
pub use safe_matmul::matrix::Matrix;

/// Everything a Keras-style program needs: `use neuralflow::prelude::*;`
pub mod prelude {
    pub use crate::Matrix;
    pub use crate::activation::Activation;
    pub use crate::layer::{Dense, Input, Layer};
    pub use crate::loss::{BinaryCrossentropy, Loss, MeanSquaredError};
    pub use crate::optimizer::{Adam, Optimizer, SGD};
    pub use crate::random::set_random_seed;
    pub use crate::sequential_model::{FitOptions, History, Sequential};
}
