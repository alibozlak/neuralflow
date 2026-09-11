use safe_matmul::matrix::Matrix;
use crate::matrix_ops::MatrixExt;

/// Moves the model's variables (every W and b) against their gradients.
pub trait Optimizer {
    /// `variables[i]` moves using `gradients[i]`. The model passes them in the
    /// same order on every step, so an optimizer can keep state per index.
    fn apply_gradients(&mut self, variables: &mut [&mut Matrix], gradients: &[Matrix]);
}

fn validate_lengths(variables: &[&mut Matrix], gradients: &[Matrix]) {
    if variables.len() != gradients.len() {
        panic!("Got {} variables but {} gradients !!", variables.len(), gradients.len());
    }
}

/// Keras' `SGD(learning_rate)`: plain gradient descent, Ng's `w = w - α·dJ/dw`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Default for SGD {
    /// Keras' default learning rate, 0.01.
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Optimizer for SGD {
    fn apply_gradients(&mut self, variables: &mut [&mut Matrix], gradients: &[Matrix]) {
        validate_lengths(variables, gradients);

        let learning_rate = self.learning_rate;
        for (variable, gradient) in variables.iter_mut().zip(gradients) {
            **variable = variable.zip_map(gradient, |w, dw| w - learning_rate * dw);
        }
    }
}

/// Keras' `Adam(learning_rate)`, written the way Ng's Deep Learning course
/// does, with t counting the steps:
///
/// ```text
/// m = β1·m + (1 - β1)·dw        v = β2·v + (1 - β2)·dw²
/// m̂ = m / (1 - β1^t)            v̂ = v / (1 - β2^t)
/// w = w - α·m̂ / (√v̂ + ε)
/// ```
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    /// t
    iterations: i32,
    /// m and v of every variable, created on the first step.
    first_moments: Vec<Matrix>,
    second_moments: Vec<Matrix>,
}

impl Adam {
    /// β1 = 0.9, β2 = 0.999 and ε = 1e-7, Keras' defaults.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-7,
            iterations: 0,
            first_moments: Vec::new(),
            second_moments: Vec::new(),
        }
    }
}

impl Default for Adam {
    /// Keras' default learning rate, 0.001.
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl Optimizer for Adam {
    fn apply_gradients(&mut self, variables: &mut [&mut Matrix], gradients: &[Matrix]) {
        validate_lengths(variables, gradients);

        // First step, or the model changed shape (a layer was added): start from zero.
        let state_fits = self.first_moments.len() == gradients.len()
            && self.first_moments.iter().zip(gradients).all(|(m, dw)| m.shape() == dw.shape());
        if !state_fits {
            self.first_moments = gradients.iter()
                .map(|dw| Matrix::zeros(dw.row_count(), dw.col_count()))
                .collect();
            self.second_moments = self.first_moments.clone();
            self.iterations = 0;
        }

        self.iterations += 1;
        let (learning_rate, beta_1, beta_2, epsilon) = (self.learning_rate, self.beta_1, self.beta_2, self.epsilon);
        let first_correction = 1. - beta_1.powi(self.iterations);
        let second_correction = 1. - beta_2.powi(self.iterations);

        for (index, (variable, gradient)) in variables.iter_mut().zip(gradients).enumerate() {
            let m = self.first_moments[index].zip_map(gradient, |m, dw| beta_1 * m + (1. - beta_1) * dw);
            let v = self.second_moments[index].zip_map(gradient, |v, dw| beta_2 * v + (1. - beta_2) * dw * dw);
            let step = m.zip_map(&v, |m, v| {
                learning_rate * (m / first_correction) / ((v / second_correction).sqrt() + epsilon)
            });

            **variable = variable.zip_map(&step, |w, step| w - step);
            self.first_moments[index] = m;
            self.second_moments[index] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(datas: &[f64]) -> Matrix {
        Matrix::from_vec(1, datas.len(), datas.to_vec()).unwrap()
    }

    #[test]
    fn sgd_steps_against_the_gradient() {
        let mut w = row(&[1., 2.]);
        SGD::new(0.1).apply_gradients(&mut [&mut w], &[row(&[0.5, -1.])]);

        assert!((w.get(0, 0).unwrap() - 0.95).abs() < 1e-12);
        assert!((w.get(0, 1).unwrap() - 2.1).abs() < 1e-12);
    }

    /// On the first step m̂/√v̂ is dw/|dw|, so every weight moves by about α
    /// however big or small its gradient is.
    #[test]
    fn adam_first_step_moves_every_weight_by_the_learning_rate() {
        let mut w = row(&[0., 0.]);
        Adam::new(0.01).apply_gradients(&mut [&mut w], &[row(&[1000., -0.001])]);

        assert!((w.get(0, 0).unwrap() + 0.01).abs() < 1e-6);
        assert!((w.get(0, 1).unwrap() - 0.01).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "Got 1 variables but 2 gradients")]
    fn every_variable_needs_a_gradient() {
        let mut w = row(&[0.]);
        SGD::default().apply_gradients(&mut [&mut w], &[row(&[1.]), row(&[1.])]);
    }
}
