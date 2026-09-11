use safe_matmul::matrix::Matrix;
use crate::activation::Activation;
use crate::matrix_ops::MatrixExt;

/// What `Sequential::compile` minimises. Like Keras, the cost and its
/// gradient average over every sample and every output unit.
pub trait Loss {
    /// The cost J of predicting `y_pred` when the answer is `y_true`.
    fn cost(&self, y_true: &Matrix, y_pred: &Matrix) -> f64;

    /// dJ/dA: how J changes with each prediction.
    fn gradient(&self, y_true: &Matrix, y_pred: &Matrix) -> Matrix;

    /// dJ/dZ of the output layer, where backprop starts. The default is the
    /// chain rule; a loss can override it for an activation it pairs with.
    fn output_gradient(&self, y_true: &Matrix, y_pred: &Matrix, z_matrix: &Matrix, activation: Activation) -> Matrix {
        activation.dz_matrix(&self.gradient(y_true, y_pred), z_matrix)
    }
}

/// How many values J averages over.
fn element_count(matrix: &Matrix) -> f64 {
    (matrix.row_count() * matrix.col_count()) as f64
}

/// Keras keeps predictions this far from 0 and 1 so log(0) never happens.
const EPSILON: f64 = 1e-7;

/// Keras' `BinaryCrossentropy()`, for a sigmoid output and 0/1 targets:
/// `J = -mean(y·log(a) + (1 - y)·log(1 - a))`
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryCrossentropy;

impl Loss for BinaryCrossentropy {
    fn cost(&self, y_true: &Matrix, y_pred: &Matrix) -> f64 {
        let losses = y_true.zip_map(y_pred, |y, a| {
            let a = a.clamp(EPSILON, 1. - EPSILON);
            -(y * a.ln() + (1. - y) * (1. - a).ln())
        });
        losses.as_slice().iter().sum::<f64>() / element_count(y_true)
    }

    fn gradient(&self, y_true: &Matrix, y_pred: &Matrix) -> Matrix {
        let n = element_count(y_true);
        y_true.zip_map(y_pred, |y, a| {
            let a = a.clamp(EPSILON, 1. - EPSILON);
            (-y / a + (1. - y) / (1. - a)) / n
        })
    }

    /// With a sigmoid output the chain rule collapses to `dZ = (A - Y) / n`,
    /// Ng's `dz = a - y`. Unlike the chain rule it keeps learning where the
    /// sigmoid has saturated to exactly 0 or 1 and g'(z) is 0.
    fn output_gradient(&self, y_true: &Matrix, y_pred: &Matrix, z_matrix: &Matrix, activation: Activation) -> Matrix {
        if activation != Activation::Sigmoid {
            return activation.dz_matrix(&self.gradient(y_true, y_pred), z_matrix);
        }

        let n = element_count(y_true);
        y_pred.zip_map(y_true, |a, y| (a - y) / n)
    }
}

/// Keras' `MeanSquaredError()`, for a linear output (regression):
/// `J = mean((a - y)²)`.
///
/// Unlike the cost in Ng's courses there is no 1/2 in front, so
/// `dJ/dA = 2(a - y) / n`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn cost(&self, y_true: &Matrix, y_pred: &Matrix) -> f64 {
        let squared_errors = y_true.zip_map(y_pred, |y, a| (a - y).powi(2));
        squared_errors.as_slice().iter().sum::<f64>() / element_count(y_true)
    }

    fn gradient(&self, y_true: &Matrix, y_pred: &Matrix) -> Matrix {
        let n = element_count(y_true);
        y_true.zip_map(y_pred, |y, a| 2. * (a - y) / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(datas: &[f64]) -> Matrix {
        Matrix::from_vec(datas.len(), 1, datas.to_vec()).unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn binary_crossentropy_by_hand() {
        let y = column(&[1., 0.]);
        let a = column(&[0.8, 0.4]);
        assert_close(BinaryCrossentropy.cost(&y, &a), -(0.8_f64.ln() + 0.6_f64.ln()) / 2.);
    }

    #[test]
    fn binary_crossentropy_survives_certain_wrong_answers() {
        let cost = BinaryCrossentropy.cost(&column(&[1., 0.]), &column(&[0., 1.]));
        assert!(cost.is_finite());
    }

    #[test]
    fn mean_squared_error_by_hand() {
        let y = column(&[1., 2.]);
        let a = column(&[2., 4.]);
        assert_close(MeanSquaredError.cost(&y, &a), (1. + 4.) / 2.);
        assert_eq!(MeanSquaredError.gradient(&y, &a).as_slice(), &[1., 2.]);
    }

    /// dJ/dA against (J(a + h) - J(a - h)) / 2h, one prediction at a time.
    #[test]
    fn gradients_match_the_numerical_slope() {
        let y = column(&[1., 0., 1.]);
        let a = column(&[0.3, 0.6, 0.9]);
        let h = 1e-6;
        let losses: [&dyn Loss; 2] = [&BinaryCrossentropy, &MeanSquaredError];

        for loss in losses {
            let gradient = loss.gradient(&y, &a);
            for i in 0..3 {
                let nudged = |delta: f64| {
                    let mut datas = a.as_slice().to_vec();
                    datas[i] += delta;
                    loss.cost(&y, &column(&datas))
                };
                assert_close(gradient.get(i, 0).unwrap(), (nudged(h) - nudged(-h)) / (2. * h));
            }
        }
    }

    #[test]
    fn sigmoid_shortcut_agrees_with_the_chain_rule() {
        let y = column(&[1., 0.]);
        let z = column(&[0.5, -1.2]);
        let a = Activation::Sigmoid.a_with_matrix(&z);

        let shortcut = BinaryCrossentropy.output_gradient(&y, &a, &z, Activation::Sigmoid);
        let chain = Activation::Sigmoid.dz_matrix(&BinaryCrossentropy.gradient(&y, &a), &z);
        for i in 0..2 {
            assert_close(shortcut.get(i, 0).unwrap(), chain.get(i, 0).unwrap());
        }
    }

    /// z = 40 makes the sigmoid exactly 1.0 in f64, so g'(z) = 0 and the
    /// chain rule would stop learning on a completely wrong answer.
    #[test]
    fn sigmoid_shortcut_keeps_learning_when_saturated() {
        let y = column(&[0.]);
        let z = column(&[40.]);
        let a = Activation::Sigmoid.a_with_matrix(&z);
        assert_eq!(a.get(0, 0), Some(1.));

        let dz = BinaryCrossentropy.output_gradient(&y, &a, &z, Activation::Sigmoid);
        assert_eq!(dz.get(0, 0), Some(1.));
    }
}
