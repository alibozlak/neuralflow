
use safe_matmul::matrix::Matrix;
use crate::activation::Activation;
use crate::matrix_ops::MatrixExt;
use crate::random;

/// Keras' `Input(shape=(feature_count,))`: how many features every sample
/// (every row of X) has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    feature_count: usize,
}

impl Input {
    pub fn new(feature_count: usize) -> Self {
        if feature_count == 0 {
            panic!("Input needs at least 1 feature !!");
        }

        Self { feature_count }
    }

    pub fn get_feature_count(&self) -> usize {
        self.feature_count
    }
}

/// Keras' `Dense(units, activation=..., name=...)`. It only describes the
/// layer: `Sequential` builds the real `Layer` once it knows the input size.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    units: usize,
    activation: Activation,
    name: Option<String>,
}

impl Dense {
    pub fn new(units: usize, activation: Activation) -> Self {
        if units == 0 {
            panic!("Dense layer needs at least 1 unit !!");
        }

        Self { units, activation, name: None }
    }

    /// Keras' `name="layer1"`, what `Sequential::get_layer` looks for.
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Glorot-uniform weights and zero biases, Keras' defaults for Dense.
    /// The weights have to be random: if they started equal, every unit of
    /// the layer would get the same gradient and learn the same thing.
    pub(crate) fn build(self, input_count: usize, default_name: String) -> Layer {
        let limit = (6. / (input_count + self.units) as f64).sqrt();
        let weights = (0..input_count * self.units)
            .map(|_| random::uniform(-limit, limit))
            .collect();

        Layer {
            name: self.name.unwrap_or(default_name),
            weight_matrix: Matrix::from_vec(input_count, self.units, weights).unwrap(),
            bias_matrix: Matrix::zeros(1, self.units),
            activation: self.activation,
        }
    }
}

/// A fully connected layer in Andrew Ng's notation:
/// `Z = A_in · W + b`, `A_out = g(Z)`.
///
/// Every row of `A_in` is one sample, so column j of W holds unit j's
/// weights — the same layout Keras' `get_weights()` returns.
#[derive(Debug, Clone)]
pub struct Layer {
    name: String,
    /// (inputs x units)
    weight_matrix: Matrix,
    /// (1 x units): one bias per unit, added to every sample.
    bias_matrix: Matrix,
    activation: Activation,
}

impl Layer {
    pub fn new(
        weight_matrix: Matrix,
        bias_matrix: Matrix,
        activation: Activation,
    ) -> Self {
        Self::validate_weight_and_bias_matrices_shapes(&weight_matrix, &bias_matrix);

        Self { name: String::from("dense"), weight_matrix, bias_matrix, activation }
    }

    /// Keras' `layer(x)`: the layer's output for every sample in `a_in`.
    pub fn call(&self, a_in: &Matrix) -> Matrix {
        self.activation.a_with_matrix(&self.z_matrix(a_in))
    }

    /// Z = A_in · W + b
    pub fn z_matrix(&self, a_in: &Matrix) -> Matrix {
        if a_in.col_count() != self.weight_matrix.row_count() {
            panic!("Layer '{}' expects {} inputs per sample, got {} !!",
                   self.name, self.weight_matrix.row_count(), a_in.col_count()
            );
        }

        a_in.matmul(&self.weight_matrix).unwrap().add_row(&self.bias_matrix)
    }

    /// Keras' `get_weights()`: (W, b).
    pub fn get_weights(&self) -> (&Matrix, &Matrix) {
        (&self.weight_matrix, &self.bias_matrix)
    }

    /// Keras' `set_weights([W, b])`. The shapes can't change.
    pub fn set_weights(&mut self, weight_matrix: Matrix, bias_matrix: Matrix) {
        if weight_matrix.shape() != self.weight_matrix.shape() {
            panic!("Layer '{}' needs a {}x{} weight matrix, got {}x{} !!",
                   self.name, self.weight_matrix.row_count(), self.weight_matrix.col_count(),
                   weight_matrix.row_count(), weight_matrix.col_count()
            );
        }
        Self::validate_weight_and_bias_matrices_shapes(&weight_matrix, &bias_matrix);

        self.weight_matrix = weight_matrix;
        self.bias_matrix = bias_matrix;
    }

    /// W and b, in the same order as their gradients.
    pub(crate) fn variables_mut(&mut self) -> [&mut Matrix; 2] {
        [&mut self.weight_matrix, &mut self.bias_matrix]
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_units(&self) -> usize {
        self.weight_matrix.col_count()
    }

    pub fn get_activation_function(&self) -> Activation {
        self.activation
    }

    /// Keras' `count_params()`: every weight plus every bias.
    pub fn count_params(&self) -> usize {
        self.weight_matrix.row_count() * self.weight_matrix.col_count() + self.bias_matrix.col_count()
    }

    pub fn validate_weight_and_bias_matrices_shapes(weight_matrix: &Matrix, bias_matrix: &Matrix) {
        if bias_matrix.row_count() != 1 || bias_matrix.col_count() != weight_matrix.col_count() {
            panic!("Bias matrix must be 1x{} (one bias per unit), got {}x{} !!",
                   weight_matrix.col_count(), bias_matrix.row_count(), bias_matrix.col_count()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(row_count: usize, col_count: usize, datas: &[f64]) -> Matrix {
        Matrix::from_vec(row_count, col_count, datas.to_vec()).unwrap()
    }

    /// Two samples through three units, by hand:
    /// X·W = [[1, 2, 1], [3, 4, 1]], plus b = [[-1, -1, 1], [1, 1, 1]].
    #[test]
    fn call_computes_g_of_x_w_plus_b() {
        let x = matrix(2, 2, &[1., 2., 3., 4.]);
        let w = matrix(2, 3, &[1., 0., -1., 0., 1., 1.]);
        let b = matrix(1, 3, &[-2., -3., 0.]);
        let layer = Layer::new(w, b, Activation::ReLU);

        assert_eq!(layer.z_matrix(&x).as_slice(), &[-1., -1., 1., 1., 1., 1.]);
        assert_eq!(layer.call(&x).as_slice(), &[0., 0., 1., 1., 1., 1.]);
    }

    /// The logistic neuron from Ng's lab: w = 2, b = -4.5.
    #[test]
    fn a_single_sigmoid_neuron() {
        let layer = Layer::new(matrix(1, 1, &[2.]), matrix(1, 1, &[-4.5]), Activation::Sigmoid);
        let a = layer.call(&matrix(1, 1, &[3.])).get(0, 0).unwrap();
        assert!((a - Activation::Sigmoid.a(1.5)).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Bias matrix must be 1x3")]
    fn bias_needs_one_value_per_unit() {
        Layer::new(matrix(2, 3, &[0.; 6]), matrix(2, 3, &[0.; 6]), Activation::Linear);
    }

    #[test]
    #[should_panic(expected = "needs a 2x3 weight matrix")]
    fn set_weights_keeps_the_shape() {
        let mut layer = Layer::new(matrix(2, 3, &[0.; 6]), matrix(1, 3, &[0.; 3]), Activation::Linear);
        layer.set_weights(matrix(3, 2, &[0.; 6]), matrix(1, 2, &[0.; 2]));
    }

    #[test]
    fn dense_builds_keras_default_weights() {
        let layer = Dense::new(4, Activation::ReLU).build(3, String::from("dense"));
        let (w, b) = layer.get_weights();
        let limit = (6. / 7_f64).sqrt();

        assert_eq!(w.shape(), (3, 4));
        assert!(w.as_slice().iter().all(|value| value.abs() <= limit));
        assert!(w.as_slice().iter().any(|&value| value != w.as_slice()[0]), "weights must differ");
        assert_eq!(b.as_slice(), &[0.; 4]);
        assert_eq!(layer.count_params(), 3 * 4 + 4);
    }

    #[test]
    fn dense_name_overrides_the_default() {
        let unnamed = Dense::new(1, Activation::Linear).build(1, String::from("dense"));
        let named = Dense::new(1, Activation::Linear).name("L1").build(1, String::from("dense"));
        assert_eq!(unnamed.get_name(), "dense");
        assert_eq!(named.get_name(), "L1");
    }
}
