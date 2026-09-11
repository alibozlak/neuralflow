
use safe_matmul::matrix::Matrix;
use crate::activation::Activation;

#[derive(Clone)]
pub struct Layer {
    weight_matrix: Matrix,
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

        Self { weight_matrix, bias_matrix, activation }
    }

    pub fn get_matrices(self) -> (Matrix, Matrix) {
        (self.weight_matrix, self.bias_matrix)
    }

    pub fn get_activation_function(&self) -> Activation {
        self.activation
    }

    pub fn summary(&self) -> String {
        format!(
            "Matrices shapes = {}x{}, activation_func = {}",
            self.weight_matrix.row_count(),
            self.weight_matrix.col_count(),
            self.activation
        )
    }

    pub fn validate_weight_and_bias_matrices_shapes(weight_matrix: &Matrix, bias_matrix: &Matrix) {
        if weight_matrix.row_count() != bias_matrix.row_count() ||
            weight_matrix.col_count() != bias_matrix.col_count()
        {
            panic!("Weight matrix and Bias matrix must have the same shape !!")
        }
    }


}