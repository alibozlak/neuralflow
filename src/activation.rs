use std::fmt;
use std::str::FromStr;
use safe_matmul::matrix::Matrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Sigmoid,
    Linear,
    ReLU,
}

impl Activation {

    pub fn a(self, z: f64) -> f64 {

        match self {
            Activation::Sigmoid => 1. / (1. + (-z).exp()),
            Activation::Linear => z,
            Activation::ReLU => z.max(0.0),
        }
    }

    pub fn a_with_matrix(self, z_matrix: &Matrix) -> Matrix {
        let z_matrix_len: usize = z_matrix.row_count() * z_matrix.col_count();
        let mut result_vector: Vec<f64> = Vec::with_capacity(z_matrix_len);

        for i in 0..z_matrix.row_count() {
            for j in 0..z_matrix.col_count() {
                result_vector.push(self.a(z_matrix.get(i,j).unwrap()))
            }
        }
        Matrix::from_vec(z_matrix.row_count(), z_matrix.col_count(), result_vector).unwrap()
    }

    pub fn linear_func_z_matrix(self, samples_matrix: &Matrix, weight_matrix: &Matrix, bias_matrix: &Matrix) -> Matrix {
        self.validate_matrices_shapes(weight_matrix, samples_matrix, bias_matrix);

        weight_matrix.transpose_matmul(samples_matrix).unwrap()
    }

    pub fn z_matrix(self, samples_matrix: &Matrix, weight_matrix: &Matrix, bias_matrix: &Matrix) -> Matrix {
        self.linear_func_z_matrix(samples_matrix, weight_matrix, bias_matrix)
    }

    pub fn linear_func_z(self, one_sample: &Matrix, weights: &Matrix, bias: f64) -> f64 {
        self.validate_weight_and_samples_matrices_shapes(weights, one_sample);

        if one_sample.col_count() != 1  || weights.col_count() != 1 {
            panic!("weights and one sample matrices should have just 1 column !!")
        }

        weights.transpose_matmul(one_sample).unwrap().get(0,0).unwrap() + bias
    }

    pub fn z(self, one_sample: &Matrix, weights: &Matrix, bias: f64) -> f64 {
        self.linear_func_z(one_sample, weights, bias)
    }

    pub fn validate_weight_and_samples_matrices_shapes(self, weight_matrix: &Matrix, samples_matrix: &Matrix) {
        if weight_matrix.row_count() != samples_matrix.row_count() {
            panic!("weight_matrix row count {} should be equal to samples_matrix row count {} !!",
                   weight_matrix.row_count(), samples_matrix.row_count()
            );
        }
    }

    pub fn validate_matrices_shapes(self, weight_matrix: &Matrix, samples_matrix: &Matrix, bias_matrix: &Matrix) {
        self.validate_weight_and_samples_matrices_shapes(weight_matrix, samples_matrix);

        if weight_matrix.col_count() != bias_matrix.row_count() {
            panic!("weight_matrix col_count should be equal bias_matrix row_count !!")
        }

        if samples_matrix.col_count() != weight_matrix.col_count() {
            panic!("samples_matrix col_count should be equal bias_matrix col_count !!")
        }
    }


}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sigmoid => "Sigmoid",
            Self::Linear => "Linear",
            Self::ReLU => "ReLU",
        };
        write!(f, "{name}")
    }
}

impl FromStr for Activation {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "Sigmoid" => Ok(Self::Sigmoid),
            "Linear" => Ok(Self::Linear),
            // "ReLU" => Ok(Self::ReLU),
            other => Err(format!("unknown activation: {other}")),
        }
    }
}