use ndarray::{ Array2 };
use crate::activation::Activation;

pub struct Layer {

    /// Weights fill column step column.
    /// matrix shape = (weight_count + 1, unit_count + 1)
    /// +1s for matrix multiplication :
    /// weight_count + 1 : each unit's weights + bias,
    /// unit_count + 1 : for a_next (layer output)
    matrix: Array2<f64>,

    activation: Activation,
}

impl Layer {
    pub fn new(
        matrix: Array2<f64>,
        activation: Activation,
    ) -> Self {
        Self { matrix, activation }
    }

    /// Weights fill column step column.
    /// matrix shape = (weight_count + 1, unit_count + 1)
    /// +1s for matrix multiplication :
    /// weight_count + 1 : each unit's weights + bias,
    /// unit_count + 1 : for a_next (layer output)
    pub fn get_matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    pub fn get_matrix_mut(&mut self) -> &mut Array2<f64> {
        &mut self.matrix
    }

    pub fn get_activation_function(&self) -> Activation {
        self.activation
    }

    pub fn build_a_next(&self, a_previous: Array2<f64>) -> Array2<f64> {
        let z_matrix_linear_output = self.get_z_matrix_linear_output(&a_previous);
        let mut a_next: Array2<f64>
            = z_matrix_linear_output.mapv(|z_ij| self.get_activation_function().apply(z_ij));

        let column_size = a_next.ncols();
        a_next.column_mut(column_size - 1).fill(1.);
        a_next
    }

    pub fn get_z_matrix_linear_output(&self, a_previous: &Array2<f64>) -> Array2<f64> {
        a_previous.dot(self.get_matrix())
    }

    pub fn summary(&self) -> String {
        format!(
            "Matrix shape = {}x{}, activation_func = {}",
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            self.activation
        )
    }


}