use ndarray::{s, Array1, Array2};
use crate::activation::Activation;
use crate::layer::Layer;
use crate::layer_request_info::LayerRequestInfo;

pub struct SequentialModel {
    layers : Array1<Layer>,
    layer_count : usize,

    ///a_matrices will always add 1 column. Because matrix multiplication and derivative shortly.
    ///Model will add 1 column inside ones to a_matrices.
    a_matrices: Array1<Array2<f64>>
}

impl SequentialModel {

    ///a_matrices will always add 1 column. Because matrix multiplication and derivative shortly.
    ///Model will add 1 column inside ones to a_matrices.
    pub fn new(
        layer_request_infos: &Vec<LayerRequestInfo>,
        a0_matrix: &Array2<f64>,
    ) -> Self {
        let layer_count = layer_request_infos.len();
        Self::model_should_have_at_least_one_layer(layer_count);

        let mut layers_vec: Vec<Layer> = Vec::with_capacity(layer_count);

        let mut column_size: usize = a0_matrix.ncols() + 1;
        for layer_index in 0..layer_count {
            let unit_count = layer_request_infos[layer_index].unit_count;
            let array2: Array2<f64> = Array2::zeros(
                (column_size, unit_count + 1)
            );

            layers_vec[layer_index] = Layer::new(array2, layer_request_infos[layer_index].activation);

            column_size = unit_count + 1;
        }
        let layers : Array1<Layer> = Array1::from_vec(layers_vec);

        let mut a_matrices: Array1<Array2<f64>> = Array1::default(layer_count + 1);
        let mut a0_new_matrix: Array2<f64> = Array2::ones(
            (a0_matrix.nrows(), a0_matrix.ncols() + 1)
        );
        a0_new_matrix.slice_mut(s![.., ..a0_matrix.ncols()]).assign(a0_matrix);
        a_matrices[0] = a0_new_matrix;

        Self { layers, layer_count, a_matrices }
    }

    ///Model will always add 1 column a_matrices. Because matrix multiplication and derivative shortly.
    pub fn generate_sequential_model_with_layers(
        layers: Array1<Layer>,
        a0_matrix: &Array2<f64>,
    ) -> Self {
        let n0 = a0_matrix.ncols();
        Self::validate_layers(&layers, n0);

        let mut a0_new_matrix: Array2<f64> = Array2::ones(
            (a0_matrix.nrows(), n0 + 1)
        );
        a0_new_matrix.slice_mut(s![.., ..n0]).assign(a0_matrix);

        let layer_count = layers.len();
        let mut a_matrices: Array1<Array2<f64>> = Array1::default(layer_count + 1);
        a_matrices[0] = a0_new_matrix;
        Self { layers, layer_count, a_matrices }
    }

    pub fn train_model(
        &mut self,
        loop_count: usize,
        learning_rate: f64
    ) {
        let layer_0: &Layer = &self.layers[0];
        let unit_count = layer_0.get_matrix().ncols() - 1;
        let row_count = layer_0.get_matrix().nrows();

        for _ in 0..loop_count {
            let mut layer_0_new_matrix: Array2<f64> = Array2::zeros((row_count, unit_count + 1));

            //FixMe: Convert to Multithread processing
            for u in 0..unit_count {
                //ToDo: ****************** I'm HERE *********************
            }
        }
    }

    ///unit_index == layer_column_size - 1.
    fn layer_inside_linear_function_derivative_j(
        &self,
        layer_index: usize,
        unit_index: usize,
        j: usize,
        y_array: &Array1<f64>,
    ) -> f64 {
        let mut result : f64 = 0.0;
        let m: usize = self.a_matrices[0].nrows();

        let z_array_column: Array1<f64>
            = self.layers[layer_index + 1]
            .get_z_matrix_linear_output(&self.a_matrices[layer_index + 1]).column(unit_index)
            .to_owned();

        let mut unit_sum: f64 = 0.;
        for i in 0..m {
            unit_sum += z_array_column[i];
            result += self.a_matrices[layer_index][[i,j]] * (unit_sum- y_array[i]);
        }

        result * 2. / (m as f64)
    }

    ///Last layer critical layer for cost !!
    pub fn cost(&mut self, outputs: &Array1<f64>) -> f64 {
        let result: f64 ;

        let predict_array: Array1<f64> = self.predict_array_for_learning();
        match self.layers[self.layer_count - 1].get_activation_function() {
            Activation::Sigmoid => {
                result = Self::get_mean_loss(&predict_array, outputs,)
            },

            Activation::Linear | Activation::ReLU => {
                result = Self::get_cost_for_linear_activation(&predict_array, outputs,)
            },
        }

        result
    }

    fn get_mean_loss(predicted_array: &Array1<f64>, real_outputs: &Array1<f64>) -> f64 {
        let array_length = predicted_array.len();
        let mut sum_loss: f64 = 0.;
        for i in 0..array_length {
            sum_loss += Self::loss_for_sigmoid(predicted_array[i], real_outputs[i])
        }
        sum_loss / (array_length as f64)
    }

    ///My Cost Function definition : Divided by sample size, Not (sample_size * 2) !!
    /// Source :
    /// https://github.com/alibozlak/multivariable_linear_regression/blob/master/math/001_dJ_daj_partial_derivative.pdf
    fn get_cost_for_linear_activation(predicted_array: &Array1<f64>, real_outputs: &Array1<f64>) -> f64 {
        let array_length = predicted_array.len();
        let mut cost: f64 = 0.;
        for i in 0..array_length {
            cost += (predicted_array[i] - real_outputs[i]).powi(2);
        }

        cost / array_length as f64
    }

    pub fn loss(&mut self, output: f64) -> f64 {
        let result : f64;

        let predict = self.predict_array_for_learning()[0];
        match self.layers[self.layer_count - 1].get_activation_function() {
            Activation::Sigmoid => {
                result = Self::loss_for_sigmoid(predict, output);
            },

            Activation::Linear | Activation::ReLU => {
                result = (predict - output).powi(2);
            },
        }

        result
    }

    fn loss_for_sigmoid(predict: f64, real_output: f64) -> f64 {
        (real_output - 1.) * (1. - predict).ln() - real_output * predict.ln()
    }

    pub fn predict_array_for_learning(&mut self) -> Array1<f64> {
        let mut a_previous_matrix: Array2<f64> = self.a_matrices[0].clone();
        for layer_index in 0..self.layer_count {
            a_previous_matrix = self.layers[layer_index].build_a_next(a_previous_matrix);
            self.a_matrices[layer_index + 1] = a_previous_matrix.clone();
        }

        a_previous_matrix.column(0).to_owned()
    }

    pub fn summary(&self) -> String {
        let mut summary: String = format!("Layer_Count: {}\n", self.layer_count);
        for layer_index in 0..self.layer_count {
            summary.push_str(&format!(
                "Layer_{}: {}\n", layer_index + 1, self.layers[layer_index].summary())
            );
        }

        summary
    }

    pub fn get_layers(&self) -> &Array1<Layer> {
        &self.layers
    }

    pub fn get_a0_matrix(&self) -> Array2<f64> {
        self.a_matrices[0].clone()
    }

    fn validate_layers(layers: &Array1<Layer>, sample_feature_size: usize,) {
        let layer_count = layers.len();

        Self::model_should_have_at_least_one_layer(layer_count);

        Self::first_layer_row_size_and_a0_feature_size_validate(&layers[0], sample_feature_size);

        let mut layer_previous_column_size: usize = layers[0].get_matrix().ncols();
        for layer_index in 1..layer_count {
            if layers[layer_index].get_matrix().nrows() != layer_previous_column_size {
                panic!("Layer_{} row size and its previous column size mismatch !!", layer_index + 1);
            }

            layer_previous_column_size = layers[layer_index].get_matrix().ncols();
        }
    }

    fn first_layer_row_size_and_a0_feature_size_validate(
        layer_first: &Layer, a0_feature_count: usize
    ) -> usize {
        if layer_first.get_matrix().nrows() != a0_feature_count + 1 {
            panic!("a0_column_size and first layer row count mismatch !!");
        }

        a0_feature_count
    }

    fn model_should_have_at_least_one_layer(layer_count: usize) {
        if layer_count < 1 {
            panic!("Model should have at least one layer");
        }
    }






}