use ndarray::{s, Array1, Array2};
use crate::activation::Activation;
use crate::layer::Layer;
use crate::layer_request_info::LayerRequestInfo;

pub struct SequentialModel {
    layers : Vec<Layer>,
    layer_count : usize,
}

impl SequentialModel {
    pub fn new(
        sample_feature_size: usize,
        layer_request_infos: &Vec<LayerRequestInfo>,
    ) -> Self {
        let layer_count = layer_request_infos.len();
        let mut layers : Vec<Layer> = Vec::with_capacity(layer_count);

        let mut column_size: usize = sample_feature_size + 1;
        for layer_index in 0..layer_count {
            let unit_count = layer_request_infos[layer_index].unit_count;
            let array2: Array2<f64> = Array2::zeros(
                (column_size, unit_count + 1)
            );

            layers.push(
                Layer::new(array2, layer_request_infos[layer_index].activation)
            );

            column_size = unit_count + 1;
        }

        Self { layers, layer_count }
    }

    pub fn generate_sequential_model_with_layers(layers: Vec<Layer>, sample_feature_size: usize,) -> Self {
        Self::validate_layers(&layers, sample_feature_size);

        let layer_count = layers.len();
        Self { layers, layer_count }
    }

    pub fn train_model(
        &mut self,
        a0_matrix : &Array2<f64>,
        outputs: &Array1<f64>,
        learning_rate: f64,
        loop_count_for_each_unit: usize
    ) {
        let mut a_output_matrices: Vec<Array2<f64>> = Vec::with_capacity(self.layer_count);
        let a0_new_matrix = Self::add_ones_column_to_a0_matrix(a0_matrix);
        a_output_matrices.push(Self::build_a_next(&self.layers[0], &a0_new_matrix));
        for layer_index in 1..self.layer_count {
            a_output_matrices.push(
                Self::build_a_next(&self.layers[layer_index], &a_output_matrices[layer_index -1])
            )
        }

        for layer_index_from_end in (0..self.layer_count).rev() {
            let how_many_layers = self.layer_count - layer_index_from_end;
            let mut layers: Vec<Layer> = Vec::with_capacity(how_many_layers);
            for layer_index in layer_index_from_end..self.layer_count {
                layers.push(self.layers[layer_index].clone());
            }

            let first_layer_row_count_minus_1 = layers[0].get_matrix().nrows() - 1;
            let mut model_for_training: SequentialModel =
                Self::generate_sequential_model_with_layers(layers, first_layer_row_count_minus_1);
            
            let unit_count = model_for_training.layers[0].get_matrix().ncols() - 1;
            let weight_and_bias_count = model_for_training.layers[0].get_matrix().nrows();
            let mut a0_matrix_for_model = &a0_new_matrix;
            if layer_index_from_end != 0 {
                a0_matrix_for_model = &a_output_matrices[layer_index_from_end - 1];
            }


            for unit_index in 0..unit_count {
                let mut new_weights: Array1<f64> = Array1::ones(model_for_training.layers[0].get_matrix().nrows());

                for _ in 0..loop_count_for_each_unit {
                    for w_index in 0..weight_and_bias_count {
                        let mut partial_derivative: f64 = 0.0;
                        let m = a0_matrix_for_model.nrows();
                        for i in 0..m {
                            partial_derivative += a0_matrix_for_model[[i, w_index]] *
                                (model_for_training.predict(&a0_matrix.row(i).to_owned()) - outputs[i])
                        }
                        partial_derivative = partial_derivative / m as f64;

                        let new_weight: f64 =
                            model_for_training.layers[0].get_matrix()[[w_index, unit_index]] -
                                learning_rate * partial_derivative;

                        new_weights[w_index] = new_weight;
                    }
                    model_for_training.layers[0].get_mut_matrix().column_mut(unit_index).assign(&new_weights);
                }
            }

            self.layers[layer_index_from_end] = model_for_training.layers[0].clone();
        }
    }

    ///Last layer critical layer for cost !!
    pub fn cost(&self, a0_matrix: &Array2<f64>, outputs: &Array1<f64>) -> f64 {
        let result: f64 ;

        let predict_array: Array1<f64> = self.predict_array(a0_matrix);
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

    ///Cost Function definition convert from mine to Andrew Ng's definition. For derivative reasoning
    fn get_cost_for_linear_activation(predicted_array: &Array1<f64>, real_outputs: &Array1<f64>) -> f64 {
        let array_length = predicted_array.len();
        let mut cost: f64 = 0.;
        for i in 0..array_length {
            cost += (predicted_array[i] - real_outputs[i]).powi(2);
        }

        cost / (2. * array_length as f64)
    }

    pub fn loss(&self, input_sample: &Array1<f64>, output: f64) -> f64 {
        let result : f64;

        let predict = self.predict(input_sample);
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

    fn predict_array(&self, a0_matrix: &Array2<f64>) -> Array1<f64> {
        Self::first_layer_row_size_and_a0_feature_size_validate(
            &self.layers[0], a0_matrix.ncols()
        );

        let mut a_previous_matrix: Array2<f64> = Self::add_ones_column_to_a0_matrix(a0_matrix);
        for layer_index in 0..self.layer_count {
            a_previous_matrix = Self::build_a_next(&self.layers[layer_index], &a_previous_matrix);
        }

        a_previous_matrix.column(0).to_owned()
    }

    ///This is model's f function
    pub fn predict(&self, input: &Array1<f64>) -> f64 {
        let mut input_matrix: Array2<f64> = Array2::zeros((1, input.len()));
        input_matrix.row_mut(0).assign(input);

        self.predict_array(&input_matrix)[0]
    }

    fn build_a_next(layer: &Layer, a_previous: &Array2<f64>) -> Array2<f64> {
        let z_matrix_linear_output = &a_previous.dot(layer.get_matrix());
        let mut a_next: Array2<f64>
            = z_matrix_linear_output.mapv(|z_ij| layer.get_activation_function().apply(z_ij));

        let column_size = a_next.ncols();
        a_next.column_mut(column_size - 1).fill(1.);
        a_next
    }

    fn add_ones_column_to_a0_matrix(a0_matrix: &Array2<f64>) -> Array2<f64> {
        let feature_size: usize = a0_matrix.ncols();
        let mut a0_new: Array2<f64> = Array2::ones(
            (a0_matrix.nrows(), feature_size+1)
        );
        a0_new.slice_mut(s![.., ..feature_size]).assign(a0_matrix);
        a0_new
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

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    fn validate_layers(layers: &Vec<Layer>, sample_feature_size: usize,) {
        let layer_count = layers.len();
        Self::first_layer_row_size_and_a0_feature_size_validate(&layers[0], sample_feature_size);

        let mut layer_previous_column_size: usize = layers[0].get_matrix().ncols();
        for layer_index in 1..layer_count {
            if layers[layer_index].get_matrix().nrows() != layer_previous_column_size {
                panic!("Layer_{} row size and its previous column size mismatch !!", layer_index + 1);
            }

            layer_previous_column_size = layers[layer_index].get_matrix().ncols();
        }
    }

    fn first_layer_row_size_and_a0_feature_size_validate(layer_first: &Layer, a0_feature_count: usize)
    {
        if layer_first.get_matrix().nrows() != a0_feature_count + 1 {
            panic!("a0_column_size and first layer row count mismatch !!");
        }
    }






}