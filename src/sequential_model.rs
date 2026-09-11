
use std::fmt;
use safe_matmul::matrix::Matrix;
use crate::layer::{Dense, Input, Layer};
use crate::loss::Loss;
use crate::matrix_ops::MatrixExt;
use crate::optimizer::Optimizer;
use crate::random;

/// Keras' `fit` keyword arguments. Set what you need and take the rest from
/// `Default`, the way you would leave keyword arguments out:
/// `FitOptions { epochs: 100, ..Default::default() }`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitOptions {
    pub epochs: usize,
    /// Samples per gradient step. More than the data set has means one step per epoch.
    pub batch_size: usize,
    /// Visit the samples in a new random order every epoch.
    pub shuffle: bool,
    /// Print the loss after every epoch.
    pub verbose: bool,
}

impl Default for FitOptions {
    /// Keras' defaults: 1 epoch, batches of 32, shuffled, printed.
    fn default() -> Self {
        Self { epochs: 1, batch_size: 32, shuffle: true, verbose: true }
    }
}

/// What `fit` returns, Keras' `history.history`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    /// The mean training loss of every epoch, first to last.
    pub loss: Vec<f64>,
}

/// Keras' `Sequential`: layers applied one after another.
pub struct Sequential {
    input: Input,
    layers: Vec<Layer>,
    loss: Option<Box<dyn Loss>>,
    optimizer: Option<Box<dyn Optimizer>>,
}

impl Sequential {

    /// Keras' `Sequential([Input(shape=(2,)), Dense(3, activation='sigmoid'), ...])`.
    pub fn new(input: Input, layers: Vec<Dense>) -> Self {
        let mut model = Self { input, layers: Vec::with_capacity(layers.len()), loss: None, optimizer: None };
        for layer in layers {
            model.add(layer);
        }

        model
    }

    /// Keras' `model.add(Dense(...))`: the new layer takes the last layer's
    /// output (or the Input) as its input.
    pub fn add(&mut self, layer: Dense) {
        let input_count = match self.layers.last() {
            Some(last_layer) => last_layer.get_units(),
            None => self.input.get_feature_count(),
        };
        let layer = layer.build(input_count, self.unused_default_name());

        if self.layers.iter().any(|existing| existing.get_name() == layer.get_name()) {
            panic!("Layer names must be unique, '{}' is used twice !!", layer.get_name());
        }
        self.layers.push(layer);
    }

    /// Keras' `model.compile(loss=..., optimizer=...)`.
    pub fn compile(&mut self, loss: impl Loss + 'static, optimizer: impl Optimizer + 'static) {
        self.loss = Some(Box::new(loss));
        self.optimizer = Some(Box::new(optimizer));
    }

    /// Keras' `model.fit(X, y, epochs=..., batch_size=...)`. Every row of `x`
    /// is one sample and the same row of `y` is its target.
    pub fn fit(&mut self, x: &Matrix, y: &Matrix, options: FitOptions) -> History {
        self.validate_samples(x, y);
        self.compiled_loss("fit");
        if options.batch_size == 0 {
            panic!("batch_size must be at least 1 !!");
        }

        let sample_count = x.row_count();
        let mut sample_order: Vec<usize> = (0..sample_count).collect();
        let mut history = History::default();

        for epoch in 1..=options.epochs {
            if options.shuffle {
                random::shuffle(&mut sample_order);
            }

            let mut loss_sum = 0.;
            for batch in sample_order.chunks(options.batch_size) {
                let batch_loss = self.train_step(&x.select_rows(batch), &y.select_rows(batch));
                loss_sum += batch_loss * batch.len() as f64;
            }
            let epoch_loss = loss_sum / sample_count as f64;

            if options.verbose {
                println!("Epoch {epoch}/{} - loss: {epoch_loss:.4}", options.epochs);
            }
            history.loss.push(epoch_loss);
        }

        history
    }

    /// Keras' `model.predict(X)`: the last layer's output for every row of `x`.
    pub fn predict(&self, x: &Matrix) -> Matrix {
        self.validate_features(x);

        let (first_layer, other_layers) = self.layers.split_first().unwrap();
        other_layers.iter().fold(first_layer.call(x), |a_matrix, layer| layer.call(&a_matrix))
    }

    /// Keras' `model.evaluate(X, y)`: the loss on these samples, without training.
    pub fn evaluate(&self, x: &Matrix, y: &Matrix) -> f64 {
        self.validate_samples(x, y);
        self.compiled_loss("evaluate").cost(y, &self.predict(x))
    }

    /// Keras' `model.get_layer("layer1")`.
    pub fn get_layer(&self, name: &str) -> &Layer {
        &self.layers[self.layer_index(name)]
    }

    /// `get_layer` for changing the layer, e.g. with `set_weights`.
    pub fn get_layer_mut(&mut self, name: &str) -> &mut Layer {
        let index = self.layer_index(name);
        &mut self.layers[index]
    }

    /// Keras' `model.layers`.
    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Keras' `model.count_params()`.
    pub fn count_params(&self) -> usize {
        self.layers.iter().map(Layer::count_params).sum()
    }

    /// Keras' `model.summary()`: prints every layer with its output shape and
    /// parameter count. `to_string()` gives the same table as a String.
    pub fn summary(&self) {
        print!("{self}");
    }

    /// One gradient step on one batch. Returns the batch's loss before the step.
    fn train_step(&mut self, x: &Matrix, y: &Matrix) -> f64 {
        let (loss, gradients) = self.loss_and_gradients(x, y);

        let optimizer = self.optimizer.as_mut().expect("compile sets the optimizer with the loss");
        let mut variables: Vec<&mut Matrix> = self.layers.iter_mut()
            .flat_map(|layer| layer.variables_mut())
            .collect();
        optimizer.apply_gradients(&mut variables, &gradients);

        loss
    }

    /// Forward propagation, then backpropagation, in Ng's notation (l = 1..L):
    ///
    /// ```text
    /// A[0] = X          Z[l] = A[l-1]·W[l] + b[l]          A[l] = g(Z[l])
    /// dZ[L] comes from the loss
    /// dW[l] = A[l-1]ᵀ·dZ[l]                db[l] = dZ[l] summed over the samples
    /// dA[l-1] = dZ[l]·W[l]ᵀ                dZ[l-1] = dA[l-1] * g'(Z[l-1])
    /// ```
    ///
    /// Returns J and the gradients in the order dW[1], db[1], dW[2], db[2], ...
    fn loss_and_gradients(&self, x: &Matrix, y: &Matrix) -> (f64, Vec<Matrix>) {
        let loss = self.compiled_loss("fit");

        // z_matrices[i] and a_matrices[i] belong to self.layers[i].
        let mut z_matrices: Vec<Matrix> = Vec::with_capacity(self.layers.len());
        let mut a_matrices: Vec<Matrix> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let z_matrix = layer.z_matrix(a_matrices.last().unwrap_or(x));
            a_matrices.push(layer.get_activation_function().a_with_matrix(&z_matrix));
            z_matrices.push(z_matrix);
        }

        let last_index = self.layers.len() - 1;
        let y_pred = &a_matrices[last_index];
        let cost = loss.cost(y, y_pred);

        let mut dz_matrix = loss.output_gradient(
            y, y_pred, &z_matrices[last_index], self.layers[last_index].get_activation_function()
        );
        let mut gradients: Vec<Matrix> = Vec::with_capacity(2 * self.layers.len());
        for index in (0..self.layers.len()).rev() {
            let a_in = if index == 0 { x } else { &a_matrices[index - 1] };
            // Pushed as db, dW so that the reverse below gives dW, db.
            gradients.push(dz_matrix.column_sums());
            gradients.push(a_in.transpose_matmul(&dz_matrix).unwrap());

            if index > 0 {
                let (weight_matrix, _) = self.layers[index].get_weights();
                let da_in = dz_matrix.matmul_transpose(weight_matrix).unwrap();
                dz_matrix = self.layers[index - 1].get_activation_function().dz_matrix(&da_in, &z_matrices[index - 1]);
            }
        }
        gradients.reverse();

        (cost, gradients)
    }

    /// Keras' naming: "dense", "dense_1", "dense_2", ...
    fn unused_default_name(&self) -> String {
        (0_usize..)
            .map(|index| if index == 0 { String::from("dense") } else { format!("dense_{index}") })
            .find(|name| self.layers.iter().all(|layer| layer.get_name() != name))
            .unwrap()
    }

    fn layer_index(&self, name: &str) -> usize {
        match self.layers.iter().position(|layer| layer.get_name() == name) {
            Some(index) => index,
            None => {
                let names: Vec<&str> = self.layers.iter().map(Layer::get_name).collect();
                panic!("There is no layer named '{name}', the layers are {names:?} !!")
            }
        }
    }

    fn compiled_loss(&self, caller: &str) -> &dyn Loss {
        match &self.loss {
            Some(loss) => loss.as_ref(),
            None => panic!("Call compile(loss, optimizer) before {caller} !!"),
        }
    }

    fn validate_features(&self, x: &Matrix) {
        if self.layers.is_empty() {
            panic!("The model has no layers, add a Dense layer first !!");
        }
        if x.col_count() != self.input.get_feature_count() {
            panic!("x has {} features per sample but the model's Input has {} !!",
                   x.col_count(), self.input.get_feature_count()
            );
        }
    }

    fn validate_samples(&self, x: &Matrix, y: &Matrix) {
        self.validate_features(x);

        if x.row_count() == 0 {
            panic!("x has no samples !!");
        }
        if y.row_count() != x.row_count() {
            panic!("x has {} samples but y has {} !!", x.row_count(), y.row_count());
        }
        let output_units = self.layers[self.layers.len() - 1].get_units();
        if y.col_count() != output_units {
            panic!("y has {} columns but the last layer has {} units !!", y.col_count(), output_units);
        }
    }
}

impl fmt::Display for Sequential {
    /// The table Keras' `summary()` prints, plus every layer's activation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = ["Layer (type)", "Output Shape", "Activation", "Param #"].map(String::from);
        let rows: Vec<[String; 4]> = self.layers.iter().map(|layer| [
            format!("{} (Dense)", layer.get_name()),
            format!("(None, {})", layer.get_units()),
            layer.get_activation_function().to_string(),
            layer.count_params().to_string(),
        ]).collect();

        let mut widths = [0; 4];
        for row in std::iter::once(&header).chain(&rows) {
            for (width, cell) in widths.iter_mut().zip(row) {
                *width = (*width).max(cell.len() + 4);
            }
        }
        let line_width: usize = widths.iter().sum();
        let format_row = |row: &[String; 4]| {
            let line: String = row.iter()
                .zip(widths)
                .map(|(cell, width)| format!(" {cell:<pad$}", pad = width - 1))
                .collect();
            line.trim_end().to_string()
        };

        writeln!(f, "Model: \"sequential\"")?;
        writeln!(f, "{}", "_".repeat(line_width))?;
        writeln!(f, "{}", format_row(&header))?;
        writeln!(f, "{}", "=".repeat(line_width))?;
        for row in &rows {
            writeln!(f, "{}", format_row(row))?;
        }
        writeln!(f, "{}", "=".repeat(line_width))?;
        writeln!(f, "Total params: {}", self.count_params())?;
        writeln!(f, "{}", "_".repeat(line_width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::loss::{BinaryCrossentropy, MeanSquaredError};
    use crate::optimizer::SGD;

    fn matrix(row_count: usize, col_count: usize, datas: &[f64]) -> Matrix {
        Matrix::from_vec(row_count, col_count, datas.to_vec()).unwrap()
    }

    /// 5 samples x 3 features.
    fn features() -> Matrix {
        matrix(5, 3, &[0.2, -0.4, 0.9, 1.1, 0.3, -0.7, -0.5, 0.8, 0.1, 0.6, -1.2, 0.4, -0.3, -0.9, 1.5])
    }

    fn two_layer_model() -> Sequential {
        Sequential::new(Input::new(3), vec![
            Dense::new(4, Activation::ReLU).name("hidden"),
            Dense::new(2, Activation::Sigmoid),
        ])
    }

    #[test]
    fn layers_are_chained_by_shape() {
        let model = two_layer_model();
        let shapes: Vec<(usize, usize)> = model.get_layers().iter()
            .map(|layer| layer.get_weights().0.shape())
            .collect();

        assert_eq!(shapes, vec![(3, 4), (4, 2)]);
        assert_eq!(model.count_params(), (3 * 4 + 4) + (4 * 2 + 2));
    }

    #[test]
    fn add_names_layers_like_keras() {
        let mut model = Sequential::new(Input::new(2), vec![Dense::new(3, Activation::ReLU)]);
        model.add(Dense::new(1, Activation::Sigmoid));

        let names: Vec<&str> = model.get_layers().iter().map(Layer::get_name).collect();
        assert_eq!(names, vec!["dense", "dense_1"]);
        assert_eq!(model.get_layer("dense_1").get_weights().0.shape(), (3, 1));
    }

    #[test]
    #[should_panic(expected = "'L1' is used twice")]
    fn layer_names_are_unique() {
        Sequential::new(Input::new(2), vec![
            Dense::new(3, Activation::ReLU).name("L1"),
            Dense::new(1, Activation::Sigmoid).name("L1"),
        ]);
    }

    #[test]
    #[should_panic(expected = "no layer named 'missing', the layers are [\"hidden\", \"dense\"]")]
    fn get_layer_lists_the_names() {
        two_layer_model().get_layer("missing");
    }

    #[test]
    fn summary_lists_every_layer() {
        let summary = two_layer_model().to_string();
        assert!(summary.contains("hidden (Dense)"));
        assert!(summary.contains("(None, 4)"));
        assert!(summary.contains("ReLU"));
        assert!(summary.contains("Total params: 26"));
    }

    #[test]
    fn predict_gives_one_row_per_sample() {
        assert_eq!(two_layer_model().predict(&features()).shape(), (5, 2));
    }

    #[test]
    #[should_panic(expected = "x has 2 features per sample but the model's Input has 3")]
    fn predict_checks_the_feature_count() {
        two_layer_model().predict(&matrix(1, 2, &[0., 0.]));
    }

    #[test]
    #[should_panic(expected = "Call compile(loss, optimizer) before fit")]
    fn fit_needs_compile() {
        let mut model = two_layer_model();
        model.fit(&matrix(1, 3, &[0.; 3]), &matrix(1, 2, &[0.; 2]), FitOptions::default());
    }

    #[test]
    #[should_panic(expected = "y has 1 columns but the last layer has 2 units")]
    fn fit_checks_the_target_shape() {
        let mut model = two_layer_model();
        model.compile(BinaryCrossentropy, SGD::default());
        model.fit(&matrix(2, 3, &[0.; 6]), &matrix(2, 1, &[0.; 2]), FitOptions::default());
    }

    /// Adds `delta` to one element of a layer's W (variable 0) or b (variable 1).
    fn nudge(model: &mut Sequential, layer_index: usize, variable_index: usize, element: usize, delta: f64) {
        let name = model.get_layers()[layer_index].get_name().to_string();
        let layer = model.get_layer_mut(&name);

        let (w, b) = layer.get_weights();
        let mut variables = [w.clone(), b.clone()];
        let (row_count, col_count) = variables[variable_index].shape();
        let mut datas = variables[variable_index].as_slice().to_vec();
        datas[element] += delta;
        variables[variable_index] = Matrix::from_vec(row_count, col_count, datas).unwrap();

        let [w, b] = variables;
        layer.set_weights(w, b);
    }

    /// Ng's gradient checking: every backprop gradient against the slope
    /// (J(θ + h) - J(θ - h)) / 2h, one parameter at a time.
    fn assert_gradients_match_numerical_slopes(model: &mut Sequential, x: &Matrix, y: &Matrix) {
        let (_, gradients) = model.loss_and_gradients(x, y);
        let h = 1e-5;

        for (index, gradient) in gradients.iter().enumerate() {
            let (layer_index, variable_index) = (index / 2, index % 2);
            for element in 0..gradient.as_slice().len() {
                nudge(model, layer_index, variable_index, element, h);
                let cost_plus = model.evaluate(x, y);
                nudge(model, layer_index, variable_index, element, -2. * h);
                let cost_minus = model.evaluate(x, y);
                nudge(model, layer_index, variable_index, element, h);

                let numerical = (cost_plus - cost_minus) / (2. * h);
                let backprop = gradient.as_slice()[element];
                assert!(
                    (backprop - numerical).abs() <= 1e-7 + 1e-5 * numerical.abs(),
                    "layer {layer_index}, variable {variable_index}, element {element}: \
                     backprop {backprop} vs numerical {numerical}"
                );
            }
        }
    }

    #[test]
    fn backprop_passes_gradient_checking_with_binary_crossentropy() {
        let mut model = Sequential::new(Input::new(3), vec![
            Dense::new(4, Activation::ReLU),
            Dense::new(3, Activation::Sigmoid),
            Dense::new(2, Activation::Sigmoid),
        ]);
        model.compile(BinaryCrossentropy, SGD::default());

        let y = matrix(5, 2, &[1., 0., 0., 1., 1., 1., 0., 0., 1., 0.]);
        assert_gradients_match_numerical_slopes(&mut model, &features(), &y);
    }

    #[test]
    fn backprop_passes_gradient_checking_with_mean_squared_error() {
        let mut model = Sequential::new(Input::new(3), vec![
            Dense::new(4, Activation::Sigmoid),
            Dense::new(3, Activation::ReLU),
            Dense::new(1, Activation::Linear),
        ]);
        model.compile(MeanSquaredError, SGD::default());

        let y = matrix(5, 1, &[0.5, -1.2, 2., 0.3, -0.7]);
        assert_gradients_match_numerical_slopes(&mut model, &features(), &y);
    }
}
