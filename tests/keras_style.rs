//! The library used the way the README shows it: Keras-style, through the prelude.

use neuralflow::prelude::*;

fn matrix(row_count: usize, col_count: usize, datas: &[f64]) -> Matrix {
    Matrix::from_vec(row_count, col_count, datas.to_vec()).unwrap()
}

fn quiet(epochs: usize) -> FitOptions {
    FitOptions { epochs, verbose: false, ..Default::default() }
}

/// The data set of Ng's logistic regression lab.
fn logistic_data() -> (Matrix, Matrix) {
    let x = matrix(6, 2, &[0.5, 1.5, 1., 1., 1.5, 0.5, 3., 0.5, 2., 2., 1., 2.5]);
    let y = matrix(6, 1, &[0., 0., 0., 1., 1., 1.]);
    (x, y)
}

/// y = 2x + 1 with one linear unit and plain gradient descent: the weights
/// should end up at w = 2, b = 1.
#[test]
fn linear_regression_finds_the_line() {
    let x = matrix(5, 1, &[-2., -1., 0., 1., 2.]);
    let y = matrix(5, 1, &[-3., -1., 1., 3., 5.]);

    let mut model = Sequential::new(Input::new(1), vec![Dense::new(1, Activation::Linear)]);
    model.compile(MeanSquaredError, SGD::new(0.1));
    model.fit(&x, &y, quiet(200));

    let (w, b) = model.get_layer("dense").get_weights();
    assert!((w.get(0, 0).unwrap() - 2.).abs() < 1e-3, "w = {w:?}");
    assert!((b.get(0, 0).unwrap() - 1.).abs() < 1e-3, "b = {b:?}");
}

/// XOR: one sigmoid unit can't separate it, one hidden layer can.
#[test]
fn xor_is_learned_with_a_hidden_layer() {
    let x = matrix(4, 2, &[0., 0., 0., 1., 1., 0., 1., 1.]);
    let y = matrix(4, 1, &[0., 1., 1., 0.]);

    set_random_seed(1234);
    let mut model = Sequential::new(Input::new(2), vec![
        Dense::new(8, Activation::ReLU).name("layer1"),
        Dense::new(1, Activation::Sigmoid).name("layer2"),
    ]);
    model.compile(BinaryCrossentropy, Adam::new(0.05));
    let history = model.fit(&x, &y, quiet(500));

    assert!(*history.loss.last().unwrap() < 0.05, "loss = {:?}", history.loss.last());
    let predictions = model.predict(&x);
    for sample in 0..4 {
        let predicted_one = predictions.get(sample, 0).unwrap() >= 0.5;
        assert_eq!(predicted_one, y.get(sample, 0).unwrap() == 1., "sample {sample}");
    }
}

/// With the whole data set as one batch, an epoch's loss is measured just
/// before its step, so the first one is what evaluate() gave before fit.
#[test]
fn fit_reports_the_loss_evaluate_measures() {
    let (x, y) = logistic_data();
    let mut model = Sequential::new(Input::new(2), vec![Dense::new(1, Activation::Sigmoid)]);
    model.compile(BinaryCrossentropy, SGD::new(0.5));

    let before = model.evaluate(&x, &y);
    let history = model.fit(&x, &y, FitOptions { epochs: 30, shuffle: false, verbose: false, ..Default::default() });

    assert_eq!(history.loss.len(), 30);
    assert_eq!(history.loss[0], before);
    assert!(history.loss.windows(2).all(|pair| pair[1] < pair[0]), "loss must fall every epoch: {:?}", history.loss);
    assert!(model.evaluate(&x, &y) < history.loss[29]);
}

#[test]
fn the_same_seed_trains_the_same_model() {
    let (x, y) = logistic_data();
    let train = || {
        set_random_seed(7);
        let mut model = Sequential::new(Input::new(2), vec![
            Dense::new(3, Activation::ReLU),
            Dense::new(1, Activation::Sigmoid),
        ]);
        model.compile(BinaryCrossentropy, Adam::default());
        model.fit(&x, &y, FitOptions { epochs: 5, batch_size: 2, verbose: false, ..Default::default() }).loss
    };

    assert_eq!(train(), train());
}

/// Ng's labs copy known weights in with set_weights and check the output.
#[test]
fn set_weights_then_predict() {
    let mut model = Sequential::new(Input::new(2), vec![Dense::new(1, Activation::Sigmoid).name("L1")]);
    model.get_layer_mut("L1").set_weights(matrix(2, 1, &[1., -1.]), matrix(1, 1, &[0.5]));

    let a = model.predict(&matrix(1, 2, &[2., 1.])).get(0, 0).unwrap();
    assert!((a - Activation::Sigmoid.a(1.5)).abs() < 1e-12);
}
