//! Trains the XOR model, exports it and prints what neuralflow predicts,
//! so onnxruntime can be asked for the same numbers.
//!
//! cargo run --example export_onnx

use neuralflow::prelude::*;

fn main() {
    let x = Matrix::from_vec(4, 2, vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let y = Matrix::from_vec(4, 1, vec![0., 1., 1., 0.]).unwrap();

    set_random_seed(1234);
    let mut model = Sequential::new(Input::new(2), vec![
        Dense::new(8, Activation::ReLU).name("layer1"),
        Dense::new(1, Activation::Sigmoid).name("layer2"),
    ]);
    model.compile(BinaryCrossentropy, Adam::new(0.05));
    model.fit(&x, &y, FitOptions { epochs: 500, verbose: false, ..Default::default() });

    model.export_onnx("xor.onnx").unwrap();

    let predictions = model.predict(&x);
    for sample in 0..4 {
        println!("{:.17}", predictions.get(sample, 0).unwrap());
    }
}
