//! XOR, written the way Keras would write it.
//!
//! cargo run --example xor

use neuralflow::prelude::*;

fn main() {
    // Every row is one sample: (x1, x2) -> x1 XOR x2
    let x = Matrix::from_vec(4, 2, vec![
        0., 0.,
        0., 1.,
        1., 0.,
        1., 1.,
    ]).unwrap();
    let y = Matrix::from_vec(4, 1, vec![0., 1., 1., 0.]).unwrap();

    set_random_seed(1234);
    let mut model = Sequential::new(Input::new(2), vec![
        Dense::new(8, Activation::ReLU).name("layer1"),
        Dense::new(1, Activation::Sigmoid).name("layer2"),
    ]);
    model.summary();

    model.compile(BinaryCrossentropy, Adam::new(0.05));
    let history = model.fit(&x, &y, FitOptions { epochs: 500, verbose: false, ..Default::default() });
    println!("loss: {:.4} -> {:.4}", history.loss[0], history.loss[history.loss.len() - 1]);

    let predictions = model.predict(&x);
    for sample in 0..4 {
        println!(
            "{} XOR {} -> {:.3}",
            x.get(sample, 0).unwrap(), x.get(sample, 1).unwrap(), predictions.get(sample, 0).unwrap()
        );
    }

    let (w1, b1) = model.get_layer("layer1").get_weights();
    println!("W1 is {:?}, b1 is {:?}", w1.shape(), b1.shape());
}
