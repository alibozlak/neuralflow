

### Neural Network util lib for Rust. (Continue to development)

**Guide: Andrew Ng's courses**

The math follows Ng's notation (`Z = A·W + b`, `A = g(Z)`, `dZ`, `dW`, `db`), the API follows Keras.

```rust
use neuralflow::prelude::*;

// Every row is one sample, every column one feature: the same layout as Keras.
let x = Matrix::from_vec(4, 2, vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
let y = Matrix::from_vec(4, 1, vec![0., 1., 1., 0.]).unwrap();

let mut model = Sequential::new(Input::new(2), vec![
    Dense::new(8, Activation::ReLU).name("layer1"),
    Dense::new(1, Activation::Sigmoid).name("layer2"),
]);
model.summary();

model.compile(BinaryCrossentropy, Adam::new(0.05));
let history = model.fit(&x, &y, FitOptions { epochs: 500, ..Default::default() });
let predictions = model.predict(&x);
```

Run the full example with `cargo run --example xor`.

| Keras | neuralflow |
| --- | --- |
| `Sequential([Input(shape=(2,)), Dense(8, activation='relu', name='layer1')])` | `Sequential::new(Input::new(2), vec![Dense::new(8, Activation::ReLU).name("layer1")])` |
| `model.add(Dense(1, activation='sigmoid'))` | `model.add(Dense::new(1, Activation::Sigmoid))` |
| `model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.05))` | `model.compile(BinaryCrossentropy, Adam::new(0.05))` |
| `history = model.fit(X, y, epochs=500, batch_size=32)` | `let history = model.fit(&x, &y, FitOptions { epochs: 500, batch_size: 32, ..Default::default() })` |
| `history.history["loss"]` | `history.loss` |
| `model.predict(X)`, `model.evaluate(X, y)` | `model.predict(&x)`, `model.evaluate(&x, &y)` |
| `model.summary()` | `model.summary()` |
| `W, b = model.get_layer("layer1").get_weights()` | `let (w, b) = model.get_layer("layer1").get_weights();` |
| `model.get_layer("layer1").set_weights([W, b])` | `model.get_layer_mut("layer1").set_weights(w, b)` |
| `tf.random.set_seed(1234)` | `set_random_seed(1234)` |

- Activations: `Sigmoid`, `Linear`, `ReLU`
- Losses: `BinaryCrossentropy`, `MeanSquaredError` (Keras' definition, without Ng's 1/2)
- Optimizers: `SGD`, `Adam`
