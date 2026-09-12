

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
- Losses: `BinaryCrossentropy`, `MeanSquaredError` (Keras' definition, without Ng's 1/2) [Source](https://github.com/alibozlak/multivariable_linear_regression/blob/master/math/001_dJ_daj_partial_derivative.pdf)
- Optimizers: `SGD`, `Adam`

### Matrix engine: `safe_matmul`

This project uses [`safe_matmul`](https://crates.io/crates/safe_matmul) for all matrix
work. Every matrix here is a `safe_matmul` `Matrix`: the samples `x`, the targets `y`,
and the weights and biases of each layer. You do not need to add it to your own
`Cargo.toml`, because neuralflow re-exports it.

**The good side**

- It is safe. The crate uses `#![forbid(unsafe_code)]`, so it has no `unsafe` code.
- It has zero dependencies. There is no BLAS, no C compiler and no system library. So
  builds are fast, and they also work offline.
- It gives the three products that backpropagation needs: `A·B` for the forward pass,
  `Aᵀ·B` for `dW`, and `A·Bᵀ` for the gradient of the input. It does not make a
  transposed copy for the last two, so it saves time and memory.
- A wrong shape returns a `Result` with an error. The program does not panic.

**The bad side**

- It is slower than a SIMD library like `matrixmultiply`. It is about 2x slower with
  small matrices and about 4x slower with big ones.
- It uses only one thread, and only `f64` numbers. There is no `f32` and no GPU.
- It has no cache blocking, so big matrices become slow.
- It is a small library. It has no broadcasting and no element-wise math, so neuralflow
  writes its own helpers in [`src/matrix_ops.rs`](src/matrix_ops.rs).
- The API is still young (version 0.1.0) and it can change.

So neuralflow is good for learning and for small or medium models. For big and fast
training, a library with BLAS or a GPU is a better choice.

You can make `safe_matmul` faster on your own machine:

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]
```
