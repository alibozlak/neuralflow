

### Neural Network util lib for my AI projects. (Continue to development)

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
| `model.export("model.onnx", format="onnx")` | `model.export_onnx("model.onnx")?` |

- Activations: `Sigmoid`, `Linear`, `ReLU`
- Losses: `BinaryCrossentropy`, `MeanSquaredError` (Keras' definition, without Ng's 1/2) [Source: 3rd line](https://github.com/alibozlak/multivariable_linear_regression/blob/master/math/001_dJ_daj_partial_derivative.pdf)
- Optimizers: `SGD`, `Adam`

### Scaling the data: `column_based_scaling`

Gradient descent has a hard time when one feature is counted in thousands and another
one in tenths. [`column_based_scaling`](src/column_based_scaling.rs) divides every
column by a power of ten, so the numbers come closer together:

```rust
use neuralflow::column_based_scaling::manipulate_datas_between_0_and_10;

// x is (samples x features) and y is (samples x 1).
let (scaled_x, scaled_y, ten_power_ratios) = manipulate_datas_between_0_and_10(x, y);
```

This is not Keras' `Normalization` layer, and it is worth knowing what it really does:

- A column's power of ten comes from that column's **first row** only. A first value with
  4 integer digits gives `10^3`, so the whole column is divided by 1000. The other rows
  are never looked at, so a column whose first sample is much smaller or much bigger than
  the rest gets the wrong scale.
- `y` must have exactly one column, and both matrices are taken by value.
- There is no inverse function. `ten_power_ratios` holds the exponent of every feature
  column, and its last entry the exponent of `y`. So a prediction goes back to the
  original scale by multiplying it with `10^` that last exponent.

### Export to ONNX

A trained model can be written as an [ONNX](https://onnx.ai) file. The file holds the
graph and every weight, so the model runs without this crate and without Rust:

```rust
model.export_onnx("xor.onnx")?;     // model.to_onnx_bytes() gives the same bytes
```

Every runtime reads ONNX: `onnxruntime` in Python or C++,
[`tract`](https://crates.io/crates/tract) in Rust, a browser through WebAssembly, a
phone, a microcontroller.

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("xor.onnx")
print(session.run(None, {"input": np.array([[0., 1.]], dtype=np.float32)}))
```

Every `Dense` becomes one `Gemm` node, ONNX' GEneral Matrix Multiply. It computes
`alpha·A·B + beta·C`, which with `alpha = beta = 1` and no transposes is exactly
`Z = A_in·W + b`, so `W` and `b` are written the way `Layer` already stores them. The
activation becomes a second node, except for `Linear`: `A = Z` is the Gemm's own output,
so there is nothing to add. The graph's input is named `input` and its output `output`,
and the sample count is a named dimension, so one file serves a caller predicting one
sample and a caller predicting a thousand.

**The good side**

- It needs no new dependency. An ONNX file is protobuf, whose wire format is a list of
  (tag, value) pairs, and the writer for it is about fifty lines in
  [`src/onnx.rs`](src/onnx.rs).
- The file is checked. `onnx.checker` accepts it, and `onnxruntime` gives the same
  predictions as `model.predict`.

**The bad side**

- The weights are written as **f32**, not f64. Every runtime supports f32, while some
  have no f64 kernel at all. So the weights are rounded, and predictions move by about
  `1e-7`.
- It only writes. Reading an ONNX file back into a `Sequential` is not implemented.
- It covers what the library has: `Dense` with `Sigmoid`, `Linear` or `ReLU`. There is
  no `Softmax`, no convolution and no recurrent layer yet.

Run `cargo run --example export_onnx` to train the XOR model and write `xor.onnx`.

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
