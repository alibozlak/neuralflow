//! Keras' `model.export("model.onnx", format="onnx")`: a trained `Sequential`
//! as an ONNX file, so a model trained here can run anywhere.
//!
//! An ONNX file is a protobuf message. Protobuf's wire format is a flat list
//! of (tag, value) pairs, where the tag packs a field number from `onnx.proto`
//! together with a wire type saying how to read the value. Nothing else is
//! needed to write one, so `Message` below does it by hand instead of pulling
//! in a protobuf crate.
//!
//! The layers turn into ONNX's own operators: every `Dense` becomes a `Gemm`
//! (GEneral Matrix Multiply, `alpha·A·B + beta·C`) plus one node for its
//! activation, which is exactly `Z = A_in·W + b` and `A = g(Z)`.
//!
//! Checking a file needs no Rust at all:
//!
//! ```text
//! import numpy as np, onnx, onnxruntime as ort
//! onnx.checker.check_model(onnx.load("xor.onnx"))
//! session = ort.InferenceSession("xor.onnx")
//! print(session.run(None, {"input": np.array([[0., 1.]], dtype=np.float32)}))
//! ```

use std::fs;
use std::io;
use std::path::Path;
use safe_matmul::matrix::Matrix;
use crate::activation::Activation;
use crate::layer::Layer;
use crate::sequential_model::Sequential;

/// ONNX IR version 7 (ONNX 1.8): old enough that every runtime still reads it.
const IR_VERSION: i64 = 7;

/// The operator set version Gemm, Relu and Sigmoid are taken from.
const OPSET_VERSION: i64 = 13;

/// `TensorProto.DataType.FLOAT`, i.e. f32.
const FLOAT: i64 = 1;

/// `AttributeProto.AttributeType.FLOAT` and `.INT`.
const ATTRIBUTE_FLOAT: i64 = 1;
const ATTRIBUTE_INT: i64 = 2;

/// The graph's only input, Keras' `model.input`: one row per sample.
const INPUT_NAME: &str = "input";

/// The graph's only output, Keras' `model.output`.
const OUTPUT_NAME: &str = "output";

impl Sequential {
    /// Keras' `model.export(path, format="onnx")`. The file holds the graph
    /// and every weight, so it needs neither this crate nor Rust to run.
    pub fn export_onnx(&self, path: impl AsRef<Path>) -> io::Result<()> {
        fs::write(path, self.to_onnx_bytes())
    }

    /// The bytes `export_onnx` would write.
    pub fn to_onnx_bytes(&self) -> Vec<u8> {
        let layers = self.get_layers();
        if layers.is_empty() {
            panic!("The model has no layers, there is nothing to export !!");
        }

        let mut model = Message::default();
        model.int(1, IR_VERSION);                                 // ir_version
        model.text(2, "neuralflow");                              // producer_name
        model.text(3, env!("CARGO_PKG_VERSION"));                 // producer_version
        model.message(7, |graph| write_graph(graph, layers));     // graph
        model.message(8, |operator_set| {                         // opset_import
            operator_set.text(1, "");                             //   domain: "" is ai.onnx
            operator_set.int(2, OPSET_VERSION);                   //   version
        });

        model.bytes
    }
}

/// `GraphProto`: the nodes in the order they run, the weights they read, and
/// the shapes a caller has to respect.
fn write_graph(graph: &mut Message, layers: &[Layer]) {
    for (index, layer) in layers.iter().enumerate() {
        let a_in = match index {
            0 => String::from(INPUT_NAME),
            _ => output_name(layers, index - 1),
        };
        write_layer(graph, layer, &a_in, &output_name(layers, index));   // node
    }

    graph.text(2, "sequential");                                        // name

    for layer in layers {
        let (weight_matrix, bias_matrix) = layer.get_weights();
        let name = layer.get_name();
        // initializer: W is (inputs x units), b is one value per unit.
        write_initializer(graph, &format!("{name}.weight"),
                          &[weight_matrix.row_count(), weight_matrix.col_count()], weight_matrix);
        write_initializer(graph, &format!("{name}.bias"),
                          &[bias_matrix.col_count()], bias_matrix);
    }

    // Input has no accessor, but a sample has as many features as the first
    // layer has weight rows, which is the same number.
    let feature_count = layers[0].get_weights().0.row_count();
    let unit_count = layers[layers.len() - 1].get_units();
    graph.message(11, |value| write_value_info(value, INPUT_NAME, feature_count));   // input
    graph.message(12, |value| write_value_info(value, OUTPUT_NAME, unit_count));     // output
}

/// One layer as ONNX nodes: a Gemm for `Z = A_in·W + b`, then `A = g(Z)`.
fn write_layer(graph: &mut Message, layer: &Layer, a_in: &str, a_out: &str) {
    let name = layer.get_name();
    let operator = onnx_activation(layer.get_activation_function());
    // A Linear layer has no second node, so its Gemm writes the layer's output.
    let z_out = match operator {
        Some(_) => format!("{name}_z"),
        None => String::from(a_out),
    };

    graph.message(1, |node| {
        node.text(1, a_in);                             // input: A
        node.text(1, &format!("{name}.weight"));        // input: B
        node.text(1, &format!("{name}.bias"));          // input: C
        node.text(2, &z_out);                           // output
        node.text(3, &format!("{name}_Gemm"));          // name
        node.text(4, "Gemm");                           // op_type
        // alpha·A·B + beta·C with no scaling and no transposes is exactly what
        // Layer computes, because W already sits as (inputs x units).
        write_float_attribute(node, "alpha", 1.);
        write_float_attribute(node, "beta", 1.);
        write_int_attribute(node, "transA", 0);
        write_int_attribute(node, "transB", 0);
    });

    if let Some(operator) = operator {
        graph.message(1, |node| {
            node.text(1, &z_out);                       // input
            node.text(2, a_out);                        // output
            node.text(3, &format!("{name}_{operator}"));// name
            node.text(4, operator);                     // op_type
        });
    }
}

/// The ONNX operator for `g`. Linear gets none: ONNX has no identity
/// activation and does not need one, since A = Z is the Gemm's own output.
fn onnx_activation(activation: Activation) -> Option<&'static str> {
    match activation {
        // ONNX spells it Relu, not ReLU.
        Activation::ReLU => Some("Relu"),
        Activation::Sigmoid => Some("Sigmoid"),
        Activation::Linear => None,
    }
}

/// The tensor a layer's output flows out on. The last layer's is the graph's
/// output, so a caller can always read "output".
fn output_name(layers: &[Layer], index: usize) -> String {
    match index == layers.len() - 1 {
        true => String::from(OUTPUT_NAME),
        // A = g(Z) in Ng's notation; for a Linear layer the two are the same.
        false => format!("{}_a", layers[index].get_name()),
    }
}

/// `TensorProto`: a weight matrix as one of the graph's constants.
fn write_initializer(graph: &mut Message, name: &str, dims: &[usize], values: &Matrix) {
    graph.message(5, |tensor| {
        for &dim in dims {
            tensor.int(1, dim as i64);          // dims
        }
        tensor.int(2, FLOAT);                   // data_type
        tensor.text(8, name);                   // name
        tensor.raw(9, &raw_data(values));       // raw_data
    });
}

/// What ONNX calls raw_data: the matrix row by row, as little-endian f32.
/// Every runtime reads f32, and its seven significant digits are far more
/// than inference needs, so the f64 weights are rounded here.
fn raw_data(matrix: &Matrix) -> Vec<u8> {
    matrix.as_slice().iter()
        .flat_map(|&value| (value as f32).to_le_bytes())
        .collect()
}

/// `ValueInfoProto`: a name, an element type and a shape. The row count is the
/// named dimension "batch", so the file does not fix how many samples a caller
/// may pass in one call.
fn write_value_info(value: &mut Message, name: &str, col_count: usize) {
    value.text(1, name);                                            // name
    value.message(2, |type_proto| {                                 // type
        type_proto.message(1, |tensor| {                            // tensor_type
            tensor.int(1, FLOAT);                                   // elem_type
            tensor.message(2, |shape| {                             // shape
                shape.message(1, |dim| dim.text(2, "batch"));       // dim: dim_param
                shape.message(1, |dim| dim.int(1, col_count as i64));// dim: dim_value
            });
        });
    });
}

/// `AttributeProto` holds one value per possible type and a `type` field
/// saying which one was filled in.
fn write_float_attribute(node: &mut Message, name: &str, value: f32) {
    node.message(5, |attribute| {
        attribute.text(1, name);                    // name
        attribute.float(2, value);                  // f
        attribute.int(20, ATTRIBUTE_FLOAT);         // type
    });
}

fn write_int_attribute(node: &mut Message, name: &str, value: i64) {
    node.message(5, |attribute| {
        attribute.text(1, name);                    // name
        attribute.int(3, value);                    // i
        attribute.int(20, ATTRIBUTE_INT);           // type
    });
}

/// Wire types: the three the writer above emits.
const VARINT: u32 = 0;
const LENGTH_DELIMITED: u32 = 2;
const FIXED32: u32 = 5;

/// A protobuf message being built. Every method appends one field, so the
/// writers above read like the `onnx.proto` definitions they follow.
#[derive(Default)]
struct Message {
    bytes: Vec<u8>,
}

impl Message {
    /// Base 128, seven bits per byte, low bits first; a set high bit means
    /// another byte follows.
    fn varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.bytes.push(value as u8 | 0x80);
            value >>= 7;
        }
        self.bytes.push(value as u8);
    }

    /// The field number and how to read the value that follows it.
    fn tag(&mut self, field: u32, wire_type: u32) {
        self.varint(((field << 3) | wire_type) as u64);
    }

    fn int(&mut self, field: u32, value: i64) {
        self.tag(field, VARINT);
        self.varint(value as u64);
    }

    fn float(&mut self, field: u32, value: f32) {
        self.tag(field, FIXED32);
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn text(&mut self, field: u32, value: &str) {
        self.raw(field, value.as_bytes());
    }

    /// Bytes, a string or a nested message: a length, then that many bytes.
    fn raw(&mut self, field: u32, value: &[u8]) {
        self.tag(field, LENGTH_DELIMITED);
        self.varint(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    /// A nested message has to be built before it is appended, because its
    /// length is written in front of it.
    fn message(&mut self, field: u32, write: impl FnOnce(&mut Message)) {
        let mut nested = Message::default();
        write(&mut nested);
        self.raw(field, &nested.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Dense, Input};

    fn matrix(row_count: usize, col_count: usize, datas: &[f64]) -> Matrix {
        Matrix::from_vec(row_count, col_count, datas.to_vec()).unwrap()
    }

    /// Two layers with weights set by hand, so the bytes are predictable.
    fn model() -> Sequential {
        let mut model = Sequential::new(Input::new(2), vec![
            Dense::new(3, Activation::ReLU).name("hidden"),
            Dense::new(1, Activation::Sigmoid).name("out"),
        ]);
        model.get_layer_mut("hidden")
            .set_weights(matrix(2, 3, &[1., 2., 3., 4., 5., 6.]), matrix(1, 3, &[-1., 0., 1.]));
        model.get_layer_mut("out")
            .set_weights(matrix(3, 1, &[0.5, -0.5, 2.]), matrix(1, 1, &[0.25]));

        model
    }

    /// One protobuf value: enough of the wire format to read back what the
    /// writer produced, so the tests check the real fields and not just bytes.
    #[derive(Debug, PartialEq)]
    enum Value {
        Varint(u64),
        Fixed32([u8; 4]),
        Bytes(Vec<u8>),
    }

    /// Every (field number, value) pair of a message, in the written order.
    fn read_fields(mut bytes: &[u8]) -> Vec<(u32, Value)> {
        let mut fields = Vec::new();
        while !bytes.is_empty() {
            let tag = read_varint(&mut bytes);
            let (field, wire_type) = ((tag >> 3) as u32, (tag & 7) as u32);
            let value = match wire_type {
                VARINT => Value::Varint(read_varint(&mut bytes)),
                FIXED32 => Value::Fixed32(take(&mut bytes, 4).try_into().unwrap()),
                LENGTH_DELIMITED => {
                    let length = read_varint(&mut bytes) as usize;
                    Value::Bytes(take(&mut bytes, length).to_vec())
                }
                other => panic!("the writer never emits wire type {other} !!"),
            };
            fields.push((field, value));
        }

        fields
    }

    fn read_varint(bytes: &mut &[u8]) -> u64 {
        let mut value = 0;
        for (index, &byte) in bytes.iter().enumerate() {
            value |= ((byte & 0x7f) as u64) << (7 * index);
            if byte < 0x80 {
                *bytes = &bytes[index + 1..];
                return value;
            }
        }
        panic!("a varint ran off the end of the message !!");
    }

    fn take<'a>(bytes: &mut &'a [u8], count: usize) -> &'a [u8] {
        let (head, tail) = bytes.split_at(count);
        *bytes = tail;
        head
    }

    /// The payloads of a repeated length-delimited field, e.g. a graph's nodes.
    fn payloads(bytes: &[u8], field: u32) -> Vec<Vec<u8>> {
        read_fields(bytes).into_iter()
            .filter_map(|(number, value)| match value {
                Value::Bytes(payload) if number == field => Some(payload),
                _ => None,
            })
            .collect()
    }

    /// A repeated varint field, e.g. a tensor's dims.
    fn ints(bytes: &[u8], field: u32) -> Vec<u64> {
        read_fields(bytes).into_iter()
            .filter_map(|(number, value)| match value {
                Value::Varint(number_value) if number == field => Some(number_value),
                _ => None,
            })
            .collect()
    }

    fn texts(bytes: &[u8], field: u32) -> Vec<String> {
        payloads(bytes, field).into_iter()
            .map(|payload| String::from_utf8(payload).unwrap())
            .collect()
    }

    /// The one value of a field that is written at most once.
    fn one(bytes: &[u8], field: u32) -> Value {
        let mut values = read_fields(bytes).into_iter().filter(|(number, _)| *number == field);
        let value = values.next().unwrap_or_else(|| panic!("there is no field {field} !!")).1;
        assert!(values.next().is_none(), "field {field} is written more than once");

        value
    }

    fn payload(bytes: &[u8], field: u32) -> Vec<u8> {
        match one(bytes, field) {
            Value::Bytes(payload) => payload,
            other => panic!("field {field} is {other:?} !!"),
        }
    }

    fn text(bytes: &[u8], field: u32) -> String {
        String::from_utf8(payload(bytes, field)).unwrap()
    }

    fn int(bytes: &[u8], field: u32) -> u64 {
        match one(bytes, field) {
            Value::Varint(value) => value,
            other => panic!("field {field} is {other:?} !!"),
        }
    }

    fn float(bytes: &[u8], field: u32) -> f32 {
        match one(bytes, field) {
            Value::Fixed32(value) => f32::from_le_bytes(value),
            other => panic!("field {field} is {other:?} !!"),
        }
    }

    fn graph_of(model: &Sequential) -> Vec<u8> {
        payload(&model.to_onnx_bytes(), 7)
    }

    #[test]
    fn varints_carry_seven_bits_per_byte() {
        for (value, expected) in [
            (0_u64, vec![0x00]),
            (1, vec![0x01]),
            (127, vec![0x7f]),
            (128, vec![0x80, 0x01]),
            (300, vec![0xac, 0x02]),
            (16_384, vec![0x80, 0x80, 0x01]),
        ] {
            let mut message = Message::default();
            message.varint(value);
            assert_eq!(message.bytes, expected, "{value}");
        }
    }

    /// A runtime reads the version fields first: they say which operator
    /// definitions the rest of the file means.
    #[test]
    fn the_file_names_its_ir_version_and_operator_set() {
        let bytes = model().to_onnx_bytes();

        assert_eq!(int(&bytes, 1), IR_VERSION as u64);
        assert_eq!(text(&bytes, 2), "neuralflow");
        let operator_set = payload(&bytes, 8);
        assert_eq!(text(&operator_set, 1), "", "the empty domain is ai.onnx");
        assert_eq!(int(&operator_set, 2), OPSET_VERSION as u64);
    }

    /// Every Dense becomes a Gemm plus its activation, and the nodes are
    /// chained by name: each one reads the tensor the one before wrote.
    #[test]
    fn the_layers_become_gemm_and_activation_nodes() {
        let graph = graph_of(&model());
        let nodes = payloads(&graph, 1);

        let op_types: Vec<String> = nodes.iter().map(|node| text(node, 4)).collect();
        assert_eq!(op_types, ["Gemm", "Relu", "Gemm", "Sigmoid"]);

        assert_eq!(texts(&nodes[0], 1), ["input", "hidden.weight", "hidden.bias"]);
        assert_eq!(texts(&nodes[0], 2), ["hidden_z"]);
        assert_eq!(texts(&nodes[1], 1), ["hidden_z"]);
        assert_eq!(texts(&nodes[1], 2), ["hidden_a"]);
        assert_eq!(texts(&nodes[2], 1), ["hidden_a", "out.weight", "out.bias"]);
        assert_eq!(texts(&nodes[2], 2), ["out_z"]);
        assert_eq!(texts(&nodes[3], 1), ["out_z"]);
        assert_eq!(texts(&nodes[3], 2), ["output"]);
    }

    /// alpha = beta = 1 and no transposes: Gemm computes A_in·W + b on the
    /// matrices exactly as Layer already stores them.
    #[test]
    fn the_gemm_attributes_keep_the_layout_layer_uses() {
        let graph = graph_of(&model());
        let gemm = payloads(&graph, 1).swap_remove(0);

        let mut floats = Vec::new();
        let mut integers = Vec::new();
        for attribute in payloads(&gemm, 5) {
            let name = text(&attribute, 1);
            match int(&attribute, 20) as i64 {
                ATTRIBUTE_FLOAT => floats.push((name, float(&attribute, 2))),
                ATTRIBUTE_INT => integers.push((name, int(&attribute, 3))),
                other => panic!("unexpected attribute type {other}"),
            }
        }

        assert_eq!(floats, [(String::from("alpha"), 1.), (String::from("beta"), 1.)]);
        assert_eq!(integers, [(String::from("transA"), 0), (String::from("transB"), 0)]);
    }

    /// A = Z for a Linear layer, so there is nothing to add after the Gemm.
    #[test]
    fn a_linear_layer_gets_no_activation_node() {
        let model = Sequential::new(Input::new(1), vec![Dense::new(1, Activation::Linear)]);
        let graph = graph_of(&model);
        let nodes = payloads(&graph, 1);

        assert_eq!(nodes.len(), 1);
        assert_eq!(text(&nodes[0], 4), "Gemm");
        assert_eq!(texts(&nodes[0], 2), ["output"], "the Gemm writes the graph's output");
    }

    /// W keeps its (inputs x units) shape and its row-major order, and b is
    /// one value per unit.
    #[test]
    fn the_weights_travel_as_little_endian_f32() {
        let graph = graph_of(&model());
        let initializers = payloads(&graph, 5);
        let named = |name: &str| initializers.iter()
            .find(|tensor| text(tensor, 8) == name)
            .unwrap_or_else(|| panic!("there is no initializer named {name}"))
            .clone();

        let weights = named("hidden.weight");
        assert_eq!(ints(&weights, 1), [2, 3]);
        assert_eq!(int(&weights, 2), FLOAT as u64);
        let expected: Vec<u8> = [1_f32, 2., 3., 4., 5., 6.].iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(payload(&weights, 9), expected);

        let biases = named("hidden.bias");
        assert_eq!(ints(&biases, 1), [3], "one bias per unit, as a 1-D tensor");
        assert_eq!(payload(&biases, 9), [(-1_f32).to_le_bytes(), 0_f32.to_le_bytes(), 1_f32.to_le_bytes()].concat());
    }

    /// The sample count is a named dimension, so one file serves a caller
    /// predicting one sample and a caller predicting a thousand.
    #[test]
    fn the_shapes_fix_the_features_but_not_the_batch() {
        let graph = graph_of(&model());

        for (field, name, col_count) in [(11, "input", 2), (12, "output", 1)] {
            let value_info = payload(&graph, field);
            assert_eq!(text(&value_info, 1), name);

            let tensor = payload(&payload(&value_info, 2), 1);
            assert_eq!(int(&tensor, 1), FLOAT as u64);
            let dims = payloads(&payload(&tensor, 2), 1);
            assert_eq!(text(&dims[0], 2), "batch");
            assert_eq!(int(&dims[1], 1), col_count);
        }
    }

    #[test]
    fn export_onnx_writes_those_bytes_to_the_file() {
        let model = model();
        let path = std::env::temp_dir().join("neuralflow_export_onnx_test.onnx");
        model.export_onnx(&path).unwrap();

        assert_eq!(fs::read(&path).unwrap(), model.to_onnx_bytes());
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[should_panic(expected = "nothing to export")]
    fn a_model_without_layers_has_nothing_to_export() {
        Sequential::new(Input::new(2), Vec::new()).to_onnx_bytes();
    }
}
