
use crate::layer::Layer;

pub struct Model {
    layers : Vec<Layer>,
    layer_count : usize,
    layer_type: String,
}

impl Model {

    // pub fn sequential_model(layers_request_infos: LayersRequestInfos, samples_feature_counts: usize) -> Self {
    // }

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

    pub fn get_mut_layers(&mut self) -> &mut Vec<Layer> {
        &mut self.layers
    }

    pub fn get_layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn get_layer_type(&self) -> &str {
        &self.layer_type
    }

    pub fn get_mut_layer(&mut self, layer_index: usize) -> Layer {
        if layer_index >= self.layer_count {
            panic!("Layer index out of bounds !! Layers count : {}, your layer_index: {}", self.layer_count, layer_index);
        }

        self.layers[layer_index].clone()
    }








}