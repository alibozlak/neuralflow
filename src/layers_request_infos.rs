use crate::activation::Activation;

pub struct LayersRequestInfos {
    pub activations: Vec<Activation>,
    pub unit_counts: Vec<usize>,
}

impl LayersRequestInfos {
    pub fn new(activations: Vec<Activation>, unit_counts: Vec<usize>) -> Self {
        Self { activations, unit_counts }
    }
}