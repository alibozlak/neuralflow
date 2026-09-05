use crate::activation::Activation;

pub struct LayerRequestInfo {
    pub activation: Activation,
    pub unit_count: usize,
}

impl LayerRequestInfo {
    pub fn new(activation: Activation, unit_count: usize) -> Self {
        Self { activation, unit_count }
    }
}