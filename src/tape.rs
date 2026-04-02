use crate::tensor::Tensor;

pub struct TapeNode {
    pub inputs: Vec<usize>,
    pub output: usize,
    pub backward_fn: Box<dyn Fn(&mut [Tensor])>,
}

#[derive(Default)]
pub struct Tape {
    pub nodes: Vec<TapeNode>,
}
