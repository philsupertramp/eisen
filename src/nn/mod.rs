pub mod linear;
pub mod optim;
pub mod embedding;
pub mod rmsnorm;

use crate::graph::Graph;

/// The core trait for all Neural Network modules.
pub trait Module {
    /// Computes the forward pass and returns the output tensor ID.
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize;
    
    /// Returns the Graph IDs of the learnable parameters in this module.
    fn params(&self) -> Vec<usize>;
}
