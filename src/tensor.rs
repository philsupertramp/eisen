use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice};

#[derive(Clone)]
pub enum Device {
    Cpu,
    Gpu(Arc<CudaDevice>),
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "Cpu"),
            Device::Gpu(_) => write!(f, "Gpu"),
        }
    }
}

pub enum Storage {
    Cpu(Vec<f32>),
    Gpu(CudaSlice<f32>),
}

impl Storage {
    /// Helper to get CPU data or panic if it's on GPU. 
    /// This allows us to keep our existing CPU math working during the transition.
    pub fn as_cpu(&self) -> &Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            Storage::Gpu(_) => panic!("Attempted to access GPU data as CPU slice"),
        }
    }

    pub fn as_cpu_mut(&mut self) -> &mut Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            Storage::Gpu(_) => panic!("Attempted to access GPU data as CPU slice"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Storage,
    pub grad: Storage,
    pub device: Device,
    pub name: Option<String>,
}

impl Tensor {
    pub fn new(id: usize, shape: Vec<usize>, data: Vec<f32>, device: Device) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "Data length {} must exactly match shape capacity {}",
            data.len(),
            size
        );

        let strides = Self::compute_strides(&shape);

        Self {
            id,
            shape,
            strides,
            data: Storage::Cpu(data),
            grad: Storage::Cpu(vec![0.0; size]),
            device: device,
            name: None,
        }
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;

        for i in (0..shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
        strides
    }

    /// Calculates the output shape when broadcasting two tensors together.
    pub fn get_broadcasted_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let max_len = a.len().max(b.len());
        let mut out = vec![0; max_len];
        for i in 0..max_len {
            let dim_a = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
            let dim_b = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
            assert!(dim_a == dim_b || dim_a == 1 || dim_b == 1, "Shapes not broadcastable");
            out[i] = dim_a.max(dim_b);
        }
        out
    }

    /// Computes virtual strides for broadcasting. Padded/Size-1 dimensions get a stride of 0.
    pub fn get_broadcasted_strides(shape: &[usize], base_strides: &[usize], broadcasted_shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; broadcasted_shape.len()];
        let pad = broadcasted_shape.len() - shape.len();
        for i in 0..shape.len() {
            if shape[i] == broadcasted_shape[pad + i] {
                strides[pad + i] = base_strides[i];
            } else {
                strides[pad + i] = 0; // The broadcasting magic: stride 0 repeats the value!
            }
        }
        strides
    }

    /// Converts a flat 1D memory index into multidimensional coordinates.
    pub fn flat_to_nd(mut flat: usize, shape: &[usize]) -> Vec<usize> {
        let mut nd = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            nd[i] = flat % shape[i];
            flat /= shape[i];
        }
        nd
    }

    /// Converts multidimensional coordinates back to a flat 1D index using specific strides.
    pub fn nd_to_flat(nd: &[usize], strides: &[usize]) -> usize {
        nd.iter().zip(strides.iter()).map(|(n, s)| n * s).sum()
    }
}
