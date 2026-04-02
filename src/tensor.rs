use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

#[derive(Clone)]
pub enum Device {
    Cpu,
    Gpu(Arc<CudaContext>, Arc<CudaStream>),
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "Cpu"),
            Device::Gpu(_, _) => write!(f, "Gpu"),
        }
    }
}

pub enum Storage {
    Cpu(Vec<f32>),
    Gpu(CudaSlice<f32>),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(v) => Storage::Cpu(v.clone()),
            // We don't want to accidentally trigger massive VRAM-to-VRAM copies 
            // without being explicit about it.
            Storage::Gpu(_) => panic!("Cloning GPU storage is not yet supported. Use explicit device-to-device transfers."),
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Cpu(v) => write!(f, "Cpu({:?})", v),
            Storage::Gpu(_) => write!(f, "Gpu(CudaSlice)"),
        }
    }
}

impl Storage {
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Gpu(s) => s.len(),
        }
    }

    pub fn as_cpu(&self) -> &Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            Storage::Gpu(_) => panic!("Attempted to access GPU data as CPU slice. Did you forget to call sync_to_cpu()?"),
        }
    }

    pub fn as_cpu_mut(&mut self) -> &mut Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            Storage::Gpu(_) => panic!("Attempted to access GPU data as CPU slice."),
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
        let strides = Self::compute_strides(&shape);

        // Real VRAM allocation logic
        let (data_storage, grad_storage) = match &device {
            Device::Cpu => (
                Storage::Cpu(data),
                Storage::Cpu(vec![0.0; size])
            ),
            Device::Gpu(ctx, stream) => {
                // Host-to-Device Copy using the stream
                let d_data = stream.clone_htod(data.as_slice()).expect("Failed to copy data to VRAM");
                // Zero-allocation on Device
                let d_grad = stream.alloc_zeros(size).expect("Failed to allocate gradients in VRAM");
                (Storage::Gpu(d_data), Storage::Gpu(d_grad))
            }
        };

        Self {
            id,
            shape,
            strides,
            data: data_storage,
            grad: grad_storage,
            device,
            name: None,
        }
    }

    /// Pulls data from VRAM back to a CPU Vec<f32>. 
    /// If already on CPU, it returns a clone.
    pub fn sync_to_cpu(&self) -> Vec<f32> {
        match &self.data {
            Storage::Cpu(v) => v.clone(),
            Storage::Gpu(s) => {
                let (ctx, stream) = match &self.device {
                    Device::Gpu(c, s) => (c, s),
                    _ => unreachable!(),
                };
                stream.clone_dtoh(s).expect("Failed to copy data from VRAM to Host")
            }
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

    pub fn get_broadcasted_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let max_len = a.len().max(b.len());
        let mut out = vec![0; max_len];
        for i in 0..max_len {
            let dim_a = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
            let dim_b = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
            out[i] = dim_a.max(dim_b);
        }
        out
    }

    pub fn get_broadcasted_strides(shape: &[usize], base_strides: &[usize], broadcasted_shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; broadcasted_shape.len()];
        let pad = broadcasted_shape.len() - shape.len();
        for i in 0..shape.len() {
            if shape[i] == broadcasted_shape[pad + i] {
                strides[pad + i] = base_strides[i];
            } else {
                strides[pad + i] = 0;
            }
        }
        strides
    }

    pub fn flat_to_nd(mut flat: usize, shape: &[usize]) -> Vec<usize> {
        let mut nd = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            nd[i] = flat % shape[i];
            flat /= shape[i];
        }
        nd
    }

    pub fn nd_to_flat(nd: &[usize], strides: &[usize]) -> usize {
        nd.iter().zip(strides.iter()).map(|(n, s)| n * s).sum()
    }
}
