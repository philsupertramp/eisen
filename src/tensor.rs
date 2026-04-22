use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

// ─── BF16 bit-cast helpers (no CUDA needed) ───────────────────────────────────

#[inline(always)]
pub fn f32_to_bf16u(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

#[inline(always)]
pub fn bf16u_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ─── Device ──────────────────────────────────────────────────────────────────

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

// ─── Storage ─────────────────────────────────────────────────────────────────

pub enum Storage {
    Cpu(Vec<f32>),
    Gpu(CudaSlice<f32>),
    #[cfg(feature = "bf16")]
    GpuBf16(CudaSlice<u16>),
    /// CPU-resident weight stored in BF16 (u16 bits) to halve RAM usage for
    /// streamed parameters.  Grad buffers are never CpuBf16 — always Cpu(f32).
    #[cfg(feature = "bf16")]
    CpuBf16(Vec<u16>),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(v) => Storage::Cpu(v.clone()),
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(v) => Storage::CpuBf16(v.clone()),
            Storage::Gpu(_) => panic!("Cloning GPU storage is not supported."),
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(_) => panic!("Cloning GpuBf16 storage is not supported."),
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Cpu(v) => write!(f, "Cpu({} f32)", v.len()),
            Storage::Gpu(s) => write!(f, "Gpu({} f32)", s.len()),
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => write!(f, "GpuBf16({} u16)", s.len()),
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(v) => write!(f, "CpuBf16({} u16)", v.len()),
        }
    }
}

impl Storage {
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Gpu(s) => s.len(),
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => s.len(),
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(v) => v.len(),
        }
    }

    pub fn as_cpu(&self) -> &Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(_) => panic!(
                "Storage is CpuBf16 — call to_f32_vec() for an owned conversion, \
                 or match explicitly."
            ),
            _ => panic!("Attempted to access GPU data as CPU slice."),
        }
    }

    pub fn as_cpu_mut(&mut self) -> &mut Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(_) => panic!("CpuBf16 has no mutable f32 view — use to_f32_vec()."),
            _ => panic!("Attempted to access GPU data as CPU slice."),
        }
    }

    /// Convert any CPU variant to an owned `Vec<f32>`.
    /// For `CpuBf16` this decompresses each element; for `Cpu` it clones.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(v) => v.clone(),
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(v) => v.iter().map(|&b| bf16u_to_f32(b)).collect(),
            _ => panic!("to_f32_vec called on GPU storage"),
        }
    }

    /// True when this is a CPU-backed storage (either Cpu or CpuBf16).
    pub fn is_cpu(&self) -> bool {
        matches!(self, Storage::Cpu(_))
            || {
                #[cfg(feature = "bf16")]
                {
                    matches!(self, Storage::CpuBf16(_))
                }
                #[cfg(not(feature = "bf16"))]
                false
            }
    }
}

// ─── Tensor ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Storage,
    pub grad: Storage,
    pub device: Device,
    pub name: Option<String>,
    pub is_pooled: bool,
}

impl Tensor {
    pub fn new(id: usize, shape: Vec<usize>, data: Vec<f32>, device: Device) -> Self {
        let size: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let strides = Self::compute_strides(&shape);

        let (data_storage, grad_storage) = match &device {
            Device::Cpu => (
                Storage::Cpu(data),
                Storage::Cpu(vec![0.0; size]),
            ),
            Device::Gpu(_ctx, stream) => {
                let d_data = stream.clone_htod(data.as_slice())
                    .expect("Failed to copy data to VRAM");
                let d_grad = stream.alloc_zeros(size)
                    .expect("Failed to allocate gradients in VRAM");
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
            is_pooled: false,
        }
    }

    pub fn sync_to_cpu(&self) -> Vec<f32> {
        match &self.data {
            Storage::Cpu(v) => v.clone(),
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(v) => v.iter().map(|&b| bf16u_to_f32(b)).collect(),
            Storage::Gpu(s) => {
                let stream = match &self.device {
                    Device::Gpu(_, s) => s,
                    _ => unreachable!(),
                };
                stream.clone_dtoh(s).expect("Failed to copy data from VRAM to Host")
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => {
                let stream = match &self.device {
                    Device::Gpu(_, s) => s,
                    _ => unreachable!(),
                };
                let u16_data = stream.clone_dtoh(s)
                    .expect("Failed to copy BF16 data from VRAM to Host");
                u16_data.into_iter().map(bf16u_to_f32).collect()
            }
        }
    }

    pub fn sync_grad_to_cpu(&self) -> Vec<f32> {
        match &self.grad {
            Storage::Cpu(v) => v.clone(),
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(v) => v.iter().map(|&b| bf16u_to_f32(b)).collect(),
            Storage::Gpu(s) => {
                let stream = match &self.device {
                    Device::Gpu(_, s) => s,
                    _ => unreachable!(),
                };
                stream.clone_dtoh(s).expect("Failed to copy data from VRAM to Host")
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => {
                let stream = match &self.device {
                    Device::Gpu(_, s) => s,
                    _ => unreachable!(),
                };
                let u16_data = stream.clone_dtoh(s)
                    .expect("Failed to copy BF16 data from VRAM to Host");
                u16_data.into_iter().map(bf16u_to_f32).collect()
            }
        }
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut current = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = current;
            current *= shape[i];
        }
        strides
    }

    pub fn get_broadcasted_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let max_len = a.len().max(b.len());
        let mut out = vec![0; max_len];
        for i in 0..max_len {
            let da = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
            let db = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
            out[i] = da.max(db);
        }
        out
    }

    pub fn get_broadcasted_strides(
        shape: &[usize],
        base_strides: &[usize],
        broadcasted_shape: &[usize],
    ) -> Vec<usize> {
        let mut strides = vec![0; broadcasted_shape.len()];
        let pad = broadcasted_shape.len() - shape.len();
        for i in 0..shape.len() {
            if shape[i] == broadcasted_shape[pad + i] {
                strides[pad + i] = base_strides[i];
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

    pub fn size(&self) -> usize {
        if self.shape.is_empty() { 1 } else { self.shape.iter().product() }
    }
}
