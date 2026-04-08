use eisen::graph::{Graph, PrecisionMode};
use eisen::tensor::Device;

#[test]
fn cpu_graph_defaults_to_fp32_precision_mode() {
    let g = Graph::new(Device::Cpu);
    assert_eq!(g.precision_mode(), PrecisionMode::Fp32);
    assert!(!g.uses_bf16_mixed_precision());
}
