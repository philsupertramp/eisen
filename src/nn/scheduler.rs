/// Cosine Learning Rate Scheduler with Linear Warmup.
///
/// Schedule shape:
///
///   ┌── linear warmup ──┬──────────── cosine decay ──────────────┐
///   │                   │                                         │
///   lr_max             lr_max                                   lr_min
///
/// During warmup:  lr(t) = lr_max · t / warmup_steps
/// After warmup:   lr(t) = lr_min + ½·(lr_max - lr_min)·(1 + cos(π · progress))
///
/// At progress = 0 → lr_max, at progress = 1 → lr_min.
///
// Usage:
// ```
// let sched = CosineScheduler::new(3e-4, 1e-5, 100, 50_000);
// for step in 0..50_000 {
//     optim.lr = sched.get_lr(step);
//     optim.step(&mut g);
// }
// ```
pub struct CosineScheduler {
    /// Peak learning rate (reached at end of warmup).
    pub lr_max: f32,
    /// Floor learning rate (reached at `total_steps`).
    pub lr_min: f32,
    /// Number of steps to linearly warm up from 0 → lr_max.
    pub warmup_steps: usize,
    /// Total training steps (warmup + cosine decay).
    pub total_steps: usize,
}

impl CosineScheduler {
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        assert!(
            total_steps > warmup_steps,
            "total_steps ({}) must be greater than warmup_steps ({})",
            total_steps,
            warmup_steps
        );
        Self { lr_max, lr_min, warmup_steps, total_steps }
    }

    /// Returns the learning rate for the given step index (0-based).
    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup: avoid lr=0 at step 0 by offsetting by 1.
            self.lr_max * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps).max(1);
            let progress = ((step - self.warmup_steps) as f32 / decay_steps as f32).clamp(0.0, 1.0);
            self.lr_min
                + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}
