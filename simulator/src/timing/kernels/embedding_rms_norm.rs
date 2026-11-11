//! Token embedding gather fused with the first decoder layer's input RMSNorm,
//! as Inductor compiles vLLM's Llama graph into one Triton reduction
//! (`triton_red_fused__to_copy_embedding_rms_norm`): read each token's int32
//! id, gather that table row, store it as the residual and store its
//! normalized copy. A plain one-addend RMSNorm over contiguous rows
//! (`multi_add_rms_norm`) has neither the gather nor the residual store and is
//! a different kernel.
//!
//! The tuning size (the deployment's `max_num_batched_tokens`) and the epsilon
//! are part of the identity: both are baked into the compiled kernel.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    /// Rows of the embedding table the token ids index.
    pub vocab_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// `f64::to_bits` of the RMSNorm epsilon.
    pub rms_eps_bits: u64,
    /// First-call token count that picks Inductor's launch config.
    pub tune_tokens: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct EmbeddingRmsNormSpec;

impl KernelSpec for EmbeddingRmsNormSpec {
    type Config = EmbeddingRmsNormKernelConfig;
    type Input = EmbeddingRmsNormKernelInput;

    const KIND: KernelKind = "embedding_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_dim", config.hidden_dim.get())
                .with("vocab_size", config.vocab_size.get())
                .with("dtype", config.dtype.as_str())
                .with("rms_eps", f64::from_bits(config.rms_eps_bits))
                .with("tune_tokens", config.tune_tokens)
        })
    }
}

register_kernel!(EmbeddingRmsNormKernel, EmbeddingRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn llama() -> EmbeddingRmsNormKernelConfig {
        EmbeddingRmsNormKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA GB300".into(),
            hidden_dim: 4096.into(),
            vocab_size: 128256.into(),
            dtype: DType::Bf16,
            rms_eps_bits: 1e-5f64.to_bits(),
            tune_tokens: 2048,
        }
    }

    #[test]
    fn payload_carries_every_profiled_field() {
        // Defect caught: a field missing from the payload would key a row the
        // profiler cannot build, or let two epsilons/tunings share one row.
        let c = llama();
        let payloads = EmbeddingRmsNormSpec::enumerate(
            &c,
            &EmbeddingRmsNormSpec::sweep_grid(&c),
            "vllm_inductor",
        );
        let fields = payloads[0].fields();
        assert_eq!(fields["num_tokens"], Value::from(1u32));
        assert_eq!(fields["hidden_dim"], Value::from(4096u32));
        assert_eq!(fields["vocab_size"], Value::from(128256u32));
        assert_eq!(fields["dtype"], Value::from("bf16"));
        assert_eq!(fields["rms_eps"], Value::from(1e-5f64));
        assert_eq!(fields["tune_tokens"], Value::from(2048u32));
        assert_eq!(fields.len(), 7);
    }

    #[test]
    fn grid_matches_the_norm_kinds() {
        // Same token axis as multi_add_rms_norm (68 points), so the first norm
        // and the per-layer norms interpolate at the same knots.
        let c = llama();
        let n = EmbeddingRmsNormSpec::enumerate(
            &c,
            &EmbeddingRmsNormSpec::sweep_grid(&c),
            "vllm_inductor",
        )
        .len();
        assert_eq!(n, 68);
    }
}
