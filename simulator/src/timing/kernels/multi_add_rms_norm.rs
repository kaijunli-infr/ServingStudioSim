//! Residual-stream sum + RMSNorm with a selectable output set, as Inductor
//! fuses it into one Triton reduction.
//!
//! vLLM's compiled graph reaches this kernel with one to three addends: a plain
//! norm, a residual add, or a residual add that re-adds the previous norm's two
//! inputs because Inductor does not store a sum that only this kernel reads.
//! A static FP8 input quantization of the next linear layer fuses in as an
//! output plane. Which planes are written is part of the kernel: each one is a
//! full `num_tokens x hidden_dim` store, so they are config fields, not a
//! post-hoc byte scale.
//!
//! `tune_tokens` (the first-call token count that picks Inductor's launch
//! config, the deployment's `max_num_batched_tokens`) and `rms_eps` are
//! optional: unset, or equal to the 8192 / 1e-6 every earlier row was measured
//! with, they stay out of the identity and the payload, so configs that never
//! set them key the rows they keyed before the fields existed.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MultiAddRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    /// Row-aligned tensors summed before the norm (1 = plain RMSNorm).
    pub num_addends: u32,
    /// The sum is written back as the new residual.
    pub store_sum: bool,
    /// The normalized rows are written in `dtype`.
    pub store_normed: bool,
    /// The normalized rows are written as static per-tensor FP8 E4M3.
    pub store_fp8: bool,
    #[compute_dtype]
    pub dtype: DType,
    /// Inductor tuning size; `None` is [`LEGACY_TUNE_TOKENS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tune_tokens: Option<u32>,
    /// `f64::to_bits` of the RMSNorm epsilon; `None` is [`LEGACY_RMS_EPS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_eps_bits: Option<u64>,
}

/// Tuning size and epsilon of every row measured before the fields existed.
pub const LEGACY_TUNE_TOKENS: u32 = 8192;
pub const LEGACY_RMS_EPS: f64 = 1e-6;

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MultiAddRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct MultiAddRmsNormSpec;

impl KernelSpec for MultiAddRmsNormSpec {
    type Config = MultiAddRmsNormKernelConfig;
    type Input = MultiAddRmsNormKernelInput;

    const KIND: KernelKind = "multi_add_rms_norm";

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
        let tune_tokens = config.tune_tokens.filter(|&t| t != LEGACY_TUNE_TOKENS);
        let rms_eps = config
            .rms_eps_bits
            .map(f64::from_bits)
            .filter(|&eps| eps != LEGACY_RMS_EPS);
        grid.expand_1d(|num_tokens| {
            let payload = ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_dim", config.hidden_dim.get())
                .with("num_addends", config.num_addends)
                .with("store_sum", config.store_sum)
                .with("store_normed", config.store_normed)
                .with("store_fp8", config.store_fp8)
                .with("dtype", config.dtype.as_str());
            let payload = match tune_tokens {
                Some(t) => payload.with("tune_tokens", t),
                None => payload,
            };
            match rms_eps {
                Some(eps) => payload.with("rms_eps", eps),
                None => payload,
            }
        })
    }
}

register_kernel!(MultiAddRmsNormKernel, MultiAddRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{MultiAddRmsNormKernelConfig, MultiAddRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    #[test]
    fn payload_forwards_the_output_set_the_profiler_compiles() {
        // The output planes change the generated kernel; dropping one from the
        // payload would price a GQA input norm with the GDN one's extra store.
        let config = MultiAddRmsNormKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: 5120.into(),
            num_addends: 3,
            store_sum: true,
            store_normed: false,
            store_fp8: true,
            dtype: DType::Bf16,
            tune_tokens: None,
            rms_eps_bits: None,
        };
        let payloads = MultiAddRmsNormSpec::enumerate(
            &config,
            &MultiAddRmsNormSpec::sweep_grid(&config),
            "vllm_inductor",
        );
        let fields = payloads[0].fields();
        assert_eq!(fields["num_tokens"], Value::from(1u32));
        assert_eq!(fields["num_addends"], Value::from(3u32));
        assert_eq!(fields["store_sum"], Value::from(true));
        assert_eq!(fields["store_normed"], Value::from(false));
        assert_eq!(fields["store_fp8"], Value::from(true));
        assert_eq!(fields.len(), 8);
    }

    fn llama_post_norm() -> MultiAddRmsNormKernelConfig {
        MultiAddRmsNormKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA GB300".to_string(),
            hidden_dim: 4096.into(),
            num_addends: 2,
            store_sum: false,
            store_normed: true,
            store_fp8: false,
            dtype: DType::Bf16,
            tune_tokens: None,
            rms_eps_bits: None,
        }
    }

    fn payload_fields(config: &MultiAddRmsNormKernelConfig) -> std::collections::BTreeMap<String, Value> {
        let payloads = MultiAddRmsNormSpec::enumerate(
            config,
            &MultiAddRmsNormSpec::sweep_grid(config),
            "vllm_inductor",
        );
        payloads[0].fields().clone()
    }

    #[test]
    fn legacy_tuning_and_eps_keep_the_old_identity_and_payload() {
        // Spelling out the values earlier rows used must not move their keys.
        let unset = llama_post_norm();
        let spelled = MultiAddRmsNormKernelConfig {
            tune_tokens: Some(super::LEGACY_TUNE_TOKENS),
            rms_eps_bits: Some(super::LEGACY_RMS_EPS.to_bits()),
            ..llama_post_norm()
        };
        assert_eq!(payload_fields(&unset), payload_fields(&spelled));
        assert_eq!(payload_fields(&unset).len(), 8);
        let described = crate::timing::kernels::engine::KernelConfig::describe_config(&unset);
        assert!(described.get("tune_tokens").is_none() && described.get("rms_eps_bits").is_none());
    }

    #[test]
    fn serving_tuning_and_eps_reach_the_profiler() {
        let served = MultiAddRmsNormKernelConfig {
            tune_tokens: Some(2048),
            rms_eps_bits: Some(1e-5f64.to_bits()),
            ..llama_post_norm()
        };
        let fields = payload_fields(&served);
        assert_eq!(fields["tune_tokens"], Value::from(2048u32));
        assert_eq!(fields["rms_eps"], Value::from(1e-5f64));
        assert_eq!(fields.len(), 10);
    }
}
