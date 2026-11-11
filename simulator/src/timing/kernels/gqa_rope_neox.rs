//! Rotary position embedding of the query and key heads of a plain GQA
//! attention layer, as Inductor compiles vLLM's QKV split plus native
//! `RotaryEmbedding` (NeoX style) into one Triton pointwise kernel. It reads
//! the query and key slices of the fused QKV output in place and writes
//! contiguous query and key tensors; the value slice is not touched.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GqaRopeNeoxKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Leading elements of each head that are rotated.
    pub rotary_dim: Dim,
    /// Rows of the cos/sin cache the positions index.
    pub max_position: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GqaRopeNeoxKernelInput {
    pub num_tokens: u32,
}

pub struct GqaRopeNeoxSpec;

impl KernelSpec for GqaRopeNeoxSpec {
    type Config = GqaRopeNeoxKernelConfig;
    type Input = GqaRopeNeoxKernelInput;

    const KIND: KernelKind = "gqa_rope_neox";

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
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("rotary_dim", config.rotary_dim.get())
                .with("max_position", config.max_position)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GqaRopeNeoxKernel, GqaRopeNeoxSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn llama() -> GqaRopeNeoxKernelConfig {
        GqaRopeNeoxKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA GB300".into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            rotary_dim: 128.into(),
            max_position: 131072,
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn grid_covers_decode_and_prefill_token_counts() {
        // Defect caught: a token axis that starts at 32 and extrapolates every
        // decode batch.
        let grid = GqaRopeNeoxSpec::sweep_grid(&llama());
        let rows = GqaRopeNeoxSpec::enumerate(&llama(), &grid, "vllm_inductor");
        assert_eq!(rows.len(), 68);
        let tokens: Vec<u64> = rows
            .iter()
            .map(|r| r.fields()["num_tokens"].as_u64().unwrap())
            .collect();
        for t in [1, 8, 16, 32, 256, 2048, 65536] {
            assert!(tokens.contains(&t), "missing {t}");
        }
    }

    #[test]
    fn enumerate_emits_exactly_the_python_args() {
        // The Python GqaRopeNeoxArgs fields, plus the backend.
        let grid = GqaRopeNeoxSpec::sweep_grid(&llama());
        let row = &GqaRopeNeoxSpec::enumerate(&llama(), &grid, "vllm_inductor")[0];
        let mut keys: Vec<&str> = row.fields().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "head_dim",
                "max_position",
                "num_kv_heads",
                "num_qo_heads",
                "num_tokens",
                "rotary_dim"
            ]
        );
    }
}
