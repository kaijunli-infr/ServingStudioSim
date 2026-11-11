//! `llama3_dense_vllm` — opt-in variant of [`llama3_dense`](super::llama3_dense)
//! that prices a bf16 Llama 3.x dense model on one GPU with the kernels vLLM's
//! compiled graph actually launches (`torch.compile`, FlashInfer TRT-LLM
//! attention, KV cache in HND layout).
//!
//! Same topology, GEMM shapes and serial `Sum` as `llama3_dense`; only the leaf
//! kernels differ. Mapping from the GB300 Llama 3.1 8B TP1 trace and the
//! Inductor code vLLM compiled for it (`profiling/env/arm64-gb300/
//! PHASE3A_DESIGN.md`):
//!
//! ```text
//! first_norm        embedding_rms_norm vllm_inductor (gather by id, store residual + normed)
//!                   = the embedding and layer 0's input norm, one kernel
//! per layer (×N):
//!   qkv_proj        single_gemm torch_linear (F.linear layout)
//!   rope            gqa_rope_neox vllm_inductor_graph                   (new: llama3_dense has no RoPE leaf)
//!   kv_cache_append kv_cache_append vllm_cuda, HND_PACKED, V strided from the QKV row
//!   prefill / decode paged_attn_batch_{prefill,decode} flashinfer_trtllm_gen, page 16 (counter fill inside)
//!   o_proj          single_gemm torch_linear (split-K reduce inside the row)
//!   post_norm       multi_add_rms_norm (2 addends, stores normed)
//!   up_gate_proj    single_gemm torch_linear
//!   activation      silu_and_mul vllm_inductor_graph
//!   down_proj       single_gemm torch_linear (split-K reduce inside the row)
//! next_input_norm   multi_add_rms_norm (3 addends, stores sum + normed) ×(N−1): layers 1..N−1's input norm
//! final_norm        multi_add_rms_norm (3 addends, stores normed)
//! lm_head           single_gemm torch_linear, one row per request
//! ```
//!
//! Serving configuration (GB300 TP1 pilot, `profiling/env/arm64-gb300/
//! PHASE3B_PILOT_RESULTS.md`): every Inductor norm, the first one included, is
//! keyed by the deployment's `max_num_batched_tokens` (its tuning size) and the
//! checkpoint's `rms_norm_eps`; the KV append writes vLLM's packed FlashInfer
//! cache with V read in place from the QKV output; attention runs the 16-token
//! page specialization. SiLU and RoPE are priced from CUDA-graph replay rows
//! (`vllm_inductor_graph`), the boundary that matched the served kernels; every
//! other leaf from cold eager rows.
//!
//! Not modeled, as in `llama3_dense`: CUDA-graph padding of a decode batch (the
//! leaves run at the real token and request counts), the engine's runtime
//! kernels (input preparation, sampling, bookkeeping) and GPU idle time. Those
//! are serving-execution terms, measured beside the prediction, not charged by
//! the architecture.

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::kv_layout::{gqa_kv_bytes_per_token, gqa_placement, KvCacheLayout, KvCachePart};
use crate::arch::model_cfg::ModelCfg;
use crate::arch::DenseParallel;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    EmbeddingRmsNormKernel, EmbeddingRmsNormKernelConfig, EmbeddingRmsNormKernelInput,
    GqaRopeNeoxKernel, GqaRopeNeoxKernelConfig, GqaRopeNeoxKernelInput, KvCacheAppendKernel,
    KvCacheAppendKernelConfig, KvCacheAppendKernelInput, MultiAddRmsNormKernel,
    MultiAddRmsNormKernelConfig, MultiAddRmsNormKernelInput, PagedAttnBatchDecodeKernel,
    PagedAttnBatchDecodeKernelConfig, PagedAttnBatchDecodeKernelInput,
    PagedAttnBatchPrefillKernel, PagedAttnBatchPrefillKernelConfig,
    PagedAttnBatchPrefillKernelInput, SiluAndMulKernel, SiluAndMulKernelConfig,
    SiluAndMulKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::glm53_common::{atomic, push_or_zero};

/// `F.linear` with the weight `[n, k]`: the operand layout vLLM's graph calls
/// (`extern_kernels.mm(x, W.T)`). Not best-of with `torch`, whose `[k, n]`
/// weight is a different cuBLAS problem.
const GEMM_BACKENDS: &[&str] = &["torch_linear"];
const INDUCTOR_BACKENDS: &[&str] = &["vllm_inductor"];
/// The same Inductor kernels timed by CUDA-graph replay (warm L2, no launch
/// gap), the boundary that matched the served SiLU and RoPE kernels.
const INDUCTOR_GRAPH_BACKENDS: &[&str] = &["vllm_inductor_graph"];
const KV_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const ATTENTION_BACKENDS: &[&str] = &["flashinfer_trtllm_gen"];
/// Tokens per KV page (the trace's attention kernel name carries `P16`); also
/// the attention leaves' `page_size`, which selects that cubin.
const KV_BLOCK_SIZE: u32 = 16;
/// vLLM's FlashInfer cache: one `[blocks, heads, page, 2 * head_dim]` buffer
/// whose halves are K and V, which TRT-LLM attention reads.
const KV_CACHE_LAYOUT: &str = "HND_PACKED";
const KV_SCALE_GRANULARITY: &str = "tensor";

/// The serving deployment's settings the leaves are keyed by.
#[derive(Clone, Debug)]
pub struct VllmServing {
    /// vLLM's `max_num_batched_tokens`: Inductor's tuning size.
    pub max_num_batched_tokens: u32,
    /// The checkpoint's `rms_norm_eps`.
    pub rms_norm_eps: f64,
}

/// Every leaf's kernel config, unresolved (all are atomic: no partition).
#[derive(Clone, Debug)]
pub struct Llama3DenseVllmConfigs {
    pub first_norm: EmbeddingRmsNormKernelConfig,
    pub qkv_proj: SingleGemmKernelConfig,
    pub rope: GqaRopeNeoxKernelConfig,
    pub kv_cache_append: KvCacheAppendKernelConfig,
    pub prefill_attention: PagedAttnBatchPrefillKernelConfig,
    pub decode_attention: PagedAttnBatchDecodeKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub post_norm: MultiAddRmsNormKernelConfig,
    pub up_gate_proj: SingleGemmKernelConfig,
    pub activation: SiluAndMulKernelConfig,
    pub down_proj: SingleGemmKernelConfig,
    pub next_input_norm: MultiAddRmsNormKernelConfig,
    pub final_norm: MultiAddRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_dtype: DType,
}

pub fn build_configs(
    model: &ModelCfg,
    parallel: &DenseParallel,
    rope_max_position: u32,
    serving: &VllmServing,
) -> Llama3DenseVllmConfigs {
    let gpu = &parallel.gpu_name;
    let dtype = model.dtype;
    let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
        backends: GEMM_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        n,
        k,
        dtype,
    };
    let norm = |num_addends: u32, store_sum: bool| MultiAddRmsNormKernelConfig {
        backends: INDUCTOR_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        hidden_dim: model.hidden.clone(),
        num_addends,
        store_sum,
        store_normed: true,
        store_fp8: false,
        dtype,
        tune_tokens: Some(serving.max_num_batched_tokens),
        rms_eps_bits: Some(serving.rms_norm_eps.to_bits()),
    };
    let attention_geometry = (
        model.num_qo_heads.clone(),
        model.num_kv_heads.clone(),
        model.head_dim.clone(),
    );
    Llama3DenseVllmConfigs {
        first_norm: EmbeddingRmsNormKernelConfig {
            backends: INDUCTOR_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden.clone(),
            vocab_size: model.vocab.clone(),
            dtype,
            rms_eps_bits: serving.rms_norm_eps.to_bits(),
            tune_tokens: serving.max_num_batched_tokens,
        },
        qkv_proj: gemm(
            (model.num_qo_heads.clone() + 2 * model.num_kv_heads.clone()) * model.head_dim.clone(),
            model.hidden.clone(),
        ),
        rope: GqaRopeNeoxKernelConfig {
            backends: INDUCTOR_GRAPH_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_qo_heads: model.num_qo_heads.clone(),
            num_kv_heads: model.num_kv_heads.clone(),
            head_dim: model.head_dim.clone(),
            // Llama rotates the whole head.
            rotary_dim: model.head_dim.clone(),
            max_position: rope_max_position,
            dtype,
        },
        kv_cache_append: KvCacheAppendKernelConfig {
            backends: KV_APPEND_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_kv_heads: model.num_kv_heads.clone(),
            head_dim: model.head_dim.clone(),
            head_dim_v: None,
            block_size: KV_BLOCK_SIZE,
            input_dtype: dtype,
            kv_dtype: model.kv_dtype,
            cache_layout: KV_CACHE_LAYOUT.to_string(),
            scale_granularity: KV_SCALE_GRANULARITY.to_string(),
            // V is the trailing slice of the fused QKV output row.
            value_row_stride: Some(
                (model.num_qo_heads.get() + 2 * model.num_kv_heads.get()) * model.head_dim.get(),
            ),
        },
        prefill_attention: PagedAttnBatchPrefillKernelConfig {
            backends: ATTENTION_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_qo_heads: attention_geometry.0.clone(),
            num_kv_heads: attention_geometry.1.clone(),
            head_dim: attention_geometry.2.clone(),
            q_dtype: dtype,
            kv_dtype: model.kv_dtype,
            o_dtype: dtype,
            page_size: Some(KV_BLOCK_SIZE),
        },
        decode_attention: PagedAttnBatchDecodeKernelConfig {
            backends: ATTENTION_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_qo_heads: attention_geometry.0,
            num_kv_heads: attention_geometry.1,
            head_dim: attention_geometry.2,
            q_dtype: dtype,
            kv_dtype: model.kv_dtype,
            o_dtype: dtype,
            page_size: Some(KV_BLOCK_SIZE),
        },
        o_proj: gemm(
            model.hidden.clone(),
            model.num_qo_heads.clone() * model.head_dim.clone(),
        ),
        post_norm: norm(2, false),
        up_gate_proj: gemm(2 * model.intermediate.clone(), model.hidden.clone()),
        activation: SiluAndMulKernelConfig {
            backends: INDUCTOR_GRAPH_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.intermediate.clone(),
            dtype,
        },
        down_proj: gemm(model.hidden.clone(), model.intermediate.clone()),
        next_input_norm: norm(3, true),
        final_norm: norm(3, false),
        lm_head: gemm(model.vocab.clone(), model.hidden.clone()),
        num_layers: model.num_layers,
        num_kv_heads: model.num_kv_heads.get(),
        head_dim: model.head_dim.get(),
        kv_dtype: model.kv_dtype,
    }
}

pub struct Llama3DenseVllmModel {
    pub name: String,
    pub num_layers: u32,
    pub kv_cache_part: KvCachePart,
    first_norm: Op<EmbeddingRmsNormKernel>,
    qkv_proj: Op<SingleGemmKernel>,
    rope: Op<GqaRopeNeoxKernel>,
    kv_cache_append: Op<KvCacheAppendKernel>,
    prefill_attention: Op<PagedAttnBatchPrefillKernel>,
    decode_attention: Op<PagedAttnBatchDecodeKernel>,
    o_proj: Op<SingleGemmKernel>,
    post_norm: Op<MultiAddRmsNormKernel>,
    up_gate_proj: Op<SingleGemmKernel>,
    activation: Op<SiluAndMulKernel>,
    down_proj: Op<SingleGemmKernel>,
    next_input_norm: Op<MultiAddRmsNormKernel>,
    final_norm: Op<MultiAddRmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    model_name: String,
    cfgs: Llama3DenseVllmConfigs,
    bridge: &PerfApiBridge,
) -> Result<Llama3DenseVllmModel, BuildError> {
    let n = model_name.as_str();
    let layer = format!("{model_name}.layer");
    let l = layer.as_str();
    let kv_cache_part = KvCachePart::per_token(
        "gqa",
        gqa_kv_bytes_per_token(
            u64::from(cfgs.num_layers),
            u64::from(cfgs.num_kv_heads),
            u64::from(cfgs.head_dim),
            u64::from(cfgs.head_dim),
            u64::from(cfgs.kv_dtype.size_bytes()),
        ),
    )
    .placed(gqa_placement(u64::from(cfgs.num_kv_heads), 1));
    let mut model = Llama3DenseVllmModel {
        first_norm: atomic(n, "first_norm", cfgs.first_norm, EmbeddingRmsNormKernel::build, bridge)?,
        qkv_proj: atomic(l, "qkv_proj", cfgs.qkv_proj, SingleGemmKernel::build, bridge)?,
        rope: atomic(l, "rope", cfgs.rope, GqaRopeNeoxKernel::build, bridge)?,
        kv_cache_append: atomic(
            l,
            "kv_cache_append",
            cfgs.kv_cache_append,
            KvCacheAppendKernel::build,
            bridge,
        )?,
        prefill_attention: atomic(
            l,
            "prefill_attention",
            cfgs.prefill_attention,
            PagedAttnBatchPrefillKernel::build,
            bridge,
        )?,
        decode_attention: atomic(
            l,
            "decode_attention",
            cfgs.decode_attention,
            PagedAttnBatchDecodeKernel::build,
            bridge,
        )?,
        o_proj: atomic(l, "o_proj", cfgs.o_proj, SingleGemmKernel::build, bridge)?,
        post_norm: atomic(l, "post_norm", cfgs.post_norm, MultiAddRmsNormKernel::build, bridge)?,
        up_gate_proj: atomic(l, "up_gate_proj", cfgs.up_gate_proj, SingleGemmKernel::build, bridge)?,
        activation: atomic(l, "activation", cfgs.activation, SiluAndMulKernel::build, bridge)?,
        down_proj: atomic(l, "down_proj", cfgs.down_proj, SingleGemmKernel::build, bridge)?,
        next_input_norm: atomic(
            n,
            "next_input_norm",
            cfgs.next_input_norm,
            MultiAddRmsNormKernel::build,
            bridge,
        )?,
        final_norm: atomic(n, "final_norm", cfgs.final_norm, MultiAddRmsNormKernel::build, bridge)?,
        lm_head: atomic(n, "lm_head", cfgs.lm_head, SingleGemmKernel::build, bridge)?,
        name: model_name,
        num_layers: cfgs.num_layers,
        kv_cache_part,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    tracing::info!(
        "[build] cost tree ({} leaf slots):\n{}",
        tree.n_slots(),
        tree.describe()
    );
    Ok(model)
}

impl Llama3DenseVllmModel {
    /// Layers after the first: each issues the next layer's fused input norm.
    fn later_layers(&self) -> u32 {
        self.num_layers.saturating_sub(1)
    }

    /// `Sum( first_norm, Scale{N}(layer), Scale{N-1}(next_input_norm),
    /// final_norm, lm_head )`; the second fold is omitted for a one-layer model.
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut parts = vec![self.first_norm.compile(&mut b)];
        parts.push(CostNode::Labeled {
            label: "layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.num_layers,
                child: Box::new(CostNode::Sum(vec![
                    self.qkv_proj.compile(&mut b),
                    self.rope.compile(&mut b),
                    self.kv_cache_append.compile(&mut b),
                    self.prefill_attention.compile(&mut b),
                    self.decode_attention.compile(&mut b),
                    self.o_proj.compile(&mut b),
                    self.post_norm.compile(&mut b),
                    self.up_gate_proj.compile(&mut b),
                    self.activation.compile(&mut b),
                    self.down_proj.compile(&mut b),
                ])),
            }),
        });
        if self.later_layers() > 0 {
            parts.push(CostNode::Scale {
                n: self.later_layers(),
                child: Box::new(self.next_input_norm.compile(&mut b)),
            });
        }
        parts.push(self.final_norm.compile(&mut b));
        parts.push(self.lm_head.compile(&mut b));
        let root = CostNode::Labeled {
            label: format!(
                "{} [dense local, vLLM kernels, {} layers]",
                self.name, self.num_layers
            ),
            child: Box::new(CostNode::Sum(parts)),
        };
        b.finish(root)
    }

    /// Leaves in [`cost_tree`](Self::cost_tree)'s slot order.
    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let g = &batch.groups[0];
        let tokens = g.batch_tokens;
        let rows = SingleGemmKernelInput { m: tokens };
        let norm = MultiAddRmsNormKernelInput { num_tokens: tokens };
        let prefill_tokens: u32 = g.prefill_chunk_pairs.iter().map(|&(_, new)| new).sum();
        let decodes = g.decode_kv_lens.len() as u32;
        push_or_zero(
            &self.first_norm,
            EmbeddingRmsNormKernelInput { num_tokens: tokens },
            false,
            ev,
        );
        push_or_zero(&self.qkv_proj, rows.clone(), false, ev);
        push_or_zero(&self.rope, GqaRopeNeoxKernelInput { num_tokens: tokens }, false, ev);
        push_or_zero(
            &self.kv_cache_append,
            KvCacheAppendKernelInput {
                num_tokens: prefill_tokens + decodes,
            },
            false,
            ev,
        );
        push_or_zero(
            &self.prefill_attention,
            PagedAttnBatchPrefillKernelInput {
                requests: g.prefill_chunk_pairs.clone(),
            },
            g.prefill_chunk_pairs.is_empty(),
            ev,
        );
        push_or_zero(
            &self.decode_attention,
            PagedAttnBatchDecodeKernelInput {
                batch_size: decodes,
                max_kv_len: g.decode_kv_lens.iter().copied().max().unwrap_or(0),
                total_kv_len: g.decode_kv_lens.iter().map(|&l| u64::from(l)).sum(),
            },
            decodes == 0,
            ev,
        );
        push_or_zero(&self.o_proj, rows.clone(), false, ev);
        push_or_zero(&self.post_norm, norm.clone(), false, ev);
        push_or_zero(&self.up_gate_proj, rows.clone(), false, ev);
        push_or_zero(&self.activation, SiluAndMulKernelInput { num_tokens: tokens }, false, ev);
        push_or_zero(&self.down_proj, rows, false, ev);
        if self.later_layers() > 0 {
            push_or_zero(&self.next_input_norm, norm.clone(), false, ev);
        }
        push_or_zero(&self.final_norm, norm, false, ev);
        push_or_zero(
            &self.lm_head,
            SingleGemmKernelInput {
                m: g.request_count(),
            },
            false,
            ev,
        );
    }
}

impl IterwiseUnifiedModel for Llama3DenseVllmModel {
    fn kv_cache_layout(&self) -> KvCacheLayout {
        KvCacheLayout::new().with_part(self.kv_cache_part)
    }

    fn gpus_per_replica(&self) -> u16 {
        1
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        assert_eq!(batch.groups.len(), 1, "llama3_dense_vllm has exactly one group");
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(ev.filled(), self.n_slots, "eval cursor must fill every slot");
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        assert_eq!(batch.groups.len(), 1, "llama3_dense_vllm has exactly one group");
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(ev.filled(), self.n_slots, "eval cursor must fill every slot");
        let agg = CostTree::aggregate(&self.cost_flat, slots, scratch);
        debug_assert_eq!(inputs.len(), self.n_slots, "slot_input must align to slots");
        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configs() -> Llama3DenseVllmConfigs {
        build_configs(
            &ModelCfg::llama3_8b(),
            &DenseParallel {
                gpu_name: "NVIDIA GB300".to_string(),
            },
            131072,
            &VllmServing {
                max_num_batched_tokens: 2048,
                rms_norm_eps: 1e-5,
            },
        )
    }

    #[test]
    fn gemms_keep_llama3_dense_shapes_on_the_linear_layout_only() {
        let c = configs();
        let shapes: Vec<(u32, u32)> = [&c.qkv_proj, &c.o_proj, &c.up_gate_proj, &c.down_proj, &c.lm_head]
            .iter()
            .map(|g| (g.n.get(), g.k.get()))
            .collect();
        assert_eq!(
            shapes,
            [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336), (128256, 4096)]
        );
        for g in [&c.qkv_proj, &c.o_proj, &c.up_gate_proj, &c.down_proj, &c.lm_head] {
            assert_eq!(g.backends, vec!["torch_linear"]);
            assert_eq!(g.dtype, DType::Bf16);
        }
    }

    #[test]
    fn norms_follow_the_compiled_fusions() {
        // (addends, store_sum, store_normed, store_fp8) of each traced Inductor
        // norm: post stores only the normed rows (its sum is re-added by the
        // next one), next-input stores the new residual and the normed rows,
        // final only the normed rows.
        let c = configs();
        let key = |n: &MultiAddRmsNormKernelConfig| (n.num_addends, n.store_sum, n.store_normed, n.store_fp8);
        assert_eq!(key(&c.post_norm), (2, false, true, false));
        assert_eq!(key(&c.next_input_norm), (3, true, true, false));
        assert_eq!(key(&c.final_norm), (3, false, true, false));
        for n in [&c.post_norm, &c.next_input_norm, &c.final_norm] {
            assert_eq!(n.backends, vec!["vllm_inductor"]);
            assert_eq!(n.hidden_dim.get(), 4096);
            // Serving's tuning size and Llama 3.1's epsilon, both keyed.
            assert_eq!(n.tune_tokens, Some(2048));
            assert_eq!(n.rms_eps_bits.map(f64::from_bits), Some(1e-5));
        }
    }

    #[test]
    fn first_norm_is_the_fused_embedding_kernel_not_a_plain_norm() {
        // Defect caught: pricing the first norm as a 1-addend RMSNorm (no
        // gather, no residual store), the configuration the runner rejects.
        let c = configs();
        let f = &c.first_norm;
        assert_eq!(f.backends, vec!["vllm_inductor"]);
        assert_eq!((f.hidden_dim.get(), f.vocab_size.get()), (4096, 128256));
        assert_eq!(f.dtype, DType::Bf16);
        assert_eq!(f64::from_bits(f.rms_eps_bits), 1e-5);
        assert_eq!(f.tune_tokens, 2048);
        let model = built(32);
        let manifest = model.cost_log_manifest();
        let first = manifest.slots.iter().find(|s| s.name == "llama.first_norm").unwrap();
        assert_eq!(first.kind, "embedding_rms_norm");
        assert!(manifest
            .slots
            .iter()
            .filter(|s| s.kind == "multi_add_rms_norm")
            .all(|s| s.kernel_config["num_addends"] != serde_json::json!(1)));
    }

    #[test]
    fn attention_kv_and_rope_match_the_trace() {
        let c = configs();
        // Packed FlashInfer cache; V read from the QKV row of (32 + 2 * 8) * 128.
        assert_eq!(c.kv_cache_append.cache_layout, "HND_PACKED");
        assert_eq!(c.kv_cache_append.value_row_stride, Some(6144));
        assert_eq!(c.kv_cache_append.block_size, 16);
        assert_eq!(c.decode_attention.page_size, Some(16));
        assert_eq!(c.prefill_attention.page_size, Some(16));
        assert_eq!((c.kv_cache_append.input_dtype, c.kv_cache_append.kv_dtype), (DType::Bf16, DType::Bf16));
        for (q, kv, o) in [
            (c.decode_attention.q_dtype, c.decode_attention.kv_dtype, c.decode_attention.o_dtype),
            (c.prefill_attention.q_dtype, c.prefill_attention.kv_dtype, c.prefill_attention.o_dtype),
        ] {
            assert_eq!((q, kv, o), (DType::Bf16, DType::Bf16, DType::Bf16));
        }
        assert_eq!(c.decode_attention.backends, vec!["flashinfer_trtllm_gen"]);
        assert_eq!((c.rope.num_qo_heads.get(), c.rope.num_kv_heads.get()), (32, 8));
        assert_eq!((c.rope.head_dim.get(), c.rope.rotary_dim.get(), c.rope.max_position), (128, 128, 131072));
        assert_eq!(c.activation.hidden_dim.get(), 14336);
        // Graph-replay rows, never the eager rows of the same kernel.
        assert_eq!(c.activation.backends, vec!["vllm_inductor_graph"]);
        assert_eq!(c.rope.backends, vec!["vllm_inductor_graph"]);
    }

    fn built(num_layers: u32) -> Llama3DenseVllmModel {
        let mut cfgs = configs();
        cfgs.num_layers = num_layers;
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build("llama".to_string(), cfgs, &bridge).unwrap()
    }

    /// Times each leaf is charged per iteration: aggregate the tree with a unit
    /// time in that slot alone.
    fn charges(model: &Llama3DenseVllmModel) -> Vec<(String, f32)> {
        let manifest = model.cost_log_manifest();
        let mut scratch = Vec::new();
        (0..model.n_slots)
            .map(|slot| {
                let mut buf = vec![LeafMetrics::ZERO; model.n_slots];
                buf[slot].m.time_ms = 1.0;
                let total = CostTree::aggregate(&model.cost_flat, &buf, &mut scratch);
                (manifest.slots[slot].name.clone(), total.m.time_ms)
            })
            .collect()
    }

    #[test]
    fn every_operation_is_charged_exactly_once_per_layer_or_iteration() {
        let model = built(32);
        let per_layer = [
            "qkv_proj",
            "rope",
            "kv_cache_append",
            "prefill_attention",
            "decode_attention",
            "o_proj",
            "post_norm",
            "up_gate_proj",
            "activation",
            "down_proj",
        ];
        let expected: Vec<(String, f32)> = std::iter::once(("llama.first_norm".to_string(), 1.0))
            .chain(per_layer.iter().map(|s| (format!("llama.layer.{s}"), 32.0)))
            .chain([
                ("llama.next_input_norm".to_string(), 31.0),
                ("llama.final_norm".to_string(), 1.0),
                ("llama.lm_head".to_string(), 1.0),
            ])
            .collect();
        assert_eq!(charges(&model), expected);
        // 32 input norms in all (first + 31 fused next-layer ones), 32 post
        // norms, one final norm: one per RMSNorm of the HF model. The
        // embedding is charged once, inside first_norm's fused kernel, and
        // has no leaf of its own.
        let manifest = model.cost_log_manifest();
        assert!(manifest.slots.iter().all(|s| !s.name.contains("embed")));
        assert_eq!(
            manifest.slots.iter().filter(|s| s.kind == "embedding_rms_norm").count(),
            1
        );
        let kinds: Vec<&str> = manifest.slots.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(kinds.iter().filter(|k| **k == "silu_and_mul").count(), 1);
        assert_eq!(kinds.iter().filter(|k| k.starts_with("paged_attn_batch")).count(), 2);
    }

    #[test]
    fn a_one_layer_model_has_no_next_input_norm() {
        let model = built(1);
        let names: Vec<String> = charges(&model).into_iter().map(|(n, c)| {
            assert_eq!(c, 1.0);
            n
        }).collect();
        assert_eq!(names.len(), 13);
        assert!(!names.iter().any(|n| n.ends_with("next_input_norm")));
    }

    #[test]
    fn kv_bytes_per_token_match_a_bf16_gqa_cache() {
        // 32 layers × (K + V) × 8 heads × 128 × 2 B.
        let model = built(32);
        assert_eq!(model.kv_cache_layout().per_token_bytes(), 32 * 2 * 8 * 128 * 2);
    }
}
