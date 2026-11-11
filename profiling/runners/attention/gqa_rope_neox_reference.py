"""Independent value reference for GQA RoPE (NeoX) on a fused QKV projection output.

Written from the definition, not from vLLM: the query heads are the first
``Hq * D`` columns of each QKV row, the key heads the next ``Hkv * D``; in every
head the first ``rotary_dim`` elements rotate in NeoX halves, element ``i`` with
``i + rotary_dim / 2``, by the angle the cos/sin cache row of the token's
position holds (``cache[pos, :r/2]`` cosines, ``cache[pos, r/2:r]`` sines);
the rest of the head passes through. Computed in float32, returned in the
input dtype as contiguous ``[tokens, Hq * D]`` and ``[tokens, Hkv * D]``.
"""

from __future__ import annotations

from typing import Any


def gqa_rope_neox_reference(
    torch: Any,
    positions: Any,
    qkv: Any,
    cos_sin_cache: Any,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[Any, Any]:
    half = rotary_dim // 2
    tokens = qkv.shape[0]
    q = qkv[:, : num_qo_heads * head_dim].float().reshape(tokens, num_qo_heads, head_dim)
    k = (
        qkv[:, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim]
        .float()
        .reshape(tokens, num_kv_heads, head_dim)
    )
    rows = cos_sin_cache.float()[positions.long()]  # [tokens, rotary_dim]
    cos = rows[:, None, :half]
    sin = rows[:, None, half:rotary_dim]

    def rotate(x: Any) -> Any:
        x1, x2 = x[..., :half], x[..., half:rotary_dim]
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)

    out_q = rotate(q).reshape(tokens, num_qo_heads * head_dim).to(qkv.dtype)
    out_k = rotate(k).reshape(tokens, num_kv_heads * head_dim).to(qkv.dtype)
    return out_q.contiguous(), out_k.contiguous()


def llama3_cos_sin_cache(
    torch: Any,
    rotary_dim: int,
    max_position: int,
    base: float = 500000.0,
    factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
) -> Any:
    """Llama 3 scaled inverse frequencies, as the published rope_scaling defines them."""
    import math

    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    low_wavelen = original_max_position / low_freq_factor
    high_wavelen = original_max_position / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    smooth = (original_max_position / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    scaled = torch.where(
        wavelen > low_wavelen,
        inv_freq / factor,
        torch.where(
            wavelen < high_wavelen,
            inv_freq,
            (1 - smooth) * inv_freq / factor + smooth * inv_freq,
        ),
    )
    angles = torch.arange(max_position, dtype=torch.float64)[:, None] * scaled[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).float()


__all__ = ["gqa_rope_neox_reference", "llama3_cos_sin_cache"]
