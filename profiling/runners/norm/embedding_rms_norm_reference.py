"""Independent value reference for the fused embedding gather + RMSNorm."""

from typing import Any


def embedding_rms_norm_reference(
    torch: Any, ids: Any, table: Any, weight: Any, eps: float
) -> tuple[Any, Any]:
    """Return (residual, normalized): the gathered rows and their RMSNorm, in the table dtype.

    Gathers with ``index_select`` rather than ``F.embedding``, and normalizes in
    float32, so it shares no code path with the block it checks.
    """
    rows = table.index_select(0, ids.to(torch.int64))
    x = rows.float()
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    return rows, normalized.to(table.dtype)


__all__ = ["embedding_rms_norm_reference"]
