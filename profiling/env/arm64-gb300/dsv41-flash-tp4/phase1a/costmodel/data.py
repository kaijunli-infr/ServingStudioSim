"""Loaders for the compact tables written by `extract.py`, plus per-iteration feature derivation.

Everything here is read-only.  A `PassData` holds the three tables of one nsys pass and the
partition roles of its iterations; `features()` derives the model inputs (stage, batch, regime,
prefill tokens, chunk pairs, kv offsets, chain id) that every component consumes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

from . import groups as G

NRANK = 4
GRAPH_CAP_TOKENS = 256          # vLLM cudagraph capture cap (FULL_AND_PIECEWISE) in the frozen config
NCCL_RING_MIN_TOKENS = 8192     # observed switch from cute-DSL twoshot to NCCL ring all-reduce (total tokens P + b; 7216 twoshot, 8274 ring)
CHUNK_FULL_TOKENS = 16000       # a "clean 16k chunk" has one prefill chunk of >= this many tokens


def decode_regime(b: int) -> str:
    return "b<=16" if b <= 16 else ("b17-24" if b <= 24 else "b>=25")


@dataclass
class Feat:
    iteration: int
    stage: str                 # decode | mixed
    excluded: bool
    b: int                     # decode requests
    kv_max: int
    kv_sum: int
    P: int                     # prefill tokens
    chunks: list               # [[kv0, tokens], ...]
    wall_ns: int
    t0_ns: int
    t1_ns: int
    # derived
    regime: str = ""           # decode regime or mixed regime
    mixed_kind: str = ""       # graph | eager | ring (mixed only)
    is_chunk16k: bool = False
    kv0: int = 0               # kv offset of the single 16k chunk
    chain: int = -1
    role: str = "excluded"
    cls: str = ""
    lead_ns: int = 0           # previous iteration's GPU end − this iteration's earliest host start (>0: host ran ahead)
    prev_stage: str = ""

    @property
    def total_tokens(self) -> int:
        return self.P + self.b

    @property
    def sum_t2(self) -> float:
        return float(sum(c[1] ** 2 for c in self.chunks))

    @property
    def kv_weighted(self) -> float:
        """Σ tokens_i · kv0_i / 16384: KV depth seen by the indexer, in 16k-chunk-equivalents."""
        return float(sum(c[1] * c[0] for c in self.chunks)) / 16384.0


def mixed_kind(P: int, b: int) -> str:
    if P + b <= GRAPH_CAP_TOKENS:
        return "graph"
    return "ring" if P + b >= NCCL_RING_MIN_TOKENS else "eager"


def padded_batch(b: int) -> int:
    """vLLM cudagraph capture size used for a decode batch of b tokens (1, 2, 4, 8, 16, 24, ...)."""
    if b <= 2:
        return b
    if b <= 4:
        return 4
    return int(-(-b // 8) * 8)


@dataclass
class PassData:
    label: str
    table_dir: str
    meta: dict
    it: dict                   # column -> np.ndarray (iterations.parquet)
    ir: dict                   # column -> np.ndarray (iter_rank.parquet), sorted by (iteration, rank)
    feats: list = field(default_factory=list)
    _ir_index: dict = field(default_factory=dict)
    _coll: dict | None = None
    _coll_bounds: np.ndarray | None = None
    _coll_imin: int = 0

    # ---- construction --------------------------------------------------------------------------
    @classmethod
    def load(cls, label: str, table_dir: str, partition=None) -> "PassData":
        meta = json.load(open(os.path.join(table_dir, "meta.json")))
        it_t = pq.read_table(os.path.join(table_dir, "iterations.parquet")).to_pydict()
        it = {k: (np.array(v) if k not in ("chunk_pairs", "stage") else list(v)) for k, v in it_t.items()}
        ir_t = pq.read_table(os.path.join(table_dir, "iter_rank.parquet")).to_pydict()
        ir = {k: np.array(v) for k, v in ir_t.items()}
        o = np.lexsort((ir["rank"], ir["iteration"]))
        ir = {k: v[o] for k, v in ir.items()}
        pd = cls(label=label, table_dir=table_dir, meta=meta, it=it, ir=ir)
        pd._ir_index = {(int(a), int(r)): i for i, (a, r) in enumerate(zip(ir["iteration"], ir["rank"]))}
        pd.feats = features(pd, partition)
        return pd

    # ---- accessors -----------------------------------------------------------------------------
    def rank_col(self, col: str, iteration: int) -> np.ndarray:
        return np.array([self.ir[col][self._ir_index[(iteration, r)]] for r in range(NRANK)])

    def rank_rows(self, iteration: int) -> np.ndarray:
        return np.array([self._ir_index[(iteration, r)] for r in range(NRANK)])

    def group_sums(self, iteration: int) -> np.ndarray:
        """(NRANK, len(GROUPS)) kernel-duration sums in ns."""
        rows = self.rank_rows(iteration)
        return np.stack([self.ir[f"g_{g}_sum_ns"][rows] for g in G.GROUPS], axis=1)

    def comm_kind_counts(self, iteration: int) -> np.ndarray:
        rows = self.rank_rows(iteration)
        return np.stack([self.ir[f"c_{k}_count"][rows] for k in G.COMM_KINDS], axis=1)

    def load_collectives(self):
        if self._coll is not None:
            return
        t = pq.read_table(os.path.join(self.table_dir, "collectives.parquet")).to_pydict()
        C = {k: np.array(v) for k, v in t.items()}
        o = np.lexsort((C["k"], C["rank"], C["iteration"]))
        C = {k: v[o] for k, v in C.items()}
        self._coll = C
        self._coll_imin = int(C["iteration"].min())
        self._coll_bounds = np.searchsorted(C["iteration"], np.arange(self._coll_imin, int(C["iteration"].max()) + 2))

    def collectives(self, iteration: int) -> dict | None:
        """Per-rank aligned collective calls of one iteration: dict of (NRANK, K) arrays, or None."""
        self.load_collectives()
        j = iteration - self._coll_imin
        if j < 0 or j + 1 >= len(self._coll_bounds):
            return None
        lo, hi = int(self._coll_bounds[j]), int(self._coll_bounds[j + 1])
        if hi <= lo or (hi - lo) % NRANK:
            return None
        K = (hi - lo) // NRANK
        out = {k: v[lo:hi].reshape(NRANK, K) for k, v in self._coll.items()}
        if not (out["kind"] == out["kind"][0]).all():
            out["_kind_mismatch"] = True
        return out

    def segments(self, iteration: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Per-rank ordered timeline of one iteration.

        Returns (start_rel (R,), seg_len (R, K+1), intrinsic (K,), coll_dur (R, K)) in ns, where
        seg_len[:, k] is the window from the end of collective k-1 (or the first kernel) to the start
        of collective k and seg_len[:, K] is the tail after the last collective.
        """
        c = self.collectives(iteration)
        if c is None:
            return None
        start_rel = self.rank_col("first_start_ns", iteration) - self.it_value("t0_ns", iteration)
        K = c["k"].shape[1]
        seg = np.concatenate([c["seg_len_ns"], c["tail_len_ns"][:, :1]], axis=1)
        dur = c["end_rel_ns"] - c["start_rel_ns"]
        intrinsic = dur.min(axis=0)
        return start_rel.astype(np.int64), seg.astype(np.int64), intrinsic.astype(np.int64), dur.astype(np.int64)

    def it_value(self, col: str, iteration: int):
        idx = self._it_index(iteration)
        return self.it[col][idx]

    def _it_index(self, iteration: int) -> int:
        if not hasattr(self, "_it_pos"):
            self._it_pos = {int(a): i for i, a in enumerate(self.it["iteration"])}
        return self._it_pos[iteration]


def chains(feats: list[Feat]) -> dict[int, int]:
    """Chain id per clean-16k-chunk iteration (same rule as the planning split: consecutive chunks of one
    request, kv0 == previous kv0 + previous tokens, within 3 iterations)."""
    rows = sorted((f.iteration, f.chunks[0][0], f.chunks[0][1]) for f in feats if f.is_chunk16k)
    out, cid, last = {}, 0, None
    for it, kv0, ln in rows:
        if last and kv0 == last[1] + last[2] and it - last[0] <= 3:
            out[it] = cid
        else:
            cid += 1
            out[it] = cid
        last = (it, kv0, ln)
    return out


def features(pd: PassData, partition=None) -> list[Feat]:
    it = pd.it
    feats = []
    for i in range(len(it["iteration"])):
        chunks = json.loads(it["chunk_pairs"][i])
        f = Feat(iteration=int(it["iteration"][i]), stage=it["stage"][i], excluded=bool(it["excluded"][i]),
                 b=int(it["decode_requests"][i]), kv_max=int(it["kv_max"][i]), kv_sum=int(it["kv_sum"][i]),
                 P=int(it["prefill_tokens"][i]), chunks=chunks, wall_ns=int(it["wall_ns"][i]),
                 t0_ns=int(it["t0_ns"][i]), t1_ns=int(it["t1_ns"][i]))
        if f.stage == "decode":
            f.regime = decode_regime(f.b)
        else:
            f.mixed_kind = mixed_kind(f.P, f.b)
            f.regime = f"mixed-{f.mixed_kind}"
            if len(chunks) == 1 and chunks[0][1] >= CHUNK_FULL_TOKENS:
                f.is_chunk16k = True
                f.kv0 = int(chunks[0][0])
        if partition is not None and pd.label in partition.role:
            f.role = partition.role_of(pd.label, f.iteration)
            f.cls = partition.class_of(pd.label, f.iteration)
        feats.append(f)
    ch = chains(feats)
    for f in feats:
        f.chain = ch.get(f.iteration, -1)
    # host lead over the GPU (needs the previous iteration in the same capture)
    by_it = {f.iteration: f for f in feats}
    for f in feats:
        prev = by_it.get(f.iteration - 1)
        if prev is None:
            continue
        hs = int(min(pd.ir["host_start_ns"][pd._ir_index[(f.iteration, r)]] for r in range(NRANK)))
        f.lead_ns = prev.t1_ns - hs
        f.prev_stage = prev.stage
    return feats
