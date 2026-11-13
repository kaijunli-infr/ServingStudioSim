"""Component 3: MoE expert-GEMM statistical cost (TP-sharded experts, no EP, on this GB300 config).

Per-iteration expert routing was not captured by the nsys passes (fused TRT-LLM MXFP4 grouped GEMM;
routes exist only in the separate token-corpus pass), so the expert-count dependence of the decode
MoE kernel time is a statistical model:

    t_moe(b) = α + β · E(b_pad) + γ · b_pad    (per rank, ns)

where b_pad is the cudagraph capture size the step runs with (1, 2, 4, 8, 16, 24, 32, 40, 48: padded
tokens are routed too, so the kernel sees b_pad tokens — the measured MoE time is flat inside a
bucket) and E(b_pad) is the expected number of distinct experts touched per layer by b_pad tokens,
estimated by drawing random tokens from the calibration requests of the corpus.  The
uncertainty range is obtained by propagating the p10–p90 of E(b) through the fitted β, and a
distribution-shift check re-estimates E(b) from the held-out corpus requests (different routing
entropy / imbalance).  The model is a central estimate with a range; it does not predict the
routing of any individual iteration (gap G3).

For prefill (16k chunks) nearly all 384 experts are touched; the MoE group time is flat in KV and
handled by the prefill compute model (linear in tokens).  The B200 EP4 preset is a different
execution structure (experts partitioned across ranks) and shares no coefficient with this model.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import groups as G
from .data import padded_batch

BATCH_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25, 26, 27, 30, 32, 36, 40, 48, 64, 128, 256)


class Corpus:
    def __init__(self, token_corpus_dir: str):
        man = json.load(open(os.path.join(token_corpus_dir, "manifest.json")))
        prov = json.load(open(os.path.join(token_corpus_dir, "provenance.json")))
        self.T, self.L, self.K, self.E = man["num_tokens"], man["num_layers"], man["top_k"], man["num_experts"]
        r = np.fromfile(os.path.join(token_corpus_dir, man["data_file"]), dtype="<u2")
        assert r.size == self.T * self.L * self.K, (r.size, self.T * self.L * self.K)
        self.routes = r.reshape(self.T, self.L, self.K)
        self.segments = {s["request"]: (s["start"], s["length"]) for s in prov["request_segments"]}
        self.manifest = man

    def token_index(self, requests: list[str]) -> np.ndarray:
        parts = [np.arange(a, a + n) for req, (a, n) in self.segments.items() if req in requests and n > 0]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def expert_stats(corpus: Corpus, tokens: np.ndarray, b: int, draws: int, rng: np.random.Generator, layer_step: int = 1) -> dict:
    """Distinct experts per layer and max tokens per expert for random b-token batches from `tokens`."""
    L = corpus.L
    layers = range(0, L, layer_step)
    dis, mx = [], []
    for _ in range(draws):
        idx = tokens[rng.integers(0, len(tokens), b)]
        x = corpus.routes[idx]
        dd = [len(np.unique(x[:, l, :])) for l in layers]
        dis.append(float(np.mean(dd)))
        mx.append(float(np.mean([np.bincount(x[:, l, :].ravel(), minlength=corpus.E).max() for l in layers])))
    dis = np.array(dis)
    mx = np.array(mx)
    return dict(b=b, E_mean=float(dis.mean()), E_p10=float(np.percentile(dis, 10)), E_p50=float(np.percentile(dis, 50)),
                E_p90=float(np.percentile(dis, 90)), E_std=float(dis.std()), max_tokens_per_expert_mean=float(mx.mean()), draws=draws)


class MoeModel:
    def __init__(self):
        self.E_cal: dict[int, dict] = {}
        self.E_held: dict[int, dict] = {}
        self.coef_ns: list | None = None        # [alpha, beta, gamma] per rank-median
        self.fit_rows: list = []
        self.rmse_ns = float("nan")
        self.r2 = float("nan")
        self.corpus_info: dict = {}
        self.raw_b_fit: dict = {}

    def fit(self, corpus: Corpus, cal_requests: list[str], held_requests: list[str], decode_table: dict[int, dict], seed: int = 0,
            draws: int = 200) -> "MoeModel":
        rng = np.random.default_rng(seed)
        cal_tok = corpus.token_index(cal_requests)
        held_tok = corpus.token_index(held_requests)
        self.corpus_info = dict(tokens=corpus.T, layers=corpus.L, top_k=corpus.K, experts=corpus.E, calibration_tokens=int(len(cal_tok)),
                                heldout_tokens=int(len(held_tok)), calibration_requests=len(cal_requests), heldout_requests=len(held_requests),
                                seed=seed, draws=draws)
        grid = sorted(set(BATCH_GRID) | set(decode_table) | {padded_batch(b) for b in decode_table})
        for b in grid:
            step = 1 if b <= 64 else 5
            d = draws if b <= 64 else max(20, draws // 5)
            self.E_cal[b] = expert_stats(corpus, cal_tok, b, d, rng, step)
            if len(held_tok):
                self.E_held[b] = expert_stats(corpus, held_tok, b, d, rng, step)
        # fit on the decode table (calibration batch sizes only, by construction of the table)
        bs = sorted(decode_table)
        y = np.array([np.median(decode_table[b]["groups"][:, G.GROUP_INDEX["moe"]]) for b in bs], dtype=np.float64)
        bp = [padded_batch(b) for b in bs]
        X = np.stack([np.ones(len(bs)), [self.E_cal[b]["E_mean"] for b in bp], bp], axis=1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        self.coef_ns = [float(c) for c in coef]
        self.rmse_ns = float(np.sqrt(np.mean(resid ** 2)))
        self.r2 = float(1 - resid.var() / y.var()) if y.var() > 0 else float("nan")
        self.fit_rows = [dict(b=b, b_pad=padded_batch(b), E_mean=self.E_cal[padded_batch(b)]["E_mean"], moe_ns=float(yy), pred_ns=float(pp), n=decode_table[b]["n"]) for b, yy, pp in zip(bs, y, X @ coef)]
        # comparison: the same fit on the raw batch size (reported, not used)
        Xr = np.stack([np.ones(len(bs)), [self.E_cal[b]["E_mean"] for b in bs], bs], axis=1).astype(np.float64)
        cr, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        rr = y - Xr @ cr
        self.raw_b_fit = dict(coef_ns=[float(c) for c in cr], r2=float(1 - rr.var() / y.var()) if y.var() > 0 else float("nan"), rmse_ns=float(np.sqrt(np.mean(rr ** 2))))
        return self

    def predict(self, b: int) -> dict:
        a, be, ga = self.coef_ns
        bp = padded_batch(b)
        E = self.E_cal.get(bp)
        status = "in-range" if (E is not None and min(r["b"] for r in self.fit_rows) <= b <= max(r["b"] for r in self.fit_rows)) else "extrapolation"
        if E is None:
            return dict(b=b, b_pad=bp, status="extrapolation", note="no E(b) estimate for this padded batch (outside the corpus grid)")
        central = a + be * E["E_mean"] + ga * bp
        lo = a + be * min(E["E_p10"], E["E_p90"]) + ga * bp
        hi = a + be * max(E["E_p10"], E["E_p90"]) + ga * bp
        out = dict(b=b, b_pad=bp, status=status, central_ns=central, low_ns=min(lo, hi), high_ns=max(lo, hi), E_mean=E["E_mean"], E_p10=E["E_p10"], E_p90=E["E_p90"])
        if bp in self.E_held:
            Eh = self.E_held[bp]
            out["heldout_routing_central_ns"] = a + be * Eh["E_mean"] + ga * bp
            out["heldout_E_mean"] = Eh["E_mean"]
        return out

    def to_dict(self) -> dict:
        return dict(coef_ns=self.coef_ns, rmse_ns=self.rmse_ns, r2=self.r2, fit_rows=self.fit_rows, corpus=self.corpus_info, raw_b_fit=self.raw_b_fit,
                    E_calibration={str(b): v for b, v in self.E_cal.items()}, E_heldout={str(b): v for b, v in self.E_held.items()},
                    status={"alpha,beta,gamma": "inferred (fit to batch-level kernel medians through corpus-derived E(b))",
                            "E(b)": "measured on the corpus pass (different pass from the timing passes)",
                            "per-iteration routing": "not identifiable (not captured)",
                            "padded batch": "measured structure (compute steps at b = 3/4, 5–8, 9–16 are flat)"})

    @classmethod
    def from_dict(cls, d: dict) -> "MoeModel":
        m = cls()
        m.coef_ns, m.rmse_ns, m.r2, m.fit_rows, m.corpus_info = d["coef_ns"], d["rmse_ns"], d["r2"], d["fit_rows"], d["corpus"]
        m.raw_b_fit = d.get("raw_b_fit", {})
        m.E_cal = {int(k): v for k, v in d["E_calibration"].items()}
        m.E_held = {int(k): v for k, v in d["E_heldout"].items()}
        return m
