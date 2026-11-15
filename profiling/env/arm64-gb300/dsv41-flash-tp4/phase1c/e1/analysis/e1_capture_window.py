#!/usr/bin/env python3
"""Capture-window coverage for the E1 nsys passes: iteration census and delayed-start simulation (CPU, read-only).

    python analysis/e1_capture_window.py census   --run <pass dir> [--out census.md] [--json census.json]
    python analysis/e1_capture_window.py simulate --run-root <stage15 run root> [--passes a,b,c] [--delays 0,2,3,3.6,3.7,4,10]
                                                  [--cell 8] [--duration SEC] [--out sim.md] [--json sim.json]

Timing model (alignment/runner.py::run_profile): `measurement_ready()` is called when the frontend is ready and the server
has drained. With `nsys.capture_start_delay_seconds` absent, `/start_profile` is called synchronously *inside*
`measurement_ready()`, before `replay_start_monotonic_ns` is taken and before the first request is released, so the capture
window starts in a drained server and covers the whole measured replay. With a delay d, a timer thread arms the profiler d
seconds after `measurement_ready()` returns (i.e. d seconds after `replay_start_monotonic_ns`, within the timer resolution),
and the capture then runs to the end of the replay (no bounded stop) or for `capture_duration_seconds`.

Every EngineCore iteration record in `<name>_metrics.jsonl` carries `observed_start/end_monotonic_ns` in the same
CLOCK_MONOTONIC domain as `replay_start_monotonic_ns` in `<name>_drive_summary.json`, so a pass's iterations can be placed on
the replay clock exactly, and a hypothetical window [d, end] can be checked against them without a GPU.

An iteration is *captured* when it lies entirely inside the window, *straddling* when the window start or end falls strictly
inside it (such an iteration would reach the parser without its full set of NVTX ranges and fail the 28-range gate), and
*outside* otherwise. Only `json`, `csv`, `re` are used so the module runs under any Python 3.10+.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field

TAG = re.compile(r"e1_b(\d+)_kv(\d+)k_w(\d+)_r(\d+)")


def parse_tag(request_id: str):
    m = TAG.search(request_id)
    return (int(m.group(1)), int(m.group(2)) * 1024, int(m.group(3))) if m else None


@dataclass
class Iteration:
    index: int
    start_s: float            # relative to replay_start_monotonic_ns
    end_s: float
    decode_requests: int
    prefill_tokens: int
    cells: tuple = ()         # distinct (b, kv) of the decode requests in flight (from decode_request_progress tags)
    waves: tuple = ()

    @property
    def kind(self) -> str:
        """prefill_only | mixed | pure_b<b> | decode_ramp_b<b> | overrun | idle_marker"""
        if len(self.cells) > 1:
            return "overrun"
        if self.prefill_tokens > 0 and self.decode_requests == 0:
            return "prefill_only"
        if self.prefill_tokens > 0:
            return "mixed" + (f"_b{self.cells[0][0]}" if self.cells else "")
        if self.decode_requests == 0:
            return "idle_marker"
        if self.cells and self.decode_requests == self.cells[0][0]:
            return f"pure_b{self.cells[0][0]}"
        return f"decode_ramp_b{self.cells[0][0]}" if self.cells else f"decode_ramp_untagged"

    @property
    def cell_label(self) -> str:
        return ",".join(f"b{b}_kv{kv // 1024}k" for b, kv in self.cells) or "-"


@dataclass
class PassTiming:
    name: str
    replay_start_ns: int
    replay_end_ns: int
    iterations: list = field(default_factory=list)

    @property
    def replay_span_s(self) -> float:
        return (self.replay_end_ns - self.replay_start_ns) / 1e9


def iteration_from_record(r: dict, replay_start_ns: int) -> Iteration:
    cells, waves = set(), set()
    for p in r.get("decode_request_progress") or []:
        t = parse_tag(p["request_id"])
        if t:
            cells.add((t[0], t[1])); waves.add(t[2])
    return Iteration(index=r["iteration_index"],
                     start_s=(r["observed_start_monotonic_ns"] - replay_start_ns) / 1e9,
                     end_s=(r["observed_end_monotonic_ns"] - replay_start_ns) / 1e9,
                     decode_requests=r.get("decode_requests", 0), prefill_tokens=r.get("prefill_tokens", 0) or 0,
                     cells=tuple(sorted(cells)), waves=tuple(sorted(waves)))


def load_pass(run_dir: str) -> PassTiming:
    """Timeline of one pass from its drive summary and metrics jsonl (`<run dir>/profile/vllm/`)."""
    ds_path = glob.glob(os.path.join(run_dir, "profile", "vllm", "*_drive_summary.json"))
    me_path = glob.glob(os.path.join(run_dir, "profile", "vllm", "*_metrics.jsonl"))
    if not ds_path or not me_path:
        raise FileNotFoundError(f"drive summary / metrics jsonl missing under {run_dir}/profile/vllm")
    ds = json.load(open(ds_path[0]))
    rs, re_ = ds["replay_start_monotonic_ns"], ds["replay_end_monotonic_ns"]
    recs = [json.loads(l) for l in open(me_path[0]) if l.strip()]
    recs.sort(key=lambda r: r["iteration_index"])
    return PassTiming(os.path.basename(run_dir.rstrip("/")), rs, re_, [iteration_from_record(r, rs) for r in recs])


# ----------------------------------------------------------------------------------------------------------------------
def window(delay_s: float | None, replay_span_s: float, duration_s: float | None = None) -> tuple[float, float]:
    """[start, end] of the capture on the replay clock. No delay: the profiler is armed before the first request
    (start = -inf, i.e. everything from the drained server on). No duration: the capture runs to the end of the replay
    and `/stop_profile` follows the post-replay idle wait, so end = +inf relative to the measured iterations."""
    start = float("-inf") if delay_s is None else float(delay_s)
    end = float("inf") if duration_s is None else max(start, 0.0) + float(duration_s)
    return start, end


def coverage(iterations: list, start_s: float, end_s: float) -> dict:
    """Per-kind counts of captured / straddling / outside iterations for the window [start_s, end_s]."""
    cap, strad, out, total = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    first_cap, last_cap, strad_idx = None, None, []
    for it in iterations:
        k = it.kind; total[k] += 1
        inside = start_s <= it.start_s and it.end_s <= end_s
        straddles = (it.start_s < start_s < it.end_s) or (it.start_s < end_s < it.end_s)
        if inside:
            cap[k] += 1
            first_cap = it.index if first_cap is None else first_cap; last_cap = it.index
        elif straddles:
            strad[k] += 1; strad_idx.append(it.index)
        else:
            out[k] += 1
    return {"window": [start_s, end_s], "total": dict(total), "captured": dict(cap), "straddling": dict(strad), "outside": dict(out),
            "first_captured": first_cap, "last_captured": last_cap, "straddling_indices": strad_idx,
            "n_captured": sum(cap.values()), "n_straddling": sum(strad.values()), "n_total": len(iterations)}


def cell_gate(cov: dict, b: int, need: int) -> tuple[bool, str]:
    """PASS iff every pure b=<b> iteration is captured, nothing straddles, and the count meets the plan minimum."""
    k = f"pure_b{b}"
    tot, cap = cov["total"].get(k, 0), cov["captured"].get(k, 0)
    ok = tot > 0 and cap == tot and cov["n_straddling"] == 0 and cap >= need
    return ok, f"pure b={b}: captured {cap} of {tot} (need >= {need}); straddling iterations {cov['n_straddling']} {cov['straddling_indices'][:4]}"


def wave_bounds(iterations: list, b: int) -> dict:
    """Replay-clock positions of the b=<b> wave: first iteration owning its requests, first/last pure iteration, wave end."""
    own = [it for it in iterations if it.cells and all(c[0] == b for c in it.cells)]
    pure = [it for it in own if it.kind == f"pure_b{b}"]
    if not own:
        return {}
    # prefill-only iterations immediately preceding the first owned iteration belong to the wave (first chunks)
    idx = {it.index: n for n, it in enumerate(iterations)}
    n0 = idx[own[0].index]
    while n0 > 0 and iterations[n0 - 1].kind == "prefill_only":
        n0 -= 1
    return {"first_iteration": iterations[n0].index, "first_start_s": iterations[n0].start_s,
            "first_pure_iteration": pure[0].index if pure else None, "first_pure_start_s": pure[0].start_s if pure else None,
            "last_pure_iteration": pure[-1].index if pure else None, "last_pure_end_s": pure[-1].end_s if pure else None,
            "wave_end_s": own[-1].end_s, "n_pure": len(pure), "n_owned": len(own) + (idx[own[0].index] - n0)}


# ----------------------------------------------------------------------------------------------------------------------
def census(run_dir: str, parsed_only: bool = True) -> dict:
    """Classify every iteration of a pass; with parsed.json present, also the captured (parsed) subset and its JIT iterations."""
    pt = load_pass(run_dir)
    by = {it.index: it for it in pt.iterations}
    out = {"pass": pt.name, "replay_span_s": round(pt.replay_span_s, 3), "metrics_iterations": len(pt.iterations),
           "metrics_index_range": [pt.iterations[0].index, pt.iterations[-1].index],
           "kinds_all": dict(collections.Counter(it.kind for it in pt.iterations)),
           "waves": {f"b{b}": wave_bounds(pt.iterations, b) for b in sorted({c[0] for it in pt.iterations for c in it.cells})}}
    pj = os.path.join(run_dir, "profile", "parsed.json")
    if os.path.exists(pj):
        d = json.load(open(pj)); det = d["iteration_details"]
        its = sorted(x["iteration"] for x in det)
        missing = [i for i in its if i not in by]
        kinds = collections.Counter(by[i].kind for i in its if i in by)
        owners = collections.Counter(by[i].cell_label for i in its if i in by)
        stage = collections.Counter(x.get("stage") for x in det)
        jit = [(x["iteration"], by[x["iteration"]].kind if x["iteration"] in by else "?") for x in det if x.get("jit_module_loads")]
        first = by.get(its[0]); prev = by.get(its[0] - 1) or by.get(its[0] - 2) or by.get(its[0] - 3)
        out["parsed"] = {"iterations": len(its), "index_range": [its[0], its[-1]], "missing_in_metrics": missing,
                         "contiguous": len(its) == its[-1] - its[0] + 1, "kinds": dict(kinds), "owners": dict(owners),
                         "parser_stage": dict(stage), "jit_iterations": jit,
                         "first_parsed_start_s": round(first.start_s, 4) if first else None,
                         "previous_metrics_iteration": {"index": prev.index, "end_s": round(prev.end_s, 4), "kind": prev.kind} if prev else None,
                         "nsys": json.load(open(os.path.join(run_dir, "profile.json"))).get("nsys") if os.path.exists(os.path.join(run_dir, "profile.json")) else None}
    return out


def render_census(c: dict) -> str:
    L = [f"# Iteration census: {c['pass']}", "",
         f"Replay span {c['replay_span_s']} s; EngineCore metrics iterations {c['metrics_iterations']} "
         f"(index {c['metrics_index_range'][0]}..{c['metrics_index_range'][1]}).", "", "| kind (all metrics iterations) | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(c["kinds_all"].items())]
    L += ["", "| wave | first iteration (start s) | first pure (start s) | last pure (end s) | wave end s | pure n | owned n |", "|---|---|---|---|---|---|---|"]
    for w, b in c["waves"].items():
        if b:
            L.append(f"| {w} | {b['first_iteration']} ({b['first_start_s']:.3f}) | {b['first_pure_iteration']} ({b['first_pure_start_s']:.3f}) | "
                     f"{b['last_pure_iteration']} ({b['last_pure_end_s']:.3f}) | {b['wave_end_s']:.3f} | {b['n_pure']} | {b['n_owned']} |")
    p = c.get("parsed")
    if p:
        L += ["", f"## Captured (parsed.json): {p['iterations']} iterations, index {p['index_range'][0]}..{p['index_range'][1]}, "
              f"contiguous={p['contiguous']}, missing in metrics={p['missing_in_metrics']}", "",
              f"nsys block: `{json.dumps(p['nsys'])}`; first captured iteration starts at {p['first_parsed_start_s']} s on the replay clock; "
              f"previous metrics iteration {p['previous_metrics_iteration']}", "",
              "| kind | captured n |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in sorted(p["kinds"].items())]
        L += ["", "| request ownership (cells of the decode requests in flight) | n |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in sorted(p["owners"].items())]
        L += ["", f"parser stage field: {p['parser_stage']}", f"JIT-affected captured iterations (index, kind): {p['jit_iterations']}"]
    return "\n".join(L) + "\n"


def simulate(run_root: str, passes: list, delays: list, cell: int, need: int, duration: float | None) -> dict:
    res = {"cell": cell, "need": need, "duration_s": duration, "passes": {}}
    for p in passes:
        pt = load_pass(os.path.join(run_root, p))
        rows = []
        for d in delays:
            s, e = window(d, pt.replay_span_s, duration)
            cov = coverage(pt.iterations, s, e)
            ok, msg = cell_gate(cov, cell, need)
            rows.append({"delay_s": d, "window": [s, e], "gate": "PASS" if ok else "FAIL", "msg": msg,
                         "captured_total": cov["n_captured"], "first_captured": cov["first_captured"], "last_captured": cov["last_captured"],
                         "captured_by_kind": cov["captured"], "straddling_by_kind": cov["straddling"]})
        res["passes"][p] = {"replay_span_s": pt.replay_span_s, "wave": wave_bounds(pt.iterations, cell), "rows": rows}
    return res


def render_sim(r: dict) -> str:
    L = [f"# Delayed-start simulation for the b={r['cell']} cell (need >= {r['need']} pure iterations captured, 0 straddling)", ""]
    for p, v in r["passes"].items():
        w = v["wave"]
        L += [f"## {p}: replay span {v['replay_span_s']:.3f} s; b={r['cell']} wave first iteration {w.get('first_iteration')} at {w.get('first_start_s', 0):.3f} s, "
              f"first pure {w.get('first_pure_iteration')} at {w.get('first_pure_start_s', 0):.3f} s, last pure end {w.get('last_pure_end_s', 0):.3f} s, pure n={w.get('n_pure')}", "",
              "| delay s | window | gate | pure captured | straddling | captured total | first..last captured |", "|---|---|---|---|---|---|---|"]
        for row in v["rows"]:
            k = f"pure_b{r['cell']}"
            L.append(f"| {'none' if row['delay_s'] is None else row['delay_s']} | [{row['window'][0]}, {row['window'][1]}] | {row['gate']} | "
                     f"{row['captured_by_kind'].get(k, 0)} | {sum(row['straddling_by_kind'].values())} {dict(row['straddling_by_kind']) or ''} | "
                     f"{row['captured_total']} | {row['first_captured']}..{row['last_captured']} |")
        L.append("")
    return "\n".join(L) + "\n"


def _delays(s: str) -> list:
    out = []
    for x in s.split(","):
        x = x.strip()
        out.append(None if x in ("none", "None", "") else float(x))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census"); c.add_argument("--run", required=True); c.add_argument("--out"); c.add_argument("--json")
    s = sub.add_parser("simulate"); s.add_argument("--run-root", required=True)
    s.add_argument("--passes", default="e1_dry_off_v2,e1_dry_on_v2,e1_dry_off2_v2,e1_nsys_val")
    s.add_argument("--delays", default="none,0,1,2,3,3.5,3.6,3.7,4,10"); s.add_argument("--cell", type=int, default=8)
    s.add_argument("--need", type=int, default=400); s.add_argument("--duration", type=float); s.add_argument("--out"); s.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "census":
        r = census(a.run); text = render_census(r)
    else:
        r = simulate(a.run_root, a.passes.split(","), _delays(a.delays), a.cell, a.need, a.duration); text = render_sim(r)
    print(text, end="")
    if a.out:
        open(a.out, "w").write(text)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
