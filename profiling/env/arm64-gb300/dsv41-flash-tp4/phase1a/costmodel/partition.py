"""Frozen calibration / held-out partition (component 8 guard rails).

`materialize()` flattens the accepted planning split (`phase1/out/validation_split.json`,
sha256 c2376e92…, branch phase1-planning @ cc06d9c2) plus the MoE routing hold-out into
`partition/partition.json` and records its sha256 in `partition/PARTITION.sha256`.  It is run once,
before any fitting; afterwards `load()` refuses a partition whose hash differs from the frozen one.

`Partition.calibration_only(label, ids)` is the leakage guard used by every fitter: it raises
`LeakageError` when a held-out or excluded iteration id would enter a fit.
"""
from __future__ import annotations

import hashlib
import json
import os

MOE_HOLDOUT_MIN_TOKENS = 100      # requests shorter than this are trivially imbalanced (2-token requests)
MOE_HOLDOUT_EACH_SIDE = 5

ROLE_CAL, ROLE_HELD, ROLE_EXCL = "calibration", "heldout", "excluded"


class LeakageError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize(split_json: str, routing_json: str, out_dir: str) -> dict:
    split = json.load(open(split_json))
    routing = json.load(open(routing_json))
    passes = {}
    for label, v in split.items():
        role_of, cls_of = {}, {}
        for cls, roles in v["classes"].items():
            for role in (ROLE_CAL, ROLE_HELD, ROLE_EXCL):
                for it in roles[role]:
                    if it in role_of:
                        raise ValueError(f"{label}: iteration {it} appears twice in the split")
                    role_of[it] = role
                    cls_of[it] = cls
        passes[label] = dict(
            role=dict(sorted(((str(k), r) for k, r in role_of.items()), key=lambda kv: int(kv[0]))),
            cls=dict(sorted(((str(k), c) for k, c in cls_of.items()), key=lambda kv: int(kv[0]))),
            exclusions=v["exclusions"], held_out_batch_sizes=v["held_out_batch_sizes"],
            held_out_chains=v["held_out_chains"], held_out_kv0_band=v["held_out_kv0_band"],
            counts={r: sum(1 for x in role_of.values() if x == r) for r in (ROLE_CAL, ROLE_HELD, ROLE_EXCL)},
        )
    reqs = [z for z in routing["per_request"] if z["tokens"] >= MOE_HOLDOUT_MIN_TOKENS]
    reqs.sort(key=lambda z: z["imb_mean"])
    moe_held = [z["request"] for z in reqs[:MOE_HOLDOUT_EACH_SIDE]] + [z["request"] for z in reqs[-MOE_HOLDOUT_EACH_SIDE:]]
    part = dict(
        schema_version=1,
        sources={"validation_split.json": sha256_file(split_json), "routing_stats.json": sha256_file(routing_json)},
        rule="roles copied verbatim from the planning split; MoE hold-out = 5 least + 5 most imbalanced corpus requests with >= 100 tokens",
        passes=passes,
        moe_heldout_requests=moe_held,
        moe_calibration_requests=[z["request"] for z in reqs if z["request"] not in moe_held],
        moe_short_requests_excluded=[z["request"] for z in routing["per_request"] if z["tokens"] < MOE_HOLDOUT_MIN_TOKENS],
        salvaged_pass_usage="diverse_nsys_salvaged is cross-check only: never used for calibration, its calibration role is ignored",
    )
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "partition.json")
    with open(p, "w") as f:
        json.dump(part, f, indent=0, sort_keys=True)
    digest = sha256_file(p)
    with open(os.path.join(out_dir, "PARTITION.sha256"), "w") as f:
        f.write(f"{digest}  partition.json\n")
    return part


class Partition:
    def __init__(self, part: dict):
        self.part = part
        self.role = {lab: {int(k): r for k, r in v["role"].items()} for lab, v in part["passes"].items()}
        self.cls = {lab: {int(k): c for k, c in v["cls"].items()} for lab, v in part["passes"].items()}
        self.calibration_passes = {lab for lab in self.role if lab != "diverse_nsys_salvaged"}

    def ids(self, label: str, role: str) -> set[int]:
        return {it for it, r in self.role[label].items() if r == role}

    def role_of(self, label: str, it: int) -> str:
        return self.role[label].get(it, ROLE_EXCL)

    def class_of(self, label: str, it: int) -> str:
        return self.cls[label].get(it, "unknown")

    def calibration_only(self, label: str, ids) -> list[int]:
        """Return `ids` if every one of them is a calibration iteration of a calibration pass, else raise."""
        if label not in self.calibration_passes:
            raise LeakageError(f"pass {label} is not a calibration pass")
        bad = [it for it in ids if self.role[label].get(int(it)) != ROLE_CAL]
        if bad:
            raise LeakageError(f"{label}: {len(bad)} non-calibration iteration(s) in a fit, e.g. {bad[:5]}")
        return list(ids)


def load(part_dir: str) -> Partition:
    p = os.path.join(part_dir, "partition.json")
    frozen = open(os.path.join(part_dir, "PARTITION.sha256")).read().split()[0]
    digest = sha256_file(p)
    if digest != frozen:
        raise LeakageError(f"partition.json sha256 {digest} != frozen {frozen}; the partition must not change after freezing")
    part = json.load(open(p))
    part["_sha256"] = digest
    return Partition(part)


if __name__ == "__main__":
    import sys
    materialize(sys.argv[1], sys.argv[2], sys.argv[3])
    print(open(os.path.join(sys.argv[3], "PARTITION.sha256")).read())
