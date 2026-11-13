"""Predictor: loads coefficients.json and predicts one iteration from its structural features."""
from __future__ import annotations

import json

import numpy as np

from .collectives import CollectiveModel
from .data import Feat
from .decode import DecodeModel
from .longctx import PrefillComputeModel
from .mixed import MixedModel
from .moe import MoeModel


class Predictor:
    def __init__(self, coeffs: dict):
        self.coeffs = coeffs
        self.decode = DecodeModel.from_dict(coeffs["decode"])
        self.coll = CollectiveModel.from_dict(coeffs["collectives"])
        self.prefill = PrefillComputeModel.from_dict(coeffs["prefill"])
        self.mixed = MixedModel.from_dict(coeffs["mixed"])
        self.moe = MoeModel.from_dict(coeffs["moe"])
        self._decode_cache: dict[int, dict] = {}

    @classmethod
    def load(cls, path: str) -> "Predictor":
        return cls(json.load(open(path)))

    def predict(self, f: Feat) -> dict:
        if f.stage == "decode":
            if f.b not in self._decode_cache:
                self._decode_cache[f.b] = self.decode.predict(f)
            p = dict(self._decode_cache[f.b])
            p["kv_status"] = "in-range" if f.kv_max <= self.decode.kv_max_observed else "kv-extrapolation"
            return p
        return self.mixed.predict(f, self.prefill, self.coll, lead_ns=f.lead_ns)
