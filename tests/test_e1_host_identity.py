"""Host identity cross-check: missing, conflicting and malformed identities (checks/e1_host_identity.py)."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_host_identity.py"
HOST = "pod4-gb300-10-tray16-f3"


def _load():
    spec = importlib.util.spec_from_file_location("e1_host_identity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lc(host=HOST, **ident):
    return {"host": host, "host_identity": {"hostname": host, "boot_id": "54a59f08", **ident}}


RES = {"node": HOST, "phase": "Running"}


def test_agreeing_sources_pass():
    out = _load().cross_check(_lc(), HOST, RES, [HOST, HOST])
    assert out["status"] == "PASS" and out["sources"]["reservation.node"] == HOST


@pytest.mark.parametrize(
    ("lc", "log", "res", "gpu", "expect"),
    [
        (
            {"phase": "report_written"},
            HOST,
            RES,
            [HOST],
            "INCONCLUSIVE",
        ),  # the V1 record: no host at all
        (None, HOST, RES, [HOST], "INCONCLUSIVE"),
        (_lc(), None, RES, [HOST], "INCONCLUSIVE"),  # run log without a start line
        (_lc(), HOST, None, [HOST], "INCONCLUSIVE"),  # no reservation record
        (_lc(), HOST, RES, [], "INCONCLUSIVE"),  # no GPU observation
        (_lc(), "pod4-gb300-10-tray14-f3", RES, [HOST], "FAIL"),  # run log from another tray
        (
            _lc(),
            HOST,
            {"node": "pod4-gb300-10-tray15-f3", "phase": "Running"},
            [HOST],
            "FAIL",
        ),  # reservation elsewhere
        (_lc(), HOST, RES, [HOST, "pod4-gb300-10-tray02-f3"], "FAIL"),  # a census from another tray
        (
            _lc(),
            HOST,
            {"node": HOST, "phase": "Pending"},
            [HOST],
            "FAIL",
        ),  # reservation not Running
        (_lc(host=""), "", RES, [HOST], "FAIL"),  # malformed: empty
        (_lc(host="tray 16; rm -rf"), HOST, RES, [HOST], "FAIL"),  # malformed: not a host name
        (_lc(host=16), HOST, RES, [HOST], "FAIL"),  # malformed: not a string
        (_lc(host="a" * 300), HOST, RES, [HOST], "FAIL"),  # malformed: too long
        (
            {"host": HOST, "host_identity": {"hostname": "other", "boot_id": "b"}},
            HOST,
            RES,
            [HOST],
            "FAIL",
        ),  # record disagrees with itself
        (
            {"host": HOST, "host_identity": {"hostname": HOST}},
            HOST,
            RES,
            [HOST],
            "FAIL",
        ),  # no boot id
        (_lc(), HOST, RES, [HOST, None], "FAIL"),  # an observation without a host
    ],
)
def test_missing_conflicting_and_malformed_identities(lc, log, res, gpu, expect):
    assert _load().cross_check(lc, log, res, gpu)["status"] == expect
