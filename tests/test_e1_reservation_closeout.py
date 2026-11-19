"""V4 bounded closeout and deletion of the GPU reservation (checks/e1_reservation_closeout.py), CPU only, against the stand-in
for kubectl (e1_v4scale_mock_kubectl.py; never a cluster).

  record / verify   exact ownership of the reservation pod; the proof that a UID precondition is honoured
  watch (tray)      no permit before the run has ended (no early release); permit only with the registry evidence, two
                    consecutive clean GPU censuses and verified ownership; the verdicts, with exact boundaries
  delete (laptop)   nothing is deleted without two consecutive permits; one request, with the recorded UID as precondition;
                    a foreign pod of the same name is never deleted; late or unverifiable is a failure
The watcher and the deleter signal nothing; the watcher never sends a delete request.
"""

from __future__ import annotations

# ruff: noqa: E501

import fcntl
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
TOOL, MOCK = E1 / "checks" / "e1_reservation_closeout.py", E1 / "e1_v4scale_mock_kubectl.py"
D = json.loads((E1 / "E1-V4-SCALE-INPUTS.json").read_text())
CEILING, LATEST = float(D["reservation"]["hard_ceiling_s"]), float(D["reservation"]["launch_max_age_s"])
K = D["reservation"]["closeout"]
TAIL, MARGIN = float(K["watcher_tail_s"]), float(K["never_launched_margin_s"])
HOST = socket.gethostname()
CREATED = 1_800_000_000.0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C, M = _load(TOOL, "e1_reservation_closeout_t"), _load(MOCK, "e1_v4scale_mock_kubectl_t")


class Cluster:
    """The stand-in cluster: one reservation pod on this host, created at CREATED (or `created`)."""

    def __init__(self, tmp: Path, *, created: float = CREATED, **pod):
        tmp.mkdir(parents=True, exist_ok=True)
        self.path, self.created = tmp / "kube.json", created
        p = M.reservation_pod(pod.pop("node", HOST), created=created, **pod)
        self.uid, self.name, self.ns = p["metadata"]["uid"], p["metadata"]["name"], p["metadata"]["namespace"]
        self.path.write_text(json.dumps({"pods": [p], "unreachable": False, "delete_delay_s": 0, "requests": []}))

    def prefix(self, role: str) -> list[str]:
        return [sys.executable, "-I", str(MOCK), f"--mock-state={self.path}", f"--mock-role={role}"]

    def edit(self, fn) -> None:
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            st = json.load(fh)
            fn(st)
            fh.seek(0)
            fh.truncate()
            json.dump(st, fh)

    def state(self) -> dict:
        return json.loads(self.path.read_text())

    def deletes(self, role: str | None = None, dry: bool = False) -> list[dict]:
        return [r for r in self.state()["requests"] if r["args"][:1] == ["delete"] and bool(r.get("dry_run")) == dry and (role is None or r["role"] == role)]

    def remove_pod(self) -> None:
        self.edit(lambda st: st.update(pods=[]))

    def replace_pod(self) -> None:
        self.edit(lambda st: st["pods"][0]["metadata"].update(uid=str(uuid.uuid4())))

    def record(self, tmp: Path) -> Path:
        out = tmp / "reservation.json"
        rc, lines = C.record(self.prefix("deleter"), self.ns, self.name, HOST, out)
        assert rc == 0, lines
        return out


# ---- record and verify


def test_record_describes_the_pod_and_proves_the_uid_precondition(tmp_path):
    cl = Cluster(tmp_path)
    out = tmp_path / "reservation.json"
    rc, lines = C.record(cl.prefix("deleter"), cl.ns, cl.name, HOST, out)
    r = json.loads(out.read_text())
    assert rc == 0 and r["uid"] == cl.uid and r["pod"] == cl.name and r["namespace"] == cl.ns and r["node"] == HOST and r["phase"] == "Running" and r["gpu_request"] == "4" and r["creation_timestamp"] == M.iso(CREATED)
    assert r["deletion_path_proof"] == {"dry_run_with_recorded_uid": "accepted", "dry_run_with_other_uid": "conflict", "pod_unchanged_afterwards": True, "at": r["deletion_path_proof"]["at"], "ok": True}
    assert len(cl.deletes(dry=True)) == 2 and cl.deletes(dry=False) == [] and len(cl.state()["pods"]) == 1 and "deletionTimestamp" not in cl.state()["pods"][0]["metadata"]  # nothing was deleted
    assert [x["precondition_uid"] for x in cl.deletes(dry=True)] == [cl.uid, C.ZERO_UID]


def test_record_refuses_when_the_server_does_not_honour_the_precondition(tmp_path):
    cl = Cluster(tmp_path)
    cl.edit(lambda st: st.update(ignore_preconditions=True))
    rc, lines = C.record(cl.prefix("deleter"), cl.ns, cl.name, HOST, tmp_path / "reservation.json")
    assert rc == 1 and "must be a Conflict" in lines[-1] and not (tmp_path / "reservation.json").exists()


def test_record_refuses_without_the_deleting_credential(tmp_path):
    cl = Cluster(tmp_path)
    rc, lines = C.record(cl.prefix("reader"), cl.ns, cl.name, HOST, tmp_path / "reservation.json")
    assert rc == 1 and "Forbidden" in lines[-1] and not (tmp_path / "reservation.json").exists()


@pytest.mark.parametrize(("pod", "word"), [({"node": "another-node"}, "node"), ({"image": "nvcr.io/some/workload:1"}, "pause"), ({"gpus": "2"}, "GPU request"), ({"labels": {"app": "dsv41-gb300-reserve", "owner": "somebody-else"}}, "label owner"),
                                           ({"labels": {"app": "amoeba", "owner": "kevin"}}, "label app")])
def test_record_refuses_a_pod_that_is_not_the_reservation_placeholder(tmp_path, pod, word):
    cl = Cluster(tmp_path, **pod)
    rc, lines = C.record(cl.prefix("deleter"), cl.ns, cl.name, HOST, tmp_path / "reservation.json")
    assert rc == 1 and word in lines[-1] and not (tmp_path / "reservation.json").exists() and cl.deletes() == [] and cl.deletes(dry=True) == []  # not even a dry run against a pod that is not ours


def test_record_refuses_a_missing_or_terminating_or_controller_owned_pod(tmp_path):
    cl = Cluster(tmp_path / "a")
    assert C.record(cl.prefix("deleter"), cl.ns, "no-such-pod", HOST, tmp_path / "r.json")[0] == 1
    cl.edit(lambda st: st["pods"][0]["metadata"].update(deletionTimestamp="2026-11-19T00:00:00Z"))
    assert C.record(cl.prefix("deleter"), cl.ns, cl.name, HOST, tmp_path / "r.json")[0] == 1
    cl2 = Cluster(tmp_path / "b")
    cl2.edit(lambda st: st["pods"][0]["metadata"].update(ownerReferences=[{"kind": "StatefulSet", "name": "amoeba-tp2"}]))
    rc, lines = C.record(cl2.prefix("deleter"), cl2.ns, cl2.name, HOST, tmp_path / "r.json")
    assert rc == 1 and "owner" in lines[-1] and not (tmp_path / "r.json").exists()


def test_verify_with_the_read_only_credential(tmp_path):
    cl = Cluster(tmp_path)
    res = cl.record(tmp_path)
    rc, text = C.verify(cl.prefix("reader"), res, HOST)
    assert rc == 0 and text.startswith("PASS  reservation ownership") and cl.uid in text
    assert C.verify(cl.prefix("reader"), res, "another-host")[0] == 1  # the record is for another node than the one asking
    cl.replace_pod()
    rc, text = C.verify(cl.prefix("reader"), res, HOST)
    assert rc == 1 and "uid" in text
    cl.remove_pod()
    assert C.verify(cl.prefix("reader"), res, HOST)[0] == 1
    cl.edit(lambda st: st.update(unreachable=True))
    assert C.verify(cl.prefix("reader"), res, HOST)[0] == 3


def test_verify_needs_the_deletion_path_proof_and_a_complete_record(tmp_path):
    cl = Cluster(tmp_path)
    res = cl.record(tmp_path)
    good = json.loads(res.read_text())
    for broken in ({k: v for k, v in good.items() if k != "deletion_path_proof"}, {**good, "deletion_path_proof": {"ok": False}}, {k: v for k, v in good.items() if k != "uid"}, {**good, "creation_timestamp": "yesterday"}):
        res.write_text(json.dumps(broken))
        assert C.verify(cl.prefix("reader"), res, HOST)[0] == 1
    res.write_text("{")
    assert C.verify(cl.prefix("reader"), res, HOST)[0] == 1


# ---- watch


class Tray:
    """The capture tray's side with a fake clock: the run's registry state and the GPU census are what the test says they are."""

    def __init__(self, tmp: Path, cluster: Cluster, *, age: float):
        self.cl, self.run = cluster, tmp / "root" / "e1_v4_scale"
        self.run.mkdir(parents=True)
        self.registry, self.out = tmp / "root" / "pids" / "owned.tsv", self.run / "reservation_closeout.json"
        cluster.record(self.run)
        self.t = cluster.created + age
        self.alive, self.driver, self.refused, self.lookalikes, self.released = [{"pid": 4242, "label": "driver:e1_v4_scale"}], True, [], [], True
        self.events, self.seen = [], []

    def at(self, age: float, action) -> None:
        self.events.append((age, action))
        self.events.sort(key=lambda e: e[0])

    def sleep(self, s: float) -> None:
        self.seen.append(json.loads(self.out.read_text()))
        self.t += s
        while self.events and self.events[0][0] <= self.t - self.cl.created:
            self.events.pop(0)[1]()

    def census(self) -> dict:
        return {"at": self.t, "verdict": {"available": True, "released": self.released, "reason": "no compute process on any GPU" if self.released else "1 compute process(es); 4 GPU(s) above 64 MiB"},
                "compute_apps": [] if self.released else [{"pid": 999}], "gpus": [{"index": i, "memory_used_mib": 0 if self.released else 260_000} for i in range(4)]}

    def run_state(self) -> dict:
        return {"driver_registered": self.driver, "records": 5, "owned_alive": list(self.alive), "refused": list(self.refused), "stale": 0, "unattributed_lookalikes": list(self.lookalikes), "attributed_lookalikes": 0}

    def go(self, interval: float = 5.0, **kw):
        rc = C.watch(self.run, self.registry, self.out, self.cl.prefix("reader"), ceiling_s=CEILING, launch_latest_age_s=LATEST, never_launched_margin_s=MARGIN, tail_s=TAIL, interval=interval, host=HOST,
                     now=lambda: self.t, sleep=self.sleep, census=self.census, state_of_run=self.run_state, nsight=lambda: [{"pid": 5000, "exe": "/opt/nvidia/nsight-systems/x/QdstrmImporter"}], **kw)
        rec = json.loads(self.out.read_text())
        assert rec["signals_sent"] == 0 and rec["delete_requests_sent"] == 0 and self.cl.deletes("reader") == [] and self.cl.deletes("reader", dry=True) == []  # the watcher only reads
        return rc, rec

    def decisions(self) -> list[str]:
        return [r["decision"] for r in self.seen]


def end_run(tr: Tray):
    return lambda: setattr(tr, "alive", [])


def test_no_permit_while_the_run_is_active_whatever_the_gpus_say(tmp_path):
    """No early release: the GPUs are idle and the pod is ours, but a registered process of the run is alive until after the ceiling."""
    tr = Tray(tmp_path, Cluster(tmp_path), age=4000)
    rc, rec = tr.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "NOT_DELETED" and rec["permits"] == 0 and rec["first_permit"] is None and set(tr.decisions()) == {"WAIT"}
    assert "no early release" in tr.seen[-1]["reasons"][0] and rec["ended_at"] - CREATED > CEILING + TAIL and len(tr.cl.state()["pods"]) == 1
    gets = [r for r in tr.cl.state()["requests"] if r["args"][:2] == ["get", "pod"]]
    assert len(gets) <= (CEILING + TAIL - 4000) / 60 + 4  # while waiting the pod is read once a minute, not every evaluation


def test_permit_after_the_run_ended_needs_two_consecutive_clean_censuses_then_the_deletion_is_observed(tmp_path):
    tr = Tray(tmp_path, Cluster(tmp_path), age=4000)
    tr.at(4012, end_run(tr))
    tr.at(4040, tr.cl.remove_pod)  # the deleter's work, 20 s after the first permit
    rc, rec = tr.go()
    assert rc == 0 and rec["verdict"] == "DELETED_IN_TIME" and rec["complete"] is True
    assert tr.decisions()[:5] == ["WAIT", "WAIT", "WAIT", "REFUSE", "PERMIT"] and "a second consecutive census is required" in tr.seen[3]["reasons"][0]
    p = rec["first_permit"]
    assert p["reservation_age_s"] == 4020.0 and p["evidence"]["run"]["owned_alive"] == [] and p["evidence"]["run_ended"] == "registered tree gone" and p["evidence"]["gpu_census"]["released"] is True and p["evidence"]["gpu_census"]["previous_evaluation_released"] is True
    assert p["evidence"]["ownership"] == {"verified": True, "uid": tr.cl.uid, "terminating": False} and rec["deleted_observed_at"] - CREATED == 4040.0 and rec["after_deletion"]["gpu_pods_on_node"] == 0
    assert p["evidence"]["nsight_processes_alive"] == [{"pid": 5000, "exe": "/opt/nvidia/nsight-systems/x/QdstrmImporter"}]  # a live importer is recorded and does not block the permit
    assert all(r["sequence"] == i + 1 for i, r in enumerate(tr.seen)) and rec["reservation"]["uid"] == tr.cl.uid


def test_busy_gpus_refuse_and_a_single_clean_census_is_not_enough(tmp_path):
    tr = Tray(tmp_path, Cluster(tmp_path), age=4000)
    tr.alive, tr.released = [], False
    tr.at(4018, lambda: setattr(tr, "released", True))
    tr.at(4022, lambda: setattr(tr, "released", False))  # a compute process appears again: the single clean census must not have permitted anything
    tr.at(4032, lambda: setattr(tr, "released", True))
    tr.at(4050, tr.cl.remove_pod)
    rc, rec = tr.go()
    assert tr.decisions()[:9] == ["REFUSE"] * 8 + ["PERMIT"] and "D3 GPU census: 1 compute process" in tr.seen[0]["reasons"][0] and tr.seen[0]["evidence"]["gpu_census"]["compute_pids"] == [999]
    assert rec["first_permit"]["reservation_age_s"] == 4040.0 and rc == 0


@pytest.mark.parametrize(("attr", "value", "word"), [("refused", [{"pid": 77, "reason": "another uid"}], "D1 registry refusal"), ("lookalikes", [{"pid": 78, "cmd": "python -m launcher alignment profile /x/dsv41_flash_gb300_tp4_e1_v4scale/profile.json"}], "D2 process(es) that look like the experiment")])
def test_registry_refusal_or_unregistered_lookalike_refuses_until_the_end(tmp_path, attr, value, word):
    tr = Tray(tmp_path, Cluster(tmp_path), age=5300)
    tr.alive = []
    setattr(tr, attr, value)
    rc, rec = tr.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "NOT_DELETED" and rec["permits"] == 0 and set(tr.decisions()) == {"REFUSE"} and word in tr.seen[-1]["reasons"][0] and len(tr.cl.state()["pods"]) == 1


def test_a_live_importer_is_recorded_and_is_not_a_refusal(tmp_path):
    """The importer's command line names the run directory, so the registry tool lists it as a lookalike. It is identified by its
    executable path, holds no GPU and is never signalled: it must not keep the reservation from being deleted."""
    tr = Tray(tmp_path, Cluster(tmp_path), age=4000)
    tr.alive = []
    tr.lookalikes = [{"pid": 5000, "cmd": "QdstrmImporter --input-file /tmp/nsys-report-1.qdstrm --output-file /x/dsv41_flash_gb300_tp4_e1_v4scale/e1_v4_scale/profile/nsys/x.nsys-rep"}]
    tr.at(4030, tr.cl.remove_pod)
    rc, rec = tr.go()
    assert rc == 0 and tr.decisions()[:2] == ["REFUSE", "PERMIT"] and rec["first_permit"]["evidence"]["run"]["unattributed_lookalikes"] == [] and rec["first_permit"]["evidence"]["nsight_processes_alive"][0]["pid"] == 5000
    other = Tray(tmp_path / "other", Cluster(tmp_path / "other"), age=5300)  # the same command line from a process that is NOT an Nsight executable is a refusal
    other.alive, other.lookalikes = [], [{"pid": 5001, "cmd": tr.lookalikes[0]["cmd"]}]
    rc, rec = other.go(interval=30.0)
    assert rc == 1 and rec["permits"] == 0


def test_a_transient_lookalike_withdraws_the_permit_for_that_evaluation_only(tmp_path):
    tr = Tray(tmp_path, Cluster(tmp_path), age=4000)
    tr.alive = []
    tr.at(4012, lambda: setattr(tr, "lookalikes", [{"pid": 78, "cmd": "bash monitor_e1_v4scale.sh"}]))
    tr.at(4018, lambda: setattr(tr, "lookalikes", []))
    tr.at(4030, tr.cl.remove_pod)
    rc, rec = tr.go()
    assert tr.decisions()[:6] == ["REFUSE", "PERMIT", "PERMIT", "REFUSE", "PERMIT", "PERMIT"] and rec["first_permit"]["sequence"] == 2 and rec["permits"] == 4 and rc == 0


@pytest.mark.parametrize(("change", "word"), [
    (lambda st: st["pods"][0]["metadata"]["labels"].update(owner="somebody-else"), "label owner"),
    (lambda st: st["pods"][0]["spec"].update(nodeName="another-node"), "node"),
    (lambda st: st["pods"][0]["spec"]["containers"][0].update(image="nvcr.io/some/workload:1"), "pause"),
    (lambda st: st["pods"][0]["metadata"].update(creationTimestamp="2026-01-01T00:00:00Z"), "creation_timestamp"),
    (lambda st: st["pods"][0]["metadata"].update(namespace="inference-wideep"), None),  # another namespace: for the recorded namespace the pod is gone
    (lambda st: st["pods"][0]["metadata"].update(deletionTimestamp="2026-11-19T00:00:00Z"), "terminating although no permit"),
])
def test_ownership_must_be_exact(tmp_path, change, word):
    tr = Tray(tmp_path, Cluster(tmp_path), age=5300)
    tr.alive = []
    tr.cl.edit(change)
    rc, rec = tr.go(interval=30.0)
    assert rc == 1 and rec["permits"] == 0 and rec["first_permit"] is None
    if word is None:
        assert rec["verdict"] == "GONE_WITHOUT_PERMIT"
    else:
        assert rec["verdict"] == "NOT_DELETED" and word in str(tr.seen[-1]["reasons"]) and "D4" in str(tr.seen[-1]["reasons"])


def test_pod_gone_or_replaced_before_any_permit(tmp_path):
    for i, change in enumerate(("remove_pod", "replace_pod")):
        tr = Tray(tmp_path / str(i), Cluster(tmp_path / str(i)), age=1000)
        tr.at(1100, getattr(tr.cl, change))
        rc, rec = tr.go()
        assert rc == 1 and rec["verdict"] == "GONE_WITHOUT_PERMIT" and rec["first_permit"] is None and "outside the registered mechanism" in rec["verdict_reason"]
        assert 1100 <= rec["deleted_observed_at"] - CREATED <= 1100 + 60 + 5  # noticed at the next read of the pod (once a minute while the run is active)
        if change == "replace_pod":
            assert "replaced by uid" in rec["evidence"]["pod_state"] and len(tr.cl.state()["pods"]) == 1  # the foreign pod is untouched


@pytest.mark.parametrize(("start", "expected"), [(LATEST + MARGIN - 10, ["WAIT", "WAIT", "WAIT", "REFUSE", "PERMIT"])])
def test_never_launched_permits_only_after_the_launch_window_and_its_margin(tmp_path, start, expected):
    tr = Tray(tmp_path, Cluster(tmp_path), age=start)
    tr.alive, tr.driver = [], False
    tr.at(start + 40, tr.cl.remove_pod)
    rc, rec = tr.go()
    assert tr.decisions()[:5] == expected and [r["reservation_age_s"] for r in tr.seen[:5]] == [start, start + 5, LATEST + MARGIN, LATEST + MARGIN + 5, LATEST + MARGIN + 10]  # exactly at the margin: still waiting
    assert "a launch is possible until reservation age 470 s" in tr.seen[0]["reasons"][0] and rec["first_permit"]["evidence"]["run_ended"].startswith("never launched") and rc == 0


@pytest.mark.parametrize(("gone_at", "verdict", "rc"), [(CEILING - 5, "DELETED_IN_TIME", 0), (CEILING, "DELETED_IN_TIME", 0), (CEILING + 5, "DELETED_LATE", 1), (CEILING + TAIL, "DELETED_LATE", 1)])
def test_deletion_boundary_at_the_ceiling(tmp_path, gone_at, verdict, rc):
    tr = Tray(tmp_path, Cluster(tmp_path), age=CEILING - 30)
    tr.alive = []
    tr.at(gone_at, tr.cl.remove_pod)
    got, rec = tr.go()
    assert (got, rec["verdict"]) == (rc, verdict) and rec["deleted_observed_at"] - CREATED == gone_at and rec["deadline_epoch"] - CREATED == CEILING and rec["first_permit"]["reservation_age_s"] == CEILING - 25


def test_not_deleted_after_the_tail_and_unverified_when_the_cluster_cannot_be_read(tmp_path):
    tr = Tray(tmp_path / "a", Cluster(tmp_path / "a"), age=CEILING - 30)
    tr.alive = []
    rc, rec = tr.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "NOT_DELETED" and rec["permits"] > 10 and "still present" in rec["verdict_reason"] and CEILING + TAIL < rec["ended_at"] - CREATED <= CEILING + TAIL + 30
    tr = Tray(tmp_path / "b", Cluster(tmp_path / "b"), age=CEILING - 30)
    tr.alive = []
    tr.at(CEILING - 10, lambda: tr.cl.edit(lambda st: st.update(unreachable=True)))
    rc, rec = tr.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "UNVERIFIED" and "D4 the pod's state could not be read" in str(tr.seen[-1]["reasons"]) and tr.seen[-1]["decision"] == "REFUSE"


def test_unusable_reservation_record_never_permits(tmp_path):
    cl = Cluster(tmp_path)
    tr = Tray(tmp_path, cl, age=4000)
    tr.alive = []
    (tr.run / "reservation.json").write_text(json.dumps({**json.loads((tr.run / "reservation.json").read_text()), "node": "another-node"}))
    rc, rec = tr.go(interval=30.0, max_s=300.0)
    assert rc == 1 and rec["verdict"] == "UNVERIFIED" and rec["permits"] == 0 and "is not this host" in str(tr.seen[-1]["reasons"])


def test_run_state_attributes_only_processes_of_the_given_registries(tmp_path):
    """D2 on real processes: a process that looks like the experiment is a refusal unless it is registered in one of the given registries."""
    pids = C._mod("e1_pids")
    marker = f"closeout-lookalike-test-{uuid.uuid4().hex}"
    pids.LOOKALIKE_RE = re.compile(marker)
    owned, other = tmp_path / "owned.tsv", tmp_path / "supervisor.tsv"
    stand_in = subprocess.Popen(["bash", "-c", f"read x; : {marker}"], start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # waits on its stdin; no child
    try:
        time.sleep(0.3)
        st = C.run_state(pids, owned, [])
        assert [x["pid"] for x in st["unattributed_lookalikes"]] == [stand_in.pid] and st["owned_alive"] == [] and st["driver_registered"] is False
        pids.register(str(other), "supervisor:e1_v4_scale", stand_in.pid)
        st = C.run_state(pids, owned, [other])
        assert st["unattributed_lookalikes"] == [] and st["attributed_lookalikes"] == 1 and st["owned_alive"] == []  # registered elsewhere: attributed, and still not part of the run's tree
        pids.register(str(owned), "driver:e1_v4_scale", stand_in.pid)
        st = C.run_state(pids, owned, [other])
        assert st["driver_registered"] is True and [(o["pid"], o["label"]) for o in st["owned_alive"]] == [(stand_in.pid, "driver:e1_v4_scale")]
    finally:
        stand_in.terminate()  # the test's own child, by its handle
        stand_in.wait(timeout=10)
    assert C.run_state(pids, owned, [other])["owned_alive"] == []


# ---- armed


def test_armed_requires_a_live_registered_watcher_a_fresh_evaluation_and_a_deleter_heartbeat(tmp_path):
    pids = C._mod("e1_pids")
    reg, rec = tmp_path / "closeout.tsv", tmp_path / "reservation_closeout.json"
    watcher = subprocess.Popen(["sleep", "60"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        pids.register(str(reg), "closeout:e1_v4_scale", watcher.pid)
        good = {"pid": watcher.pid, "updated_at": 1000.0, "sequence": 3, "decision": "WAIT", "reasons": ["no driver registered"], "first_permit": None, "complete": False, "verdict": None, "deleter_heartbeat_age_s": 4.0, "deadline": "2026-11-19T01:30:00Z"}
        assert C.armed(rec, reg, now=1005.0)[0] == 1 and "no record" in C.armed(rec, reg, now=1005.0)[1]
        rec.write_text(json.dumps(good))
        rc, text = C.armed(rec, reg, now=1005.0)
        assert rc == 0 and text.startswith("PASS  closeout watcher armed")
        for bad, word in (({"updated_at": 960.0}, "last evaluation 45 s ago"), ({"deleter_heartbeat_age_s": None}, "deleter heartbeat missing"), ({"deleter_heartbeat_age_s": 61.0}, "61 s old"), ({"decision": "PERMIT"}, "decision PERMIT"),
                          ({"first_permit": {"at": 1.0}}, "already"), ({"complete": True, "verdict": "NOT_DELETED"}, "has ended"), ({"pid": watcher.pid + 1}, "is not the record's writer")):
            rec.write_text(json.dumps({**good, **bad}))
            rc, text = C.armed(rec, reg, now=1005.0)
            assert rc == 1 and word in text, word
    finally:
        watcher.terminate()
        watcher.wait(timeout=10)
    rec.write_text(json.dumps(good))
    assert C.armed(rec, reg, now=1005.0)[0] == 1  # the registered watcher is gone


# ---- delete


class Laptop:
    """The deleting side with a fake clock. The permit command is answered by the test; kubectl is the stand-in with the deleting credential."""

    def __init__(self, tmp: Path, cluster: Cluster, *, age: float):
        self.cl, self.res, self.out = cluster, cluster.record(tmp), tmp / "reservation_deleted.json"
        self.t, self.seq, self.plan, self.events, self.published, self.fetches = cluster.created + age, 0, [], [], [], 0
        self.default = "WAIT"

    def permit(self, decision: str = "PERMIT", **over) -> dict:
        self.seq += 1
        return {"decision": decision, "complete": False, "verdict": None, "host": HOST, "pid": 4711, "started_at": CREATED + 50, "sequence": self.seq, "updated_at": self.t, "reservation": {"uid": self.cl.uid}, "first_permit": {"at": self.t - 5} if decision == "PERMIT" else None, **over}

    def runner(self, cmd, **kw):
        if cmd == ["PERMIT"]:
            self.fetches += 1
            item = self.plan.pop(0) if self.plan else self.default
            if item is None:
                return SimpleNamespace(returncode=255, stdout="", stderr="ssh: connect to host: timed out")
            return SimpleNamespace(returncode=0, stdout=json.dumps(item if isinstance(item, dict) else self.permit(item)), stderr="")
        if cmd == ["PUBLISH"]:
            self.published.append(json.loads(kw["input"]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return subprocess.run(cmd, **kw)

    def at(self, age: float, action) -> None:
        self.events.append((age, action))
        self.events.sort(key=lambda e: e[0])

    def sleep(self, s: float) -> None:
        self.t += s
        while self.events and self.events[0][0] <= self.t - self.cl.created:
            self.events.pop(0)[1]()

    def go(self, interval: float = 5.0):
        rc = C.delete(self.res, ["PERMIT"], self.out, self.cl.prefix("deleter"), ceiling_s=CEILING, tail_s=TAIL, interval=interval, publish_cmd=["PUBLISH"], now=lambda: self.t, sleep=self.sleep, runner=self.runner)
        rec = json.loads(self.out.read_text())
        assert rec["signals_sent"] == 0 and self.published and self.published[-1]["verdict"] == rec["verdict"] and rec["delete_requests_accepted"] <= 1  # the final record reached the tray; never more than one accepted request
        return rc, rec


def test_nothing_is_deleted_without_a_permit(tmp_path):
    for i, default in enumerate(("WAIT", "REFUSE", None)):
        cl = Cluster(tmp_path / str(i))
        lp = Laptop(tmp_path / str(i), cl, age=CEILING - 300)
        lp.default = default
        rc, rec = lp.go(interval=30.0)
        assert rc == 1 and rec["verdict"] == "NOT_DELETED" and rec["delete_requests_sent"] == 0 and cl.deletes() == [] and len(cl.state()["pods"]) == 1 and "nothing was deleted without a permit" in rec["verdict_reason"]
        assert rec["permit_fetch_failures"] == (rec["permit_fetches"] if default is None else 0) and "deleted_observed_at" not in rec


def test_one_permit_is_not_enough_and_the_same_evaluation_twice_is_not_two(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=CEILING - 300)
    one = lp.permit()
    lp.plan = ["WAIT", one, one, one, "REFUSE", "PERMIT", "WAIT", "PERMIT", None, "PERMIT", "REFUSE"]  # never two consecutive, different PERMIT evaluations
    rc, rec = lp.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "NOT_DELETED" and cl.deletes() == [] and len(cl.state()["pods"]) == 1


def test_two_consecutive_permits_then_one_delete_with_the_recorded_uid(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=4000)
    lp.plan = ["WAIT", "PERMIT", "PERMIT"]
    lp.default = "PERMIT"
    rc, rec = lp.go()
    dels = cl.deletes()
    assert rc == 0 and rec["verdict"] == "DELETED_IN_TIME" and [(d["role"], d["rc"], d["precondition_uid"]) for d in dels] == [("deleter", 0, cl.uid)] and cl.state()["pods"] == []
    assert rec["delete_requests_sent"] == 1 and rec["delete_requests_accepted"] == 1 and rec["uid"] == rec["precondition_uid"] == cl.uid and rec["permit"]["sequence"] == 3 and rec["gpu_pods_on_node"] == 0 and rec["reservation_pod_still_listed"] is False
    assert rec["delete_requested_epoch"] - CREATED == 4010.0 and rec["deleted_observed_epoch"] - CREATED == 4010.0 and C.epoch(rec["deleted_observed_at"]) == rec["deleted_observed_epoch"] and rec["permit_fetches"] == 3
    assert [p["verdict"] for p in lp.published] == [None, "DELETED_IN_TIME"]  # published when the request was accepted and at the end


@pytest.mark.parametrize("over", [{"reservation": {"uid": "11111111-0000-4000-8000-000000000000"}}, {"host": "another-host"}, {"reservation": {}}])
def test_a_permit_for_another_uid_or_host_deletes_nothing(tmp_path, over):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=CEILING - 200)
    lp.plan = [lp.permit(**over) for _ in range(40)]
    rc, rec = lp.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "NOT_DELETED" and cl.deletes() == [] and len(cl.state()["pods"]) == 1


def test_two_permits_from_different_watchers_are_not_consecutive(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=CEILING - 200)
    lp.plan = [lp.permit(pid=4711 + (i % 2)) for i in range(40)]
    rc, rec = lp.go(interval=30.0)
    assert rc == 1 and cl.deletes() == []


def test_a_foreign_pod_of_the_same_name_is_never_deleted(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=4000)
    lp.default = "PERMIT"
    cl.replace_pod()
    foreign = cl.state()["pods"][0]["metadata"]["uid"]
    rc, rec = lp.go()
    assert rc == 1 and rec["verdict"] == "REFUSED_FOREIGN" and rec["delete_requests_sent"] == 0 and cl.deletes() == [] and "uid" in rec["verdict_reason"]
    assert cl.state()["pods"][0]["metadata"]["uid"] == foreign and "deletionTimestamp" not in cl.state()["pods"][0]["metadata"]


def test_the_uid_precondition_stops_a_replacement_between_check_and_delete(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=4000)
    lp.default = "PERMIT"
    cl.edit(lambda st: st.update(swap_uid_on_delete=True))
    rc, rec = lp.go()
    dels = cl.deletes()
    assert rc == 1 and rec["verdict"] == "REFUSED_FOREIGN" and "UID precondition failed" in rec["verdict_reason"] and rec["delete_requests_sent"] == 1 and rec["delete_requests_accepted"] == 0
    assert len(dels) == 1 and dels[0]["rc"] == 1 and "Conflict" in dels[0]["message"] and len(cl.state()["pods"]) == 1 and "deletionTimestamp" not in cl.state()["pods"][0]["metadata"]  # the replacement was not deleted


@pytest.mark.parametrize("change", [lambda st: st["pods"][0]["metadata"]["labels"].update(owner="somebody-else"), lambda st: st["pods"][0]["spec"]["containers"][0].update(image="nvcr.io/some/workload:1"), lambda st: st["pods"][0]["spec"].update(nodeName="another-node")])
def test_the_deleter_checks_ownership_itself(tmp_path, change):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=4000)
    lp.default = "PERMIT"
    cl.edit(change)
    rc, rec = lp.go()
    assert rc == 1 and rec["verdict"] == "REFUSED_FOREIGN" and cl.deletes() == [] and len(cl.state()["pods"]) == 1


def test_pod_already_gone_is_not_claimed_as_this_deletion(tmp_path):
    cl = Cluster(tmp_path / "a")
    lp = Laptop(tmp_path / "a", cl, age=4000)
    lp.default = "PERMIT"
    cl.remove_pod()
    rc, rec = lp.go()
    assert rc == 1 and rec["verdict"] == "GONE_NOT_BY_DELETER" and cl.deletes() == []
    cl = Cluster(tmp_path / "b")
    lp = Laptop(tmp_path / "b", cl, age=4000)
    cl.remove_pod()
    lp.default = lp.permit("REFUSE", complete=True, verdict="GONE_WITHOUT_PERMIT")
    rc, rec = lp.go()
    assert rc == 1 and rec["verdict"] == "GONE_NOT_BY_DELETER" and "GONE_WITHOUT_PERMIT" in rec["verdict_reason"] and cl.deletes() == []


def test_unreachable_cluster_and_a_failed_request_are_retried_and_still_one_accepted_request(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=4000)
    lp.default = "PERMIT"
    cl.edit(lambda st: st.update(unreachable=True))
    lp.at(4030, lambda: cl.edit(lambda st: st.update(unreachable=False, fail_next_delete=True)))
    rc, rec = lp.go()
    assert rc == 0 and rec["verdict"] == "DELETED_IN_TIME" and rec["delete_requests_sent"] == 2 and rec["delete_requests_accepted"] == 1
    assert [(d["rc"], d["precondition_uid"]) for d in cl.deletes()] == [(1, cl.uid), (0, cl.uid)] and [a.get("result") for a in rec["attempts"]][-2:] == ["failed", "accepted"]


@pytest.mark.parametrize(("gone_at", "verdict", "rc"), [(CEILING, "DELETED_IN_TIME", 0), (CEILING + 5, "DELETED_LATE", 1)])
def test_deleter_boundary_at_the_ceiling(tmp_path, gone_at, verdict, rc):
    cl = Cluster(tmp_path)
    cl.edit(lambda st: st.update(delete_delay_s=1e9))  # the pod terminates slowly; the test removes it at the given moment
    lp = Laptop(tmp_path, cl, age=CEILING - 30)
    lp.default = "PERMIT"
    lp.at(gone_at, cl.remove_pod)
    got, rec = lp.go()
    assert (got, rec["verdict"]) == (rc, verdict) and rec["deleted_observed_epoch"] - CREATED == gone_at and len(cl.deletes()) == 1 and rec["reservation_s"] == gone_at


def test_pod_that_never_goes_is_not_deleted_and_no_second_request_is_sent(tmp_path):
    cl = Cluster(tmp_path)
    cl.edit(lambda st: st.update(delete_delay_s=1e9))
    lp = Laptop(tmp_path, cl, age=CEILING - 30)
    lp.default = "PERMIT"
    rc, rec = lp.go(interval=30.0)
    assert rc == 1 and rec["verdict"] == "NOT_DELETED" and len(cl.deletes()) == 1 and rec["delete_requests_accepted"] == 1 and "deleted_observed_at" not in rec


def test_unusable_reservation_record_refuses_at_once(tmp_path):
    cl = Cluster(tmp_path)
    lp = Laptop(tmp_path, cl, age=4000)
    lp.default = "PERMIT"
    lp.res.write_text(json.dumps({k: v for k, v in json.loads(lp.res.read_text()).items() if k != "uid"}))
    rc, rec = lp.go()
    assert rc == 1 and rec["verdict"] == "REFUSED" and lp.fetches == 0 and cl.deletes() == []


# ---- both sides as real processes


def test_watcher_and_deleter_as_processes_end_to_end(tmp_path):
    """Real time, real processes: a registered driver stand-in ends, the watcher permits, the deleter deletes once, both report in time."""
    cl = Cluster(tmp_path, created=time.time() - 1000)
    root = tmp_path / "root"
    run = root / "e1_v4_scale"
    run.mkdir(parents=True)
    res = cl.record(run)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text('#!/bin/sh\ncase "$1" in --query-compute-apps=*) exit 0;; --query-gpu=*) for i in 0 1 2 3; do echo "$i, GPU-$i, 1, 290000"; done;; esac\n')  # idle GPUs, whatever this host's GPUs are doing
    (bindir / "nvidia-smi").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "E1_LOOKALIKE_RE": f"closeout-e2e-test-{uuid.uuid4().hex}"}  # nothing else on this host is a lookalike for this test
    helper = tmp_path / "permit.py"
    helper.write_text(f"import pathlib, sys\nd = pathlib.Path({str(run)!r})\n(d / 'deleter.heartbeat').touch()\nsys.stdout.write((d / 'reservation_closeout.json').read_text())\n")
    publish = tmp_path / "publish.py"
    publish.write_text(f"import pathlib, sys\npathlib.Path({str(run / 'reservation_deleted.json')!r}).write_text(sys.stdin.read())\n")
    spawn = [sys.executable, "-I", str(E1 / "checks" / "e1_pids.py"), "spawn"]
    subprocess.run([*spawn, "--registry", str(root / "pids" / "owned.tsv"), "--label", "driver:e1_v4_scale", "--", "sleep", "8"], check=True, capture_output=True)
    kube_r, kube_d = " ".join(cl.prefix("reader")), " ".join(cl.prefix("deleter"))
    subprocess.run([*spawn, "--registry", str(root / "pids" / "closeout.tsv"), "--label", "closeout:e1_v4_scale", "--log", str(tmp_path / "watch.log"), "--", sys.executable, "-I", str(TOOL), "watch", "--run", str(run), "--registry", str(root / "pids" / "owned.tsv"),
                    "--out", str(run / "reservation_closeout.json"), "--launch-latest-age-s", "470", "--ceiling-s", "5400", "--tail-s", "30", "--interval", "1", "--deleter-heartbeat", str(run / "deleter.heartbeat"), "--kubectl", kube_r], check=True, capture_output=True, env=env)
    subprocess.run([*spawn, "--registry", str(tmp_path / "deleter.tsv"), "--label", "deleter:e1_v4_scale", "--log", str(tmp_path / "delete.log"), "--", sys.executable, "-I", str(TOOL), "delete", "--reservation", str(res), "--permit-cmd", f"{sys.executable} -I {helper}",
                    "--publish-cmd", f"{sys.executable} -I {publish}", "--out", str(tmp_path / "deleted.json"), "--ceiling-s", "5400", "--tail-s", "30", "--interval", "1", "--kubectl", kube_d], check=True, capture_output=True, env=env)
    armed_seen = None
    deadline = time.time() + 90
    while time.time() < deadline:
        w = C.load(run / "reservation_closeout.json") or {}
        if armed_seen is None and w.get("decision") == "WAIT" and w.get("deleter_heartbeat_age_s") is not None:
            armed_seen = subprocess.run([sys.executable, "-I", str(TOOL), "armed", "--record", str(run / "reservation_closeout.json"), "--registry", str(root / "pids" / "closeout.tsv")], capture_output=True, text=True)
        if w.get("complete") and (C.load(tmp_path / "deleted.json") or {}).get("verdict"):
            break
        time.sleep(0.5)
    w, d = C.load(run / "reservation_closeout.json"), C.load(run / "reservation_deleted.json")
    try:
        assert armed_seen is not None and armed_seen.returncode == 0 and "PASS  closeout watcher armed" in armed_seen.stdout
        assert w["verdict"] == "DELETED_IN_TIME" and d["verdict"] == "DELETED_IN_TIME" and w["reservation"]["uid"] == d["uid"] == d["precondition_uid"] == cl.uid
        assert w["first_permit"]["evidence"]["run_ended"] == "registered tree gone" and d["delete_requested_epoch"] >= w["first_permit"]["at"] and w["first_permit"]["evidence"]["gpu_census"]["previous_evaluation_released"] is True
        assert [(x["role"], x["rc"], x["precondition_uid"]) for x in cl.deletes()] == [("deleter", 0, cl.uid)] and cl.deletes("reader") == [] and cl.state()["pods"] == []
        assert w["signals_sent"] == 0 and w["delete_requests_sent"] == 0 and d["delete_requests_accepted"] == 1
    finally:
        for reg in (root / "pids" / "owned.tsv", root / "pids" / "closeout.tsv", tmp_path / "deleter.tsv"):  # nothing should be left; if something is, only through the pinned verified cleanup
            subprocess.run([sys.executable, "-I", str(E1 / "checks" / "e1_pids.py"), "cleanup", "--registry", str(reg), "--execute", "--grace", "2", "--kill9"], capture_output=True, env=env)


# ---- what the tools are not allowed to do


def test_sources_have_no_signalling_call_and_one_delete_site():
    for path in (TOOL, E1 / "checks" / "e1_closeout_verdict.py"):
        code = path.read_text().split('"""', 2)[2]
        assert not any(x in code for x in ("os.kill", "import signal", "killpg", "send_signal", ".terminate(", ".kill(")), path.name
    code = TOOL.read_text()
    watch_src = code[code.index("def watch("):code.index("def armed(")]
    assert "delete_with_uid" not in watch_src and '"delete"' not in watch_src  # the watcher has no way to delete
    assert code.count("delete_with_uid(") == 4  # the definition, the two dry runs of `record`, the one request of `delete`
    assert "dry_run=False" in code[code.index("def delete("):code.index("# ---- command line")] and code.count("dry_run=False") == 1


def test_command_line_rejects_nonsense(tmp_path):
    base = [sys.executable, "-I", str(TOOL)]
    assert subprocess.run([*base, "watch", "--run", str(tmp_path), "--registry", str(tmp_path / "r.tsv"), "--out", str(tmp_path / "o.json"), "--launch-latest-age-s", "470", "--ceiling-s", "0"], capture_output=True).returncode == 2
    assert subprocess.run([*base, "delete", "--reservation", str(tmp_path / "r.json"), "--out", str(tmp_path / "o.json"), "--ceiling-s", "5400"], capture_output=True).returncode == 2  # no permit command
    assert subprocess.run([*base, "armed", "--record", str(tmp_path / "none.json")], capture_output=True).returncode == 1


def test_registered_values():
    r = D["reservation"]
    assert r["hard_ceiling_s"] == 5400 and K["interval_s"] == 5 and K["watcher_tail_s"] == 600 and K["never_launched_margin_s"] == 60 and r["early_release_after_verified_gpu_release"]["enabled"] is False
    arm = (E1 / "arm_closeout_e1_v4scale.sh").read_text()
    assert "--label closeout:$PASS" in arm and "--registry $RUN/pids/closeout.tsv" in arm and "LATEST=$((GUARD - BUDGET))" in arm and "e1_reservation_closeout.py verify" in arm
    driver = (E1 / "run_e1_v4scale.sh").read_text().splitlines()
    order = [next(i for i, ln in enumerate(driver) if key in ln and not ln.startswith("#")) for key in ("e1_reservation_closeout.py verify", "arm_closeout_e1_v4scale.sh", "e1_reservation_closeout.py armed", "e1_reservation_guard.py eligible", "spawn gpuwatch")]
    assert order == sorted(order)  # ownership, arming, armed, launch rule; only then the first observer
