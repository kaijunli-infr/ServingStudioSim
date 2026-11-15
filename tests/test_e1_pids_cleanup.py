"""CPU tests for the experiment-owned process registry and verified termination (phase1c/e1/checks/e1_pids.py).

Positive cases spawn real background processes and register them; negative cases spawn command-line lookalikes that are
NOT registered; the previous failing case (a shell whose command line quotes a planted `tail -F` command, which `pgrep -f`
matched and killed, taking the test shell down) is kept as a regression. After every cleanup the test process and its
parent (the shell / ssh session that runs pytest) must still be alive.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
MARK = "dsv41_flash_gb300_tp4_e1_stage15"           # matches e1_pids.LOOKALIKE_RE

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc based")


def _load():
    spec = importlib.util.spec_from_file_location("e1_pids", E1 / "checks" / "e1_pids.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["e1_pids"] = mod; spec.loader.exec_module(mod)
    return mod


pids = _load()


def alive(pid: int) -> bool:
    return pids.identity(pid) is not None


def wait_gone(pid: int, t=5.0) -> bool:
    end = time.time() + t
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


@pytest.fixture
def reg(tmp_path):
    return str(tmp_path / "pids" / "owned.tsv")


@pytest.fixture
def guard():
    """Every test must leave the test process and its parent alive; spawned processes are reaped at the end."""
    me, parent = os.getpid(), os.getppid()
    spawned: list[int] = []
    yield spawned
    # reap first (never leave test processes behind): the spawned processes and their live descendants (e.g. the sleep child of
    # a lookalike bash, which would otherwise be orphaned and survive), all created by this test; then assert survival
    targets = list(spawned) + list(pids.descendants(set(spawned), pids.ppid_map()))
    for p in targets:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for p in spawned:
        try:
            os.waitpid(p, 0)
        except ChildProcessError:
            pass
    assert alive(me) and alive(parent) and os.getppid() == parent, "test shell / parent must survive"


def spawn_lookalike(guard, cwd=None, text=f"tail -F /x/{MARK}/e1_nsys_val/run.log"):
    """Unregistered lookalikes that are NOT children of the test process (a direct child would be part of our own invocation
    pipeline and is rightly excluded): `setsid -f` double-forks them, so their parent is init / the subreaper, like a watcher
    left behind by an earlier ssh session. Each carries a unique token so the test can find its pid in /proc without any
    experiment pattern, verify its identity, and reap it at the end."""
    import uuid
    tok = uuid.uuid4().hex
    # a bash whose own argv quotes the planted command text; `; true` stops bash from exec'ing the single command, so the
    # bash (with the quoted text in /proc/<pid>/cmdline) stays alive and sleep is its child
    subprocess.run(["setsid", "-f", "bash", "-c", f"sleep 300; true  # {text} {tok}A"], cwd=cwd, check=True,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # a bash carrying the experiment marker as a positional argument (argv lookalike without any experiment command)
    subprocess.run(["setsid", "-f", "bash", "-c", "sleep 300; true", "lookalike", f"#{MARK}", f"{tok}B"], cwd=cwd, check=True,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    found = {}
    for _ in range(50):
        for p in pids.all_pids():
            i = pids.identity(p)
            if i and i["name"] == "bash":
                if f"{tok}A" in i["cmd"]:
                    found["A"] = p
                if f"{tok}B" in i["cmd"]:
                    found["B"] = p
        if len(found) == 2:
            break
        time.sleep(0.1)
    assert len(found) == 2, "lookalikes did not start"
    guard += [found["A"], found["B"]]
    assert os.getpid() not in (pids.identity(found["A"])["ppid"], pids.identity(found["B"])["ppid"])
    return found["A"], found["B"]


def test_register_identity_and_verify(reg, guard):
    rec = pids.spawn(reg, "t:sleep", ["sleep", "300"], None, None); guard.append(rec["pid"])
    ident = pids.identity(rec["pid"])
    assert ident and ident["starttime"] == rec["starttime"] and ident["uid"] == os.getuid()
    recs = pids.read_registry(reg)
    assert len(recs) == 1 and int(recs[0]["pid"]) == rec["pid"] and recs[0]["label"] == "t:sleep"
    assert pids.main(["verify", str(rec["pid"]), str(rec["starttime"])]) == 0
    assert pids.main(["verify", str(rec["pid"]), str(rec["starttime"] + 1)]) == 1


def test_positive_owned_tree_is_terminated_lookalikes_untouched(reg, guard, tmp_path):
    cwd = tmp_path / MARK / "e1_test"; cwd.mkdir(parents=True)
    # owned: a bash with a child sleep (tree), spawned detached in the experiment cwd; and a log watcher
    tree = pids.spawn(reg, "t:tree", ["bash", "-c", "sleep 300 & wait"], str(cwd), None); guard.append(tree["pid"])
    log = tmp_path / "run.log"; log.write_text("x\n")
    watcher = pids.spawn(reg, "t:tail", ["tail", "-F", str(log)], None, None); guard.append(watcher["pid"])
    # not owned: lookalikes (quoted command text in a bash, marker in a sleep argument, cwd in the experiment dir)
    la_bash, la_sleep = spawn_lookalike(guard, cwd=str(cwd))
    time.sleep(0.5)
    pl = pids.plan(reg)
    owned = {o["pid"] for o in pl["owned"]}
    assert tree["pid"] in owned and watcher["pid"] in owned
    child = [o for o in pl["owned"] if o["depth"] == 1]
    assert child, "the owned bash's child sleep must be an owned descendant"
    assert {la_bash, la_sleep} <= {l["pid"] for l in pl["lookalikes"]} and la_bash not in owned and la_sleep not in owned
    assert not pl["refused"]
    rc, res = pids.cleanup(reg, execute=True, grace=5.0, kill9=False, log_path=str(tmp_path / "cleanup.jsonl"))
    assert rc == 2, "lookalikes present -> stop-and-report even though owned processes were terminated"
    for p in [tree["pid"], watcher["pid"]] + [c["pid"] for c in child]:
        assert wait_gone(p), f"owned pid {p} must be terminated"
    assert alive(la_bash) and alive(la_sleep), "lookalikes must never be signalled"
    acts = [a for a in res["actions"] if a["action"] == "SIGTERM"]
    assert {a["pid"] for a in acts} == owned
    assert not res["survivors"]
    line = json.loads((tmp_path / "cleanup.jsonl").read_text().splitlines()[-1])
    assert line["execute"] is True


def test_regression_quoted_tail_command_in_shell_cmdline_survives(reg, guard, tmp_path):
    """Stage 1.5 positive-test failure: `pgrep -f 'tail -F ...run.lo[g]'` matched the test shell's own command line and
    killed it, dropping the ssh session. Here the same quoted text is on an unregistered bash: it must survive cleanup."""
    la_bash, la_sleep = spawn_lookalike(guard)
    victim = pids.spawn(reg, "t:owned", ["sleep", "300"], None, None); guard.append(victim["pid"])
    time.sleep(0.3)
    assert "tail -F" in pids.identity(la_bash)["cmd"] and MARK in pids.identity(la_bash)["cmd"], "the quoted command text must be in the lookalike's cmdline"
    assert MARK in pids.identity(la_sleep)["cmd"]
    pl = pids.plan(reg)
    assert {la_bash, la_sleep} <= {l["pid"] for l in pl["lookalikes"]}
    rc, res = pids.cleanup(reg, execute=True, grace=5.0, kill9=False, log_path=None)
    assert wait_gone(victim["pid"])
    assert alive(la_bash), "the shell quoting the planted tail command must survive (Stage 1.5 regression)"
    assert alive(la_sleep), "lookalike with the experiment marker in its argv must not be signalled"
    assert all(a["pid"] != la_sleep and a["pid"] != la_bash for a in res["actions"])


def test_stale_record_pid_reuse_is_skipped(reg, guard):
    victim = pids.spawn(reg, "t:short", ["sleep", "0.2"], None, None)
    assert wait_gone(victim["pid"], 5.0)
    # forge: the same pid number now belongs to a new live process (simulate reuse by writing a record with a wrong start time)
    live = pids.spawn(reg + ".other", "t:live", ["sleep", "300"], None, None); guard.append(live["pid"])
    with open(reg, "a") as fh:
        fh.write("\t".join(str(x) for x in [live["pid"], live["starttime"] - 12345, live["boot_id"], live["uid"], "t:forged", "now", "h", 0, "sleep 300"]) + "\n")
    pl = pids.plan(reg)
    assert any(int(s["pid"]) == live["pid"] for s in pl["stale"]) and live["pid"] not in {o["pid"] for o in pl["owned"]}
    assert any(int(g["pid"]) == victim["pid"] for g in pl["gone"])
    rc, res = pids.cleanup(reg, execute=True, grace=1.0, kill9=False, log_path=None)
    assert alive(live["pid"]), "a record whose identity does not match the live process must not be signalled"
    assert rc in (0, 2) and not res["actions"]            # 2 only if unrelated lookalikes exist on a shared host


def test_identity_recheck_immediately_before_signal(reg, guard, monkeypatch):
    victim = pids.spawn(reg, "t:owned", ["sleep", "300"], None, None); guard.append(victim["pid"])
    pl = pids.plan(reg)
    target = [o for o in pl["owned"] if o["pid"] == victim["pid"]][0]
    real = pids.identity

    def changed(pid):                                   # the process got replaced between plan and signal
        i = real(pid)
        if i and pid == victim["pid"]:
            i = {**i, "starttime": i["starttime"] + 7}
        return i
    monkeypatch.setattr(pids, "identity", changed)
    log: list = []
    assert pids.signal_verified(target, signal.SIGTERM, log) == "changed"
    monkeypatch.setattr(pids, "identity", real)
    assert alive(victim["pid"]) and log[0]["action"] == "skip"


def test_refuses_own_ancestry_and_other_registered_ancestors(reg, guard):
    for pid, label in ((os.getpid(), "t:self"), (os.getppid(), "t:parent")):
        pids.register(reg, label, pid)
    victim = pids.spawn(reg, "t:owned", ["sleep", "300"], None, None); guard.append(victim["pid"])
    pl = pids.plan(reg)
    assert {int(r["pid"]) for r in pl["refused"]} == {os.getpid(), os.getppid()}
    rc, res = pids.cleanup(reg, execute=True, grace=1.0, kill9=False, log_path=None)
    assert rc == 3 and not res["actions"], "a refusal stops everything: even the valid owned process is not signalled"
    assert alive(victim["pid"]) and alive(os.getppid())


def test_descendant_in_own_ancestry_refuses(reg, guard):
    """If a registered pid is an ancestor of the cleanup (cleanup running inside the experiment tree), refuse."""
    anc = pids.ancestors(os.getpid())
    if len(anc) < 2:
        pytest.skip("not enough ancestry")
    pids.register(reg, "t:grandparent", anc[1])
    pl = pids.plan(reg)
    assert pl["refused"] and not pl["owned"]
    rc, _ = pids.cleanup(reg, execute=True, grace=1.0, kill9=False, log_path=None)
    assert rc == 3 and alive(anc[1]) and alive(os.getppid())


def test_dry_run_signals_nothing(reg, guard):
    victim = pids.spawn(reg, "t:owned", ["sleep", "300"], None, None); guard.append(victim["pid"])
    rc, res = pids.cleanup(reg, execute=False, grace=1.0, kill9=False, log_path=None)
    assert rc in (0, 2) and not res["actions"] and alive(victim["pid"])


def test_cli_round_trip(reg, guard, tmp_path):
    py = sys.executable; script = str(E1 / "checks" / "e1_pids.py")
    out = subprocess.run([py, "-I", script, "spawn", "--registry", reg, "--label", "cli:sleep", "--", "sleep", "300"], capture_output=True, text=True, check=True).stdout
    pid = json.loads(out)["pid"]; guard.append(pid)
    r = subprocess.run([py, "-I", script, "plan", "--registry", reg, "--json", str(tmp_path / "plan.json")], capture_output=True, text=True)
    assert r.returncode in (0, 2) and f"OWNED    pid {pid}" in r.stdout
    r = subprocess.run([py, "-I", script, "cleanup", "--registry", reg], capture_output=True, text=True)
    assert r.returncode in (0, 2) and "DRY-RUN" in r.stdout and alive(pid)
    r = subprocess.run([py, "-I", script, "cleanup", "--registry", reg, "--execute", "--grace", "5", "--json", str(tmp_path / "c.json")], capture_output=True, text=True)
    assert r.returncode in (0, 2) and "survivors 0" in r.stdout and wait_gone(pid)
    assert json.load(open(tmp_path / "c.json"))["actions"][0]["action"] == "SIGTERM"


def test_own_pipeline_sibling_is_not_a_lookalike(reg, guard, tmp_path):
    """GO attempt 1 (2026-11-15 18:26Z): the driver's own `tee .../dsv41_flash_gb300_tp4_e1_stage15/e1_b8_recovery/preflight.log`
    was reported as a lookalike by the preflight running inside the driver, which then refused to start. A process whose parent
    is the invoker (or one of its ancestors) belongs to the invocation pipeline and must not be a lookalike."""
    log = tmp_path / MARK / "e1_b8_recovery" / "preflight.log"; log.parent.mkdir(parents=True)
    # a sibling of this test process: child of our parent shell is not reproducible here, so use a child of *this* process
    # (ppid == me, which is in the excluded set) carrying the marker in argv
    tee = subprocess.Popen(["tee", str(log)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL); guard.append(tee.pid)
    time.sleep(0.3)
    pl = pids.plan(reg)
    assert tee.pid in pl["own_pipeline"]
    assert tee.pid not in {l["pid"] for l in pl["lookalikes"]}
    tee.stdin.close()


@pytest.mark.skipif(subprocess.run(["which", "tmux"], capture_output=True).returncode != 0, reason="tmux not installed")
def test_census_and_plan_pass_from_inside_the_driver_context(reg, tmp_path):
    """Exact shape of run_e1_b8.sh: inside a tmux session named e1b8-*, `bash <check> 2>&1 | tee <run root>/.../preflight.log`.
    Discovery must report CLEAN (own tee and own tmux session excluded) and the registry plan must have 0 lookalikes."""
    run = tmp_path / MARK / "e1_b8_recovery"; run.mkdir(parents=True)
    census = E1 / "checks" / "e1_process_census.sh"; script = E1 / "checks" / "e1_pids.py"
    sess = f"e1b8-test{os.getpid()}"
    done = tmp_path / "done"
    cmd = (f"bash {census} 2>&1 | tee {run}/preflight.log > /dev/null; echo census_rc=${{PIPESTATUS[0]}} >> {run}/preflight.log; "
           f"{sys.executable} -I {script} plan --registry {reg} 2>&1 | tee -a {run}/preflight.log > /dev/null; echo plan_rc=${{PIPESTATUS[0]}} >> {run}/preflight.log; touch {done}")
    subprocess.run(["tmux", "new", "-d", "-s", sess, cmd], check=True)          # default server: what the driver uses
    for _ in range(600):
        if done.exists():
            break
        time.sleep(0.1)
    subprocess.run(["tmux", "kill-session", "-t", sess], capture_output=True)     # only our own, by its unique name
    text = (run / "preflight.log").read_text()
    assert done.exists(), text
    assert "own invocation pipeline" in text and "own tmux session (excluded): " + sess in text, text
    pipe_line = next(l for l in text.splitlines() if "own invocation pipeline" in l)
    pipe = {int(x) for x in re.findall(r"(\d+)\[", pipe_line)}
    assert "[tee]" in pipe_line and pipe, pipe_line                      # our own `| tee` is in the pipeline, not a candidate
    found = {int(x) for l in text.splitlines() if l.lstrip().startswith("FOUND") for x in re.findall(r"\b(\d{2,})\b", l)}
    looka = {int(x) for x in re.findall(r"LOOKALIKE pid (\d+)", text)}
    assert not (pipe & found) and not (pipe & looka), (pipe, found, looka)   # nothing of ours is a candidate / lookalike
    assert "FOUND experiment tmux" not in text, text
    # unrelated candidates may exist on a shared host (e.g. another login shell whose command line names the dirs): then the
    # census reports them and the plan returns 2 (stop-and-report); with none, both must be clean
    if looka or found:
        assert "plan_rc=2" in text, text
    else:
        assert "plan_rc=0" in text and "census_rc=0" in text, text
