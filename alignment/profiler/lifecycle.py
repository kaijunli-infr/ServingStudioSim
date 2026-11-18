"""Profiler-stop lifecycle for Nsight ``cuda_profiler_api`` captures.

Observed on GB300 (six earlier captures and the e1_main_v3 incident): vLLM's
``/stop_profile`` returns only after Nsight has flushed the CUDA trace buffers
*and* imported the stream into the ``.nsys-rep``; the CUDA-owning workers are
blocked inside ``cudaProfilerStop`` for that whole time, so GPU memory stays
held until the report exists. For a 216 M-row trace that takes 20–30 minutes.
A fixed HTTP timeout therefore cannot be the stop contract, and interrupting
the ``nsys`` CLI while it imports destroys the stream.

This module owns the stop as a small state machine with observable phases
(see ``Phase``), one idempotent stop request, a status file that monitors can
read, an operator control file, raw-stream preservation by hard link, explicit
deadlines (soft: escalate; hard: release the GPUs by stopping the *server*,
never the importer), and identity-verified process handling. It changes
nothing about how the server is launched, scheduled or instrumented.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import procident

STOP_ACCEPTED_RE = re.compile(r"api_router\.py:\d+\] Stopping profiler")
STOP_RETURNED_RE = re.compile(r"api_router\.py:\d+\] Profiler stopped")
CAPTURE_ENDED_RE = re.compile(r"Capture range ended in the application")
STREAM_RE = re.compile(r"Generating '([^']+\.qdstrm)'")
IMPORT_PROGRESS_RE = re.compile(r"\[\d+/\d+\] \[=*\s*(\d+)%")
IMPORT_START_RE = re.compile(r"Processing events\.\.\.")
NSYS_ERROR_RE = re.compile(
    r"rpc Shutdown|canceled because the timeout|QdstrmImporter.*(fail|error)"
    r"|Import failed|No space left",
    re.I,
)
API_SERVER_PID_RE = re.compile(r"\(APIServer pid=(\d+)\)")
ENGINE_PID_RE = re.compile(r"\((EngineCore|Worker[^ )]*) pid=(\d+)\)")


class Phase(enum.StrEnum):
    CAPTURING = "capturing"
    STOP_REQUESTED = "stop_requested"  # POST sent; no answer yet
    STOP_ACCEPTED = "stop_accepted"  # API server logged the request
    CAPTURE_ENDED = "capture_ended"  # nsys: capture range ended (flush running)
    STREAM_WRITTEN = "stream_written"  # nsys: raw .qdstrm generated
    IMPORTING = "importing"  # nsys: QdstrmImporter progress visible
    REPORT_WRITTEN = "report_written"  # .nsys-rep exists and stop returned
    STOP_FAILED = "stop_failed"  # stop ended without a report


_ORDER = [
    Phase.CAPTURING,
    Phase.STOP_REQUESTED,
    Phase.STOP_ACCEPTED,
    Phase.CAPTURE_ENDED,
    Phase.STREAM_WRITTEN,
    Phase.IMPORTING,
    Phase.REPORT_WRITTEN,
]
TERMINAL = {Phase.REPORT_WRITTEN, Phase.STOP_FAILED}

#: Sidecar written next to a preserved raw stream once the Nsight session has closed. It lets a
#: later preflight tell a retained, inactive recovery stream from a live profiler session.
RETENTION_NAME = "RETENTION.json"
#: Carried in every final-stream record: a stable file and a matching checksum say the bytes did
#: not change, not that the trace is scientifically complete.
COMPLETENESS_NOTE = (
    "NOT established by a stable size or a matching checksum; requires an offline import and a "
    "semantic comparison (kernel rows, NVTX ranges, iterations, capture boundaries)"
)
MAX_STREAM_OBSERVATIONS = 500


def launch_host_identity() -> dict:
    """Host identity of the process that launches the capture (the launcher itself)."""
    return {
        "hostname": socket.gethostname(),
        "boot_id": procident.boot_id(),
        "launcher_pid": os.getpid(),
        "source": "launcher process: socket.gethostname() and /proc/sys/kernel/random/boot_id",
    }


def file_fingerprint(path: Path) -> dict:
    """Size, mtime and sha256 of a file, with a stat before and after the read."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    after = path.stat()
    return {
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "stable_during_read": (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
    }


def write_retention_record(preserve_dir: Path, record: dict) -> Path:
    target = Path(preserve_dir) / RETENTION_NAME
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str) + "\n")
    os.replace(tmp, target)
    return target


class FinalizationError(RuntimeError):
    """The profiler stop did not produce a report (see ``status``)."""

    def __init__(self, message: str, status: dict):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class FinalizePolicy:
    """Explicit bounds for the stop. Nothing here is extended silently."""

    soft_deadline_s: float = 1800.0  # escalate (log + status) after this
    hard_deadline_s: float = 3600.0  # GPU-release action after this
    stall_timeout_s: float = 900.0  # import progress unchanged → warn
    poll_interval_s: float = 1.0
    on_hard_deadline: str = "release_gpus"  # or "report_only"
    preserve_raw_stream: bool = True
    raw_preserve_dir: Path | None = None  # default: beside the stream (same filesystem)
    min_free_bytes_stream: int = 50 * 2**30
    min_free_bytes_report: int = 20 * 2**30
    session_close_timeout_s: float = 600.0
    server_term_grace_s: float = 60.0
    # Operator extensions (control file `extend <s> <reason>`) may move the hard
    # deadline by at most this much in total; 0 forbids extensions.
    max_extension_s: float = 1200.0
    # Absolute GPU-held ceiling measured from the launch (server spawn), not from
    # the stop request. When it passes, the GPU-release action runs whatever the
    # phase, and no operator extension moves it. None disables it.
    gpu_held_ceiling_s: float | None = None
    # Directory Nsight writes the raw `.qdstrm` into (TMPDIR of the nsys process).
    # None: the launcher's TMPDIR, else /tmp.
    stream_dir: Path | None = None

    def validate(self) -> None:
        for name in (
            "soft_deadline_s",
            "hard_deadline_s",
            "stall_timeout_s",
            "poll_interval_s",
            "session_close_timeout_s",
            "server_term_grace_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"finalize.{name} must be a positive number, got {value!r}")
        if self.hard_deadline_s < self.soft_deadline_s:
            raise ValueError("finalize.hard_deadline_s must be >= soft_deadline_s")
        if self.on_hard_deadline not in ("release_gpus", "report_only"):
            raise ValueError("finalize.on_hard_deadline must be 'release_gpus' or 'report_only'")
        if not isinstance(self.max_extension_s, (int, float)) or self.max_extension_s < 0:
            raise ValueError("finalize.max_extension_s must be a number >= 0")
        if self.gpu_held_ceiling_s is not None and (
            not isinstance(self.gpu_held_ceiling_s, (int, float)) or not self.gpu_held_ceiling_s > 0
        ):
            raise ValueError("finalize.gpu_held_ceiling_s must be a positive number or None")
        if self.gpu_held_ceiling_s is not None and self.gpu_held_ceiling_s < self.hard_deadline_s:
            raise ValueError(
                "finalize.gpu_held_ceiling_s must be >= hard_deadline_s (the ceiling counts "
                "from the launch, the hard deadline from the stop request)"
            )

    def stream_directory(self) -> str:
        return str(self.stream_dir) if self.stream_dir else (os.environ.get("TMPDIR") or "/tmp")


@dataclass
class _Status:
    phase: Phase = Phase.CAPTURING
    phase_history: list[dict] = field(default_factory=list)
    launch_started_at: float | None = None
    stop_requested_at: float | None = None
    http: dict = field(
        default_factory=lambda: {"status": None, "returned_at": None, "requests_sent": 0}
    )
    stream: dict = field(
        default_factory=lambda: {
            "path": None,
            "size": None,  # latest sample while the session runs; NOT the final size
            "preserved_path": None,
            "preserved_size": None,
            "observations": [],  # every distinct (size, mtime) sample with its time and phase
            "final": None,  # size + sha256 taken once, after the Nsight session closed
        }
    )
    host: dict = field(default_factory=dict)
    import_progress_pct: int | None = None
    last_progress_at: float | None = None
    report: dict = field(default_factory=lambda: {"path": None, "size": None, "written_at": None})
    deadlines: dict = field(default_factory=dict)
    escalations: list[dict] = field(default_factory=list)
    control_commands: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    nsys: dict = field(default_factory=dict)
    server: dict = field(default_factory=dict)
    gpu: dict = field(
        default_factory=lambda: {
            "release_requested_at": None,
            "released_at": None,
            "compute_pids": None,
        }
    )
    session_closed_at: float | None = None
    actions: list[dict] = field(default_factory=list)


class ProfilerStopController:
    """One capture, one stop. Every method is safe to call repeatedly."""

    def __init__(
        self,
        *,
        name: str,
        server_log: Path,
        report_stem: Path,
        status_path: Path,
        control_path: Path | None = None,
        policy: FinalizePolicy | None = None,
        stop_request: Callable[[], None],
        nsys_pid: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        wallclock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        disk_usage: Callable[[str], object] = shutil.disk_usage,
        gpu_compute_pids: Callable[[], list[int] | None] | None = None,
        signaller: Callable[
            [procident.ProcessIdentity, int, list], str
        ] = procident.signal_verified,
        identity: Callable[[int], procident.ProcessIdentity | None] = procident.identity,
        emit: Callable[[str], None] = print,
        launch_started_mono: float | None = None,
        launch_started_at: float | None = None,
        stream_scan: Callable[[], list[Path]] | None = None,
        host_identity: Callable[[], dict] = launch_host_identity,
    ):
        self.name = name
        self.server_log = Path(server_log)
        self.report_stem = Path(report_stem)
        self.status_path = Path(status_path)
        self.control_path = (
            Path(control_path) if control_path else self.status_path.with_suffix(".control")
        )
        self.policy = policy or FinalizePolicy()
        self.policy.validate()
        self._stop_request = stop_request
        self._clock, self._wall, self._sleep = clock, wallclock, sleep
        self._disk_usage = disk_usage
        self._gpu_compute_pids = (
            gpu_compute_pids if gpu_compute_pids is not None else _nvidia_smi_compute_pids
        )
        self._signal, self._identity = signaller, identity
        self._emit = emit
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._log_offset = 0
        self._log_tail = b""
        self._control_offset = 0
        self._soft_fired = False
        self._stall_warned_at: float | None = None
        self._hard_extension_s = 0.0
        self._gpus_released_action_done = False
        self._launch_mono = launch_started_mono if launch_started_mono is not None else clock()
        self._stream_scan = stream_scan if stream_scan is not None else self._scan_stream_dir
        self.s = _Status()
        self.s.launch_started_at = (
            launch_started_at if launch_started_at is not None else wallclock()
        )
        self.s.host = {**host_identity(), "recorded_at": wallclock()}
        self.nsys: procident.ProcessIdentity | None = None
        self.api_server: procident.ProcessIdentity | None = None
        self.engine_procs: list[procident.ProcessIdentity] = []
        if nsys_pid is not None:
            self.nsys = self._identity(nsys_pid)
            self.s.nsys = {"pid": nsys_pid, "identity": self.nsys.to_dict() if self.nsys else None}
        self._write_status()

    # ------------------------------------------------------------------ phases
    def _set_phase(self, phase: Phase, **extra) -> None:
        if phase == self.s.phase:
            return
        # never move backwards except into STOP_FAILED
        if phase != Phase.STOP_FAILED and self.s.phase in TERMINAL:
            return
        if phase != Phase.STOP_FAILED and _ORDER.index(phase) < _ORDER.index(self.s.phase):
            return
        self.s.phase = phase
        self.s.phase_history.append(
            {"phase": phase.value, "at": self._wall(), "mono": self._clock(), **extra}
        )
        self._emit(f"[lifecycle] {self.name}: {phase.value}" + (f" {extra}" if extra else ""))
        self._write_status()

    def warn(self, message: str, **extra) -> None:
        self.s.warnings.append({"at": self._wall(), "message": message, **extra})
        self._emit(f"[lifecycle] WARNING {message} {extra if extra else ''}")
        self._write_status()

    # ----------------------------------------------------------------- servers
    def discover_server_processes(self) -> None:
        """Record the vLLM processes from their own log lines, verified as nsys descendants."""
        if self.api_server is not None:
            return
        try:
            text = self.server_log.read_text(errors="replace")
        except OSError:
            return
        m = API_SERVER_PID_RE.search(text)
        if not m:
            return
        pid = int(m.group(1))
        ident = self._identity(pid)
        if ident is None:
            return
        if self.nsys is not None and not procident.is_descendant_of(pid, self.nsys.pid):
            self.warn(
                "API server pid from the log is not a descendant of the nsys process; not recorded",
                pid=pid,
            )
            return
        self.api_server = ident
        seen = {pid}
        for role, p in ENGINE_PID_RE.findall(text):
            p = int(p)
            if p in seen:
                continue
            seen.add(p)
            e = self._identity(p)
            if e is not None and (
                self.nsys is None or procident.is_descendant_of(p, self.nsys.pid)
            ):
                self.engine_procs.append(e)
        self.s.server = {
            "api_server": ident.to_dict(),
            "engine_processes": [e.to_dict() for e in self.engine_procs],
        }
        self._write_status()

    def _server_alive(self) -> bool:
        procs = [p for p in [self.api_server, *self.engine_procs] if p is not None]
        return any(procident.same_identity(p, self._identity(p.pid)) for p in procs)

    def _nsys_alive(self) -> bool:
        return self.nsys is not None and procident.same_identity(
            self.nsys, self._identity(self.nsys.pid)
        )

    # ------------------------------------------------------------- stop request
    def request_stop(self) -> None:
        """Send exactly one ``/stop_profile``; later calls are no-ops."""
        with self._lock:
            if self._thread is not None:
                self.s.http["duplicate_requests_suppressed"] = (
                    self.s.http.get("duplicate_requests_suppressed", 0) + 1
                )
                self._write_status()
                return
            self.s.stop_requested_at = self._wall()
            self.s.deadlines = {
                "soft_at": self.s.stop_requested_at + self.policy.soft_deadline_s,
                "hard_at": self.s.stop_requested_at + self.policy.hard_deadline_s,
                "hard_extension_s": 0.0,
                "max_extension_s": self.policy.max_extension_s,
                "gpu_held_ceiling_at": (
                    None
                    if self.policy.gpu_held_ceiling_s is None
                    else self.s.launch_started_at + self.policy.gpu_held_ceiling_s
                ),
            }
            self._stop_mono = self._clock()
            try:
                self._log_offset = self.server_log.stat().st_size
            except OSError:
                self._log_offset = 0
            self._check_disk_before_stop()
            self.discover_server_processes()
            self.s.http["requests_sent"] = 1
            self._thread = threading.Thread(
                target=self._post, name="profiler-stop-request", daemon=True
            )
            self._thread.start()
            self._set_phase(Phase.STOP_REQUESTED)

    def _post(self) -> None:
        try:
            self._stop_request()
            self.s.http["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - recorded, decided in poll()
            self.s.http["status"] = f"error: {error}"
        self.s.http["returned_at"] = self._wall()

    def _check_disk_before_stop(self) -> None:
        stream_dir = self.policy.stream_directory()
        for label, path, minimum in (
            ("stream", stream_dir, self.policy.min_free_bytes_stream),
            ("report", str(self.report_stem.parent), self.policy.min_free_bytes_report),
        ):
            try:
                free = self._disk_usage(path).free
            except OSError as error:
                self.warn(f"cannot stat free space for {label}", path=path, error=str(error))
                continue
            if free < minimum:
                self.warn(
                    f"low free space for the {label} directory",
                    path=path,
                    free_bytes=free,
                    minimum_bytes=minimum,
                )

    # -------------------------------------------------------------------- poll
    def poll(self) -> Phase:
        """Advance the state from the server log, files and processes. Non-blocking."""
        self._scan_log()
        self._track_stream()
        self._check_report()
        self._check_processes()
        self._write_status()
        return self.s.phase

    def _scan_log(self) -> None:
        try:
            with self.server_log.open("rb") as fh:
                fh.seek(self._log_offset)
                chunk = fh.read()
        except OSError:
            return
        if not chunk:
            return
        self._log_offset += len(chunk)
        data = self._log_tail + chunk
        parts = re.split(rb"[\r\n]", data)
        self._log_tail = parts.pop()  # unfinished line (progress bars end without a newline)
        for raw in parts:
            line = raw.decode(errors="replace")
            if not line:
                continue
            if STOP_ACCEPTED_RE.search(line):
                self._set_phase(Phase.STOP_ACCEPTED)
            if CAPTURE_ENDED_RE.search(line):
                self._set_phase(Phase.CAPTURE_ENDED)
            m = STREAM_RE.search(line)
            if m:
                self.s.stream["path"] = m.group(1)
                self._set_phase(Phase.STREAM_WRITTEN, path=m.group(1))
            if IMPORT_START_RE.search(line):
                self._set_phase(Phase.IMPORTING)
            for pm in IMPORT_PROGRESS_RE.finditer(line):
                pct = int(pm.group(1))
                if self.s.import_progress_pct is None or pct > self.s.import_progress_pct:
                    self.s.import_progress_pct = pct
                    self.s.last_progress_at = self._clock()
                self._set_phase(Phase.IMPORTING)
            if STOP_RETURNED_RE.search(line):
                self.s.http.setdefault("returned_logged_at", self._wall())
            if NSYS_ERROR_RE.search(line):
                self.warn("nsys reported a problem", line=line[:200])
        # a progress bar with no trailing newline: parse the pending tail too
        for pm in IMPORT_PROGRESS_RE.finditer(self._log_tail.decode(errors="replace")):
            pct = int(pm.group(1))
            if self.s.import_progress_pct is None or pct > self.s.import_progress_pct:
                self.s.import_progress_pct = pct
                self.s.last_progress_at = self._clock()
            self._set_phase(Phase.IMPORTING)

    def _track_stream(self) -> None:
        path = self.s.stream.get("path")
        if not path and self.s.stop_requested_at is not None:
            # Nsight announces the stream ("Generating '<x>.qdstrm'") only for large
            # captures; for the others the file still exists in the stream directory
            # while the stop runs. Find it by its open descriptor in the nsys tree.
            found = self._stream_scan()
            if found:
                path = str(found[0])
                self.s.stream["path"] = path
                self.s.stream["discovered_by"] = "fd-scan"
                if len(found) > 1:
                    self.warn(
                        "several raw streams open in the nsys process tree; tracking the first",
                        candidates=[str(p) for p in found],
                    )
                self._emit(f"[lifecycle] raw stream found by descriptor scan: {path}")
        if not path:
            return
        if self.s.stream.get("discovered_by") is None:
            self.s.stream["discovered_by"] = "log"
        try:
            size = os.stat(path).st_size
        except OSError:
            size = None
        if size is not None:
            self.s.stream["size"] = size
        self._observe_stream(path)
        if self.policy.preserve_raw_stream and self.s.stream.get("preserved_path") is None and size:
            self._preserve_stream(Path(path))
        elif self.s.stream.get("preserved_path"):
            try:
                self.s.stream["preserved_size"] = os.stat(self.s.stream["preserved_path"]).st_size
            except OSError:
                pass

    def _observe_stream(self, path: str) -> None:
        """Keep every distinct (size, mtime) sample of the stream; none is overwritten."""
        target = self.s.stream.get("preserved_path") or path
        try:
            st = os.stat(target)
        except OSError:
            return
        observations = self.s.stream.setdefault("observations", [])
        last = observations[-1] if observations else None
        if last is not None and (last["size"], last["mtime_ns"]) == (st.st_size, st.st_mtime_ns):
            return
        if len(observations) >= MAX_STREAM_OBSERVATIONS:
            dropped = self.s.stream.get("observations_dropped", 0)
            self.s.stream["observations_dropped"] = dropped + 1
            return
        observations.append(
            {
                "at": self._wall(),
                "phase": self.s.phase.value,
                "path": str(target),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )

    def _finalize_stream_record(self) -> None:
        """Once, after the Nsight session closed: final size and sha256 of the stream.

        The record states what was measured and when. It does not claim the trace
        is complete (see COMPLETENESS_NOTE); that needs the offline import.
        """
        stream = self.s.stream
        preserved = stream.get("preserved_path")
        candidates = [p for p in (preserved, stream.get("path")) if p]
        target = next((Path(p) for p in candidates if Path(p).is_file()), None)
        final: dict = {
            "taken_at": self._wall(),
            "basis": "after the nsys session closed (no writer left)",
            "session_closed_at": self.s.session_closed_at,
            "phase": self.s.phase.value,
            "completeness": COMPLETENESS_NOTE,
        }
        if target is None:
            final.update(available=False, reason="no stream file exists at the recorded paths")
            if stream.get("path") or preserved:
                self.warn("final stream record: no stream file available", paths=candidates)
            stream["final"] = final
            return
        try:
            final.update(available=True, path=str(target), **file_fingerprint(target))
        except OSError as error:
            final.update(available=False, reason=f"could not read the stream: {error}")
            self.warn("final stream record: stream unreadable", path=str(target), error=str(error))
            stream["final"] = final
            return
        earlier = stream.get("observations") or []
        final["changed_after_last_observation"] = bool(earlier) and (
            earlier[-1]["size"],
            earlier[-1]["mtime_ns"],
        ) != (final["size"], final["mtime_ns"])
        stream["final"] = final
        self._emit(
            f"[lifecycle] final stream record: {final['size']} B sha256 {final['sha256']} "
            f"(session closed; completeness not implied)"
        )
        if preserved and Path(preserved).is_file():
            try:
                write_retention_record(
                    Path(preserved).parent,
                    {
                        "schema_version": 1,
                        "name": self.name,
                        "stream_file": Path(preserved).name,
                        "size": final["size"],
                        "sha256": final["sha256"],
                        "mtime_ns": final["mtime_ns"],
                        "session_closed_at": self.s.session_closed_at,
                        "final_phase": self.s.phase.value,
                        "host": self.s.host.get("hostname"),
                        "nsys": self.s.nsys.get("identity"),
                        "status_path": str(self.status_path),
                        "written_by": "lifecycle controller",
                        "completeness": COMPLETENESS_NOTE,
                    },
                )
            except OSError as error:
                self.warn("could not write the retention record", error=str(error))

    def _preserve_stream(self, path: Path) -> None:
        """Hard-link the raw stream so Nsight's own unlink cannot destroy it."""
        target_dir = self.policy.raw_preserve_dir or (path.parent / f"{self.name}.raw-preserve")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{self.name}.qdstrm"
            if target.exists():
                target.unlink()
            os.link(path, target)
        except OSError as error:
            self.warn(
                "could not hard-link the raw stream; it is unprotected",
                path=str(path),
                error=str(error),
            )
            self.s.stream["preserved_path"] = False
            return
        self.s.stream["preserved_path"] = str(target)
        self.s.stream["preserved_size"] = target.stat().st_size
        self._emit(f"[lifecycle] raw stream preserved by hard link: {target}")

    def _scan_stream_dir(self) -> list[Path]:
        """`nsys-report-*.qdstrm` files in the stream directory held open by nsys or a descendant.

        Only descriptor-confirmed files are returned: a stream that merely sits in
        the directory could belong to another user's session on a shared tray.
        """
        if self.nsys is None:
            return []
        directory = Path(self.policy.stream_directory())
        try:
            candidates = {p.resolve() for p in directory.glob("nsys-report-*.qdstrm")}
        except OSError:
            return []
        if not candidates:
            return []
        found: list[Path] = []
        for pid in [self.nsys.pid, *procident.descendants(self.nsys.pid)]:
            try:
                fds = os.listdir(f"/proc/{pid}/fd")
            except OSError:
                continue
            for fd in fds:
                try:
                    target = Path(os.readlink(f"/proc/{pid}/fd/{fd}"))
                except OSError:
                    continue
                if target in candidates and target not in found:
                    found.append(target)
        return sorted(found)

    def _report_path(self) -> Path | None:
        for cand in (self.report_stem.with_suffix(".nsys-rep"), self.report_stem):
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
        return None

    def _check_report(self) -> None:
        if self.s.phase in TERMINAL:
            return
        rep = self._report_path()
        returned = (
            self.s.http.get("status") == "ok" or self.s.http.get("returned_logged_at") is not None
        )
        if rep is not None and returned:
            self.s.report = {
                "path": str(rep),
                "size": rep.stat().st_size,
                "written_at": self._wall(),
            }
            self._set_phase(Phase.REPORT_WRITTEN, path=str(rep))
            return
        status = self.s.http.get("status")
        if status and status != "ok":
            # The request failed (server gone, connection reset). Give a late report a short grace.
            returned_at = self.s.http.get("returned_at") or self._wall()
            if rep is not None:
                self.s.report = {
                    "path": str(rep),
                    "size": rep.stat().st_size,
                    "written_at": self._wall(),
                }
                self._set_phase(
                    Phase.REPORT_WRITTEN, path=str(rep), note="report present despite HTTP error"
                )
            elif self._wall() - returned_at > 30.0 and not self._nsys_alive():
                self._set_phase(Phase.STOP_FAILED, reason=status)

    def _check_processes(self) -> None:
        if self.nsys is not None:
            alive = self._nsys_alive()
            self.s.nsys["alive"] = alive
            if not alive and self.s.session_closed_at is None:
                self.s.session_closed_at = self._wall()
                self._emit(f"[lifecycle] nsys session closed (pid {self.nsys.pid} exited)")
                if self.s.phase not in TERMINAL:
                    rep = self._report_path()
                    if rep is not None:
                        self.s.report = {
                            "path": str(rep),
                            "size": rep.stat().st_size,
                            "written_at": self._wall(),
                        }
                        self._set_phase(Phase.REPORT_WRITTEN, path=str(rep))
                    else:
                        self._set_phase(
                            Phase.STOP_FAILED, reason="nsys exited before a report was written"
                        )
        if self.api_server is not None:
            alive = self._server_alive()
            self.s.server["alive"] = alive
            if not alive and self.s.gpu.get("released_at") is None:
                self._record_gpu_release("server processes exited")
        if self.s.gpu.get("released_at") is None:
            pids = self._gpu_compute_pids()
            self.s.gpu["compute_pids"] = pids
            if pids is not None and self.api_server is not None:
                ours = {p.pid for p in [self.api_server, *self.engine_procs]}
                if not (set(pids) & ours) and not self._server_alive():
                    self._record_gpu_release("no compute process of this server remains")
        if (
            self.s.session_closed_at is not None
            and self.s.phase in TERMINAL
            and self.s.stream.get("final") is None
        ):
            self._track_stream()
            self._finalize_stream_record()
            self._write_status()

    def _record_gpu_release(self, reason: str) -> None:
        self.s.gpu["released_at"] = self._wall()
        self._emit(f"[lifecycle] GPUs released ({reason}) at {self.s.gpu['released_at']:.0f}")
        self._write_status()

    # ------------------------------------------------------------------- waits
    def wait_for_report(self) -> Phase:
        """Block until the stop reaches a terminal phase or the hard deadline acted.

        Soft deadline: escalation record. Stalled import: warning. Hard deadline:
        the configured GPU-release action (stop the *server*; the importer is
        never signalled). After the hard deadline this returns the current
        phase; the caller decides what to do with a non-terminal result.
        """
        if self._thread is None:
            self.request_stop()
        while True:
            phase = self.poll()
            self._apply_control()
            if phase in TERMINAL:
                return phase
            elapsed = self._clock() - self._stop_mono
            if not self._soft_fired and elapsed > self.policy.soft_deadline_s:
                self._soft_fired = True
                self.s.escalations.append(
                    {
                        "at": self._wall(),
                        "kind": "soft_deadline",
                        "elapsed_s": elapsed,
                        "phase": phase.value,
                        "import_progress_pct": self.s.import_progress_pct,
                    }
                )
                self._emit(
                    f"[lifecycle] ESCALATION soft deadline passed ({elapsed:.0f} s) "
                    f"in phase {phase.value}; still waiting"
                )
                self._write_status()
            if (
                phase == Phase.IMPORTING
                and self.s.last_progress_at is not None
                and self._clock() - self.s.last_progress_at > self.policy.stall_timeout_s
                and (
                    self._stall_warned_at is None
                    or self._clock() - self._stall_warned_at > self.policy.stall_timeout_s
                )
            ):
                self._stall_warned_at = self._clock()
                self.warn(
                    "import progress stalled",
                    progress_pct=self.s.import_progress_pct,
                    since_s=self._clock() - self.s.last_progress_at,
                )
            held = self._clock() - self._launch_mono
            ceiling = self.policy.gpu_held_ceiling_s
            if ceiling is not None and held > ceiling:
                kind, reason, shown = "gpu_held_ceiling", "GPU-held ceiling", held
            elif elapsed > self.policy.hard_deadline_s + self._hard_extension_s:
                kind, reason, shown = "hard_deadline", "hard deadline", elapsed
            else:
                self._sleep(self.policy.poll_interval_s)
                continue
            self.s.escalations.append(
                {
                    "at": self._wall(),
                    "kind": kind,
                    "elapsed_s": shown,
                    "phase": phase.value,
                    "import_progress_pct": self.s.import_progress_pct,
                }
            )
            self._emit(
                f"[lifecycle] {reason.upper()} passed ({shown:.0f} s) in phase {phase.value}; "
                f"action={self.policy.on_hard_deadline}"
            )
            if self.policy.on_hard_deadline == "release_gpus":
                self.release_gpus(reason=reason)
                return self._wait_session_close()
            self._write_status()
            return self.poll()

    def _wait_session_close(self) -> Phase:
        deadline = self._clock() + self.policy.session_close_timeout_s
        while self._clock() < deadline:
            phase = self.poll()
            if phase in TERMINAL and not self._nsys_alive():
                return phase
            self._sleep(self.policy.poll_interval_s)
        if self._nsys_alive():
            self.warn(
                "nsys still running after the session-close timeout; left untouched "
                "(importer may be active)",
                pid=self.nsys.pid,
                phase=self.s.phase.value,
            )
        return self.poll()

    def _apply_control(self) -> None:
        """Operator commands, one per line; every line is recorded with its result.

        ``extend <seconds> <reason>`` moves the hard deadline (never the GPU-held
        ceiling) by at most ``max_extension_s`` in total; ``release-gpus <reason>``
        runs the release action; ``note <text>`` is recorded. A reason is mandatory
        for the first two. Only complete lines (ending in a newline) are consumed.
        """
        try:
            with self.control_path.open("rb") as fh:
                fh.seek(self._control_offset)
                data = fh.read()
        except OSError:
            return
        if not data:
            return
        complete = data.rfind(b"\n") + 1
        if complete == 0:
            return  # a line still being written
        self._control_offset += complete
        for line in data[:complete].decode(errors="replace").splitlines():
            text = line.strip()
            if not text:
                continue
            words = text.split(None, 2)
            record = {"at": self._wall(), "command": text, "phase": self.s.phase.value}
            if words[0] == "extend":
                record["result"] = self._control_extend(words)
            elif words[0] == "release-gpus":
                if len(words) < 2:
                    record["result"] = "rejected: a reason is required"
                else:
                    self.release_gpus(reason=f"operator: {text}")
                    record["result"] = "release requested"
            elif words[0] == "note":
                record["result"] = "noted"
            else:
                record["result"] = "rejected: unknown command"
            self.s.control_commands.append(record)
            self._emit(f"[lifecycle] CONTROL {record}")
        self._write_status()

    def _control_extend(self, words: list[str]) -> str:
        if len(words) < 3:
            return "rejected: usage is 'extend <seconds> <reason>'"
        try:
            seconds = float(words[1])
        except ValueError:
            return "rejected: seconds not a number"
        if not seconds > 0:
            return "rejected: seconds must be positive"
        if self._hard_extension_s + seconds > self.policy.max_extension_s:
            return (
                f"rejected: total extension {self._hard_extension_s + seconds:.0f} s would exceed "
                f"max_extension_s {self.policy.max_extension_s:.0f}"
            )
        if self.s.stop_requested_at is None:
            return "rejected: no stop requested yet"
        self._hard_extension_s += seconds
        self.s.deadlines["hard_extension_s"] = self._hard_extension_s
        self.s.deadlines["hard_at"] = (
            self.s.stop_requested_at + self.policy.hard_deadline_s + self._hard_extension_s
        )
        ceiling = self.s.deadlines.get("gpu_held_ceiling_at")
        note = "" if ceiling is None else f"; GPU-held ceiling unchanged at {ceiling:.0f}"
        return f"hard deadline extended by {seconds:.0f} s (operator){note}"

    # ------------------------------------------------------------ GPU release
    def release_gpus(self, *, reason: str) -> None:
        """Stop the vLLM server (verified SIGTERM, then SIGKILL survivors). Never touches nsys."""
        if self._gpus_released_action_done:
            return
        self._gpus_released_action_done = True
        self.s.gpu["release_requested_at"] = self._wall()
        self.s.gpu["release_reason"] = reason
        # what the capture was doing when the release was requested (evidence, not a decision input)
        self.s.gpu["release_context"] = {
            "phase": self.s.phase.value,
            "import_progress_pct": self.s.import_progress_pct,
            "nsys_alive": self._nsys_alive() if self.nsys is not None else None,
            "stream_size": self.s.stream.get("size"),
            "report_written": self.s.report.get("written_at") is not None,
        }
        self.discover_server_processes()
        if self.api_server is None:
            self.warn(
                "GPU release requested but no server process is recorded; nothing signalled",
                reason=reason,
            )
            return
        if self.s.stream.get("path") and not self.s.stream.get("preserved_path"):
            self._track_stream()
        log: list[dict] = []
        outcome = self._signal(self.api_server, signal.SIGTERM, log)
        sigterm_at = self._wall()
        for entry in log:
            entry.setdefault("at", sigterm_at)
        # exact time of the API-server SIGTERM and what the importer looked like right after it
        self.s.gpu["api_server_sigterm"] = {
            "at": sigterm_at,
            "pid": self.api_server.pid,
            "outcome": outcome,
        }
        self.s.gpu["importer_after_signal"] = [self._importer_state("after the API-server SIGTERM")]
        self._write_status()
        deadline = self._clock() + self.policy.server_term_grace_s
        while self._clock() < deadline and self._server_alive():
            self._sleep(min(0.5, self.policy.poll_interval_s))
        for p in [self.api_server, *self.engine_procs]:
            if procident.same_identity(p, self._identity(p.pid)):
                before = len(log)
                self._signal(p, signal.SIGKILL, log)
                for entry in log[before:]:
                    entry.setdefault("at", self._wall())
        self.s.gpu["importer_after_signal"].append(
            self._importer_state("after the server termination pass")
        )
        self.s.actions.extend([{"at": self._wall(), "reason": reason, **entry} for entry in log])
        self._emit(f"[lifecycle] GPU release ({reason}): {log}")
        self._write_status()

    def _importer_state(self, label: str) -> dict:
        """Identity and state of the Nsight process (the importer's process), for the record."""
        state: dict = {
            "at": self._wall(),
            "label": label,
            "phase": self.s.phase.value,
            "import_progress_pct": self.s.import_progress_pct,
            "nsys_pid": self.nsys.pid if self.nsys is not None else None,
            "identity_matches": self._nsys_alive() if self.nsys is not None else None,
        }
        if self.nsys is not None:
            try:
                with open(f"/proc/{self.nsys.pid}/stat") as fh:
                    stat = fh.read()
                state["proc_state"] = stat[stat.rindex(")") + 2 :].split()[0]
            except (OSError, ValueError, IndexError):
                state["proc_state"] = None
        return state

    # --------------------------------------------------------------- shutdown
    def shutdown(self, proc: subprocess.Popen) -> None:
        """End the launch after the stop reached a terminal phase (or never started).

        Order: stop the server by its own pid (verified), wait for the nsys CLI to
        exit on its own, and only then — with the report written — interrupt a
        lingering CLI. An active importer is never signalled as timeout handling.
        """
        if proc.poll() is not None:
            self.poll()
            return
        if self._thread is not None and self.s.phase not in TERMINAL:
            # A stop is in flight: finish it under the explicit deadlines first.
            self.wait_for_report()
        if self.api_server is None:
            self.discover_server_processes()
        if self.api_server is not None and self._server_alive():
            log: list[dict] = []
            self._signal(self.api_server, signal.SIGTERM, log)
            self.s.actions.extend(
                [{"at": self._wall(), "reason": "shutdown", **entry} for entry in log]
            )
        elif self._thread is None:
            # Capture never armed (startup failure): nsys handles SIGINT itself here.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                self.s.actions.append(
                    {
                        "at": self._wall(),
                        "reason": "shutdown before capture",
                        "action": "SIGINT pgid",
                    }
                )
            except ProcessLookupError:
                pass
        deadline = self._clock() + self.policy.session_close_timeout_s
        while self._clock() < deadline and proc.poll() is None:
            self.poll()
            self._sleep(self.policy.poll_interval_s)
        if proc.poll() is None:
            importing = self.s.phase == Phase.IMPORTING or (
                self.s.last_progress_at is not None
                and self._clock() - self.s.last_progress_at < self.policy.stall_timeout_s
                and self.s.phase not in TERMINAL
            )
            if self.s.phase == Phase.REPORT_WRITTEN or not importing:
                try:
                    os.kill(proc.pid, signal.SIGINT)
                    self.s.actions.append(
                        {
                            "at": self._wall(),
                            "reason": "nsys lingering after report",
                            "action": "SIGINT nsys",
                        }
                    )
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.kill(proc.pid, signal.SIGKILL)
                    self.s.actions.append(
                        {
                            "at": self._wall(),
                            "reason": "nsys ignored SIGINT",
                            "action": "SIGKILL nsys",
                        }
                    )
            else:
                self.warn(
                    "nsys still importing at shutdown; left running for the operator "
                    "(status file has the pids)",
                    pid=proc.pid,
                )
        self.poll()

    # ----------------------------------------------------------------- status
    def status_dict(self) -> dict:
        s = self.s
        return {
            # 2: host identity, stream observations + final record, release context
            "schema_version": 2,
            "name": self.name,
            "host": s.host.get("hostname"),
            "host_identity": s.host,
            "phase": s.phase.value,
            "phase_history": s.phase_history,
            "launch_started_at": s.launch_started_at,
            "stop_requested_at": s.stop_requested_at,
            "http": s.http,
            "stream": s.stream,
            "import_progress_pct": s.import_progress_pct,
            "report": s.report,
            "deadlines": s.deadlines,
            "policy": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in self.policy.__dict__.items()
            },
            "escalations": s.escalations,
            "control_commands": s.control_commands,
            "warnings": s.warnings,
            "nsys": s.nsys,
            "server": s.server,
            "gpu": s.gpu,
            "session_closed_at": s.session_closed_at,
            "actions": s.actions,
            "server_log": str(self.server_log),
            "control_path": str(self.control_path),
            "updated_at": self._wall(),
        }

    def _write_status(self) -> None:
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.status_dict(), indent=1, default=str))
            os.replace(tmp, self.status_path)
        except OSError:
            pass


def _nvidia_smi_compute_pids() -> list[int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def ignore_sighup() -> None:
    """Keep finalisation alive if the tmux/ssh session that started the launcher goes away.

    Called after the server is spawned so the children do not inherit it.
    """
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (ValueError, OSError):  # not the main thread, or unsupported
        pass


def read_status(status_path: Path) -> dict:
    return json.loads(Path(status_path).read_text())
