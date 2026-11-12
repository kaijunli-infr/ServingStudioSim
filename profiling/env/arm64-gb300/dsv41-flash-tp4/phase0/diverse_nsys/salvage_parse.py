"""Parse an already-exported nsys sqlite for a profile pass whose launcher run aborted after the capture.
Replicates alignment.runner.run_profile's parse section (no server, no GPU). Usage: salvage_parse.py <profile.json>"""
import inspect, json, sys
from pathlib import Path
from launcher.alignment import load_profile_config
from alignment.runner import _engine
from alignment.profiler import nsys_capture, record_extraction
from alignment.nsys.parse import parse_host_timeline, parse_trace, parsed_window_ns, write_kernel_sequences
from alignment.nsys.parsed_io import write_parsed
print("load_profile_config signature:", inspect.signature(load_profile_config))
cfg_path = Path(sys.argv[1])
try:
    cfg = load_profile_config(cfg_path, require_python_runtime=False)
except TypeError:
    cfg = load_profile_config(cfg_path)
log_dir = Path(cfg.log_dir)
driver, records, _ = _engine(cfg)
server_log = next(Path(log_dir, "vllm").glob("*_server.log"))
metrics_jsonl = next(Path(log_dir, "vllm").glob("*_metrics.jsonl"))
rep = next(Path(log_dir, "nsys").glob("*.nsys-rep"))
sqlite_path = rep.with_suffix(".sqlite")
validation = nsys_capture.validate_export(sqlite_path)
print("[salvage] export validation:", validation)
dp_rank_by_device = record_extraction.extract_dp_rank_by_device(server_log, records=records)
worker_ranks = driver.extract_worker_device_ranks(server_log)
print("[salvage] dp_rank_by_device", dp_rank_by_device, "worker_ranks", worker_ranks)
parsed = parse_trace(sqlite_path, metrics_jsonl, cfg.nsys.analyze_iteration_start, cfg.nsys.analyze_iteration_end,
                     range_mode="phases", worker_ranks=worker_ranks, tp_size=cfg.server.tp_size, dp_rank_by_device=dp_rank_by_device)
print("[salvage] devices", parsed["device_ids"], "dp ranks", parsed["dp_rank_by_device"])
parsed_path = log_dir / "parsed.json"
write_parsed(parsed_path, parsed)
write_kernel_sequences(log_dir / "kernel_sequences.json", parsed, parsed_path)
print(f"[salvage] parsed {len(parsed['iterations'])} iteration(s), {parsed['scanned_kernel_rows']} kernel row(s) -> {parsed_path}")
w0, w1 = parsed_window_ns(parsed)
ht = parse_host_timeline(sqlite_path, w0, w1)
(log_dir / "host_timeline.json").write_text(json.dumps(ht, separators=(",", ":")))
print(f"[salvage] host {len(ht['threads'])} thread(s), {len(ht['nvtx_ranges'])} nvtx range(s), {len(ht['api_calls'])} api call(s)")
(log_dir / "salvage_result.json").write_text(json.dumps({"note": "parsed from the 60 s capture after the launcher aborted on request-record validation (engine died after the capture stop)", "validation": validation, "iterations": len(parsed["iterations"]), "kernel_rows": parsed["scanned_kernel_rows"], "sqlite": str(sqlite_path)}, indent=2))
