#!/usr/bin/env python3
"""Apply the `--stream` flag to a req-frontend (session_runner) checkout: asserted string edits, idempotent.

    python3 patch/apply_stream_flag_rs.py <req-frontend checkout>

Then: cargo build --release --bin session_runner && cargo test ; git diff > patch/req-frontend-stream-flag.patch.
The req-frontend tree is a submodule of ServingStudioSim (gitlink 2cd7990c); this script does not commit.
"""
from __future__ import annotations

import sys
from pathlib import Path


def edit(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"already applied: {path} ({marker[:40]!r})")
        return
    assert text.count(old) == 1, f"{path}: anchor not unique/found: {old[:70]!r}"
    path.write_text(text.replace(old, new))
    print(f"edited {path}")


def main(root: str) -> None:
    r = Path(root).resolve()
    cli = r / "src" / "cli.rs"
    client = r / "src" / "backend" / "client.rs"

    edit(cli,
         "    /// Max seconds to wait for the next streaming chunk before failing a request.\n    #[arg(long, default_value_t = 600)]\n    pub stream_idle_timeout_secs: u64,\n",
         "    /// Ask the server to stream (`\"stream\": true`, the default) or to return one completed body.\n"
         "    ///\n"
         "    /// `--stream false` makes a run non-streaming *without* capturing routes, so a timing arm can\n"
         "    /// use the same wire mode as a `--routed-experts-dir` arm (Phase 1C E1: the hook-off control).\n"
         "    /// Non-streaming runs have no per-event timeline: TTFT equals the total duration and TPOT is\n"
         "    /// not measured. Capturing routes always implies non-streaming, whatever this flag says.\n"
         "    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]\n"
         "    pub stream: bool,\n\n"
         "    /// Max seconds to wait for the next streaming chunk before failing a request.\n    #[arg(long, default_value_t = 600)]\n    pub stream_idle_timeout_secs: u64,\n",
         "pub stream: bool,")
    edit(cli,
         "    #[test]\n    fn retired_split_format_and_family_flags_are_rejected() {",
         "    #[test]\n    fn streaming_is_on_by_default_and_can_be_turned_off() {\n"
         "        let parsed = Args::try_parse_from(required_arguments()).unwrap();\n"
         "        assert!(parsed.stream);\n\n"
         "        let mut off = required_arguments();\n"
         "        off.extend([\"--stream\", \"false\"]);\n"
         "        assert!(!Args::try_parse_from(off).unwrap().stream);\n\n"
         "        let mut bare = required_arguments();\n"
         "        bare.push(\"--stream\");\n"
         "        assert!(Args::try_parse_from(bare).is_err());\n"
         "    }\n\n"
         "    #[test]\n    fn retired_split_format_and_family_flags_are_rejected() {",
         "streaming_is_on_by_default_and_can_be_turned_off")
    edit(client,
         "    record_timeline: bool,\n    /// Where each request's routed experts are written.",
         "    record_timeline: bool,\n"
         "    /// `--stream`: whether requests ask for a streamed response. Capturing\n"
         "    /// routes overrides it to non-streaming.\n"
         "    stream: bool,\n"
         "    /// Where each request's routed experts are written.",
         "    stream: bool,\n    /// Where each request")
    edit(client,
         "            record_timeline: args.timeline,\n            routed_experts_dir,\n",
         "            record_timeline: args.timeline,\n            stream: args.stream,\n            routed_experts_dir,\n",
         "stream: args.stream,")
    edit(client,
         "        let capturing_routes = self.routed_experts_dir.is_some();\n",
         "        let capturing_routes = self.routed_experts_dir.is_some();\n"
         "        // `--stream false` and a route capture both mean one completed body per request.\n"
         "        let streaming = self.stream && !capturing_routes;\n",
         "let streaming = self.stream && !capturing_routes;")
    edit(client,
         "            stream: !capturing_routes,\n",
         "            stream: streaming,\n",
         "            stream: streaming,\n")
    edit(client,
         "        if let Some(directory) = &self.routed_experts_dir {\n"
         "            self.fold_body(&request_id, &payload, send_instant, &mut fold, directory)\n"
         "                .await;\n"
         "        } else {\n"
         "            self.fold_response(&request_id, &payload, send_instant, &mut fold)\n"
         "                .await;\n"
         "        }\n",
         "        if streaming {\n"
         "            self.fold_response(&request_id, &payload, send_instant, &mut fold)\n"
         "                .await;\n"
         "        } else {\n"
         "            let directory = self.routed_experts_dir.as_deref();\n"
         "            self.fold_body(&request_id, &payload, send_instant, &mut fold, directory)\n"
         "                .await;\n"
         "        }\n",
         "let directory = self.routed_experts_dir.as_deref();")
    edit(client,
         "    /// Drive one request as a single non-streaming response and persist the\n    /// routed experts it carries.\n",
         "    /// Drive one request as a single non-streaming response and, when a\n    /// directory is given, persist the routed experts it carries.\n",
         "when a\n    /// directory is given")
    edit(client,
         "        fold: &mut StreamAccumulator,\n        directory: &std::path::Path,\n    ) {\n",
         "        fold: &mut StreamAccumulator,\n        directory: Option<&std::path::Path>,\n    ) {\n",
         "directory: Option<&std::path::Path>,")
    edit(client,
         "        let _ = fold.absorb(self.backend.parse_event(&value), elapsed_ms(send_instant));\n\n        let Some(encoded) = self.backend.routed_experts(&value) else {",
         "        let _ = fold.absorb(self.backend.parse_event(&value), elapsed_ms(send_instant));\n\n"
         "        // `--stream false` without a capture: the body was the whole answer and\n"
         "        // there is nothing to persist.\n"
         "        let Some(directory) = directory else {\n            return;\n        };\n\n"
         "        let Some(encoded) = self.backend.routed_experts(&value) else {",
         "let Some(directory) = directory else {")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
