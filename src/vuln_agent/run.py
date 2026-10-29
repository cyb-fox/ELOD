"""CLI entry point (the only scan entry point): runs the vulnerability review
graph over a C/C++ file or a whole project and emits structured JSON.

A single entry point whose granularity is derived from the target, so no mode
needs to be chosen by hand:
  · target is a file  -> file-level scan (single-file graph)
  · target is a directory -> project-level scan (per-file re-runs + a shared
    project index + a cross-file final review)
  --mode can still override the automatic choice.

Usage:
    python -m vuln_agent.run path/to/file.c          # auto -> file level
    python -m vuln_agent.run path/to/project/         # auto -> project level
    python -m vuln_agent.run path/to/project/ --profile scan   # deployment profile, lower cost
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from . import config
from .graph import build_graph
from .llm import enable_trace, get_trace
from .state import Budget, VulnState
from .tools import trajectory as tj


def _initial_state(target: str, mode: str, project_index: dict | None = None,
                   profile: str | None = None) -> VulnState:
    return {
        "mode": mode,                    # type: ignore[typeddict-item]
        "profile": profile or config.PROFILE,
        "target_path": target,
        "project_files": None,
        "project_index": project_index,
        "budget": Budget(
            tokens_used=0, tokens_max=config.BUDGET_TOKENS_MAX,
            llm_calls=0, calls_max=config.BUDGET_CALLS_MAX,
            failed_calls=0, json_failures=0,
        ),
        "partial": False,
    }


# Shared by both modes of the single graph: a module-level cache, so a project
# scan does not recompile the graph for every file.
_APP = None


def _get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_one_state(target: str, mode: str, project_index: dict | None,
                  profile: str | None = None, progress: bool = False) -> dict:
    """Run the graph for a single file (project_index can be injected for
    cross-file nodes / grep). Returns the report dict.

    progress=True streams per-node progress to stderr via app.stream (live
    output); otherwise a single invoke.
    """
    app = _get_app()
    init = _initial_state(target, mode, project_index, profile)
    cfg = {"recursion_limit": 100}  # headroom (a single chain's own step count
                                   # is fixed and far below this)
    if not progress:
        state = app.invoke(init, cfg)
        return state.get("report", {})
    return _stream_run(app, init, cfg, target)


# A one-line summary printed after each node finishes: key counts taken from the
# state delta that node returned.
def _node_brief(node: str, delta: dict) -> str:
    if not isinstance(delta, dict):
        return ""
    if node == "ingest":
        n = len(delta.get("functions", []) or [])
        if delta.get("_prescreen_clean"):
            return f"{n} functions - prescreen clean (short-circuit, LLM chain skipped)"
        return f"{n} functions"
    if node == "select_knowledge":
        # knowledge is empty by design under the clf backend, so the family
        # count is the informative number.
        fams = delta.get("family_candidates", []) or []
        loaded = len(delta.get("knowledge", {}) or {})
        return f"{len(fams)} families" + (f" - {loaded} CWEs" if loaded else "")
    if node == "secure_diff":
        return f"{len(delta.get('sinks', []) or [])} candidate points from the secure diff"
    if node == "risk_forensics":
        f = delta.get("forensics", []) or []
        kept = sum(1 for x in f if x.get("screen") != "exclude")
        return f"forensics on {len(f)} points ({kept} kept)"
    if node == "arbiter":
        v = delta.get("verdict", []) or []
        vuln = sum(1 for x in v if x.get("is_vuln"))
        return f"ruled on {len(v)} points ({vuln} vulnerable)"
    if node == "project_dataflow":
        return f"cross-file final review of {len(delta.get('verdict', []) or [])} points"
    if node == "output":
        s = (delta.get("report") or {}).get("summary", {})
        return f"{s.get('vulnerabilities_found', 0)} confirmed vulnerabilities"
    return ""


def _stream_run(app, init: VulnState, cfg: dict, target: str) -> dict:
    """Run the graph in streaming mode: print progress to stderr as each node
    finishes, and return the final report."""
    import time as _t
    name = Path(target).name
    print(f"▶ scanning {name}", file=sys.stderr, flush=True)
    report: dict = {}
    t0 = _t.time()
    step = 0
    for chunk in app.stream(init, cfg, stream_mode="updates"):
        # a chunk looks like {node_name: state_delta}
        for node, delta in chunk.items():
            step += 1
            brief = _node_brief(node, delta)
            dt = _t.time() - t0
            line = f"  [{step:>2}] {node:<18} {dt:6.1f}s"
            if brief:
                line += f"  {brief}"
            print(line, file=sys.stderr, flush=True)
            if node == "output" and isinstance(delta, dict):
                report = delta.get("report", {}) or report
    print(f"✔ done {name}  in {_t.time() - t0:.1f}s", file=sys.stderr, flush=True)
    return report


def _resolve_mode(target: str, explicit: str | None) -> str:
    """The single entry point derives the scan granularity from the target; an
    explicit --mode takes priority.
    Directory -> project level; file -> file level."""
    if explicit:
        return explicit
    return "project" if Path(target).is_dir() else "file"


def run(target: str, mode: str | None = None, profile: str | None = None,
        progress: bool = False, workers: int = 1,
        checkpoint_path: str | None = None, resume: bool = False) -> dict:
    mode = _resolve_mode(target, mode)
    if mode == "project":
        from .project import run_project
        return run_project(target, workers=workers, profile=profile, progress=progress,
                           checkpoint_path=checkpoint_path, resume=resume)
    if resume:
        raise ValueError("--resume is only supported for project directory scans")
    return run_one_state(target, mode, None, profile, progress=progress)


def _finding_mark(f: dict) -> str:
    """Human-readable mark for a finding: the three values must stay
    distinguishable, and an undetermined verdict must never collapse into safe.

    vuln -> 🔴 VULN / underdetermined -> 🟡 UNDERDETERMINED / safe_proved -> 🟢 SAFE.
    underdetermined also has is_vuln=False, so reading only is_vuln would print it
    as "safe" -- for a detection tool that is precisely the worst kind of
    misreading, so the mark is always taken from the three-valued resolution.
    Records with no such field fall back to is_vuln (same view as
    nodes/output.py::_resolution_of).
    """
    res = str(f.get("resolution") or "")
    if not res:
        res = "vuln" if f.get("is_vuln") else "safe_proved"
    return {
        "vuln": "🔴 VULN",
        "underdetermined": "🟡 UNDERDETERMINED",
    }.get(res, "🟢 SAFE")


def _print_human(report: dict) -> None:
    if report.get("mode") == "project":
        _print_project(report)
        return
    s = report.get("summary", {})
    b = report.get("budget", {})
    print(f"\n=== Vulnerability review report: {report.get('target')} ===")
    if report.get("partial"):
        print(f"⚠️  {report.get('note', 'partial')}")
    print(
        f"sinks examined: {s.get('sinks_examined')} | screened out: {s.get('screened_out')} | "
        f"conclusions: {s.get('verdicts')} | vulnerabilities found: {s.get('vulnerabilities_found')}"
    )
    print(f"families considered: {s.get('families_considered')}")
    print(
        f"budget: tokens {b.get('tokens_used')}/{b.get('tokens_max')}, "
        f"calls {b.get('llm_calls')}/{b.get('calls_max')}"
    )
    # Degradation trace: non-empty means this report is a degraded product (some
    # node got no usable LLM output), which is the most common explanation for
    # "the same command gave different results twice".
    if b.get("failed_calls") or b.get("json_failures"):
        print(f"degradation counts: failed calls {b.get('failed_calls', 0)} · "
              f"unparseable JSON {b.get('json_failures', 0)}")
    for d in report.get("degraded", []) or []:
        print(f"  ⚠ {d}")
    print()
    for f in report.get("findings", []):
        mark = _finding_mark(f)
        loc = f.get("location", {})
        print(f"{mark}  {f.get('sink_id')}  [{f.get('cwe_id')}]  "
              f"line {loc.get('line_start')}  conf={f.get('confidence')}")
        for bx in f.get("basis", []):
            print(f"      · {bx}")
        if f.get("is_vuln") and f.get("fix_suggestion"):
            # The secure side is the model's "expected safe form", not something
            # measured to exist in the source -- the wording must separate
            # expected from actual, or a reader will think the code is already in
            # the file.
            print(f"      suggested safe form (expected, not measured):"
                  f" {f.get('fix_suggestion')}")
    if report.get("errors"):
        print("\nerrors:")
        for e in report["errors"]:
            print(f"  - {e}")


def _print_project(report: dict) -> None:
    s = report.get("summary", {})
    b = report.get("budget", {})
    print(f"\n=== Project vulnerability review report: {report.get('target')} ===")
    print(
        f"files scanned: {s.get('files_scanned')} | errors: {s.get('files_error')} | "
        f"partial: {s.get('files_partial')} | global vulnerabilities: {s.get('vulnerabilities_found')}"
    )
    print(f"budget: tokens {b.get('tokens_used')}, calls {b.get('llm_calls')}")
    if b.get("failed_calls") or b.get("json_failures"):
        print(f"degradation counts: failed calls {b.get('failed_calls', 0)} · "
              f"unparseable JSON {b.get('json_failures', 0)}")
    print()
    for v in report.get("vulnerabilities", []):
        loc = v.get("location", {})
        print(f"🔴 VULN  {v.get('file')}  {v.get('sink_id')}  [{v.get('cwe_id')}]  "
              f"line {loc.get('line_start')}  conf={v.get('confidence')}")
        for bx in v.get("basis", []):
            print(f"      · {bx}")
    # Errored files listed separately
    errs = [f for f in report.get("files", []) if f.get("error")]
    if errs:
        print("\nfile-level errors:")
        for f in errs:
            print(f"  - {f['file']}: {f['error']}")


def render_trace(trace: list[dict]) -> str:
    """Render complete LLM I/O and per-node totals for a terminal or log file."""
    out = io.StringIO()

    def emit(text: str = "") -> None:
        print(text, file=out)

    if not trace:
        emit("\n(no trace: not enabled, or no LLM calls)")
        return out.getvalue()
    emit(f"\n=== LLM call detail ({len(trace)} calls total) ===")
    for i, r in enumerate(trace, 1):
        status = "OK" if r.get("ok") else "FAIL"
        head = (
            f"[{i:>3}] {r.get('tag', '?'):<16} {status}  "
            f"{r.get('elapsed_s', 0)}s  "
            f"tok={r.get('total_tokens', 0)}"
            f"(in {r.get('prompt_tokens', 0)}/out {r.get('completion_tokens', 0)})"
        )
        if r.get("attempt"):
            head += f"  retries={r['attempt']}"
        emit(head)
        if not r.get("ok"):
            emit(f"       error: {r.get('error')}")
        if "system" in r:  # full mode
            emit(f"       --- system ---\n{_indent(r['system'])}")
            emit(f"       --- user ---\n{_indent(r['user'])}")
            emit(f"       --- response ---\n{_indent(r.get('response', ''))}")

    # Aggregate by node
    agg: dict[str, dict] = {}
    for r in trace:
        a = agg.setdefault(r.get("tag", "?"), {"calls": 0, "tok": 0, "s": 0.0, "fail": 0})
        a["calls"] += 1
        a["tok"] += r.get("total_tokens", 0)
        a["s"] += r.get("elapsed_s", 0) or 0
        if not r.get("ok"):
            a["fail"] += 1
    emit("\n=== token/time summary by node ===")
    tot_tok = tot_s = tot_calls = 0
    for tag, a in sorted(agg.items(), key=lambda kv: -kv[1]["tok"]):
        emit(f"  {tag:<18} calls={a['calls']:<3} tokens={a['tok']:<8} "
             f"time={a['s']:.1f}s" + (f"  failed={a['fail']}" if a["fail"] else ""))
        tot_tok += a["tok"]; tot_s += a["s"]; tot_calls += a["calls"]
    emit(f"  {'total':<18} calls={tot_calls:<3} tokens={tot_tok:<8} time={tot_s:.1f}s")
    return out.getvalue()


def _print_trace(trace: list[dict]) -> None:
    """Print each LLM call's I/O and per-node totals to the terminal."""
    print(render_trace(trace), end="")


def _write_full_log(path: str, trace: list[dict]) -> None:
    """Atomically persist the same complete trace text shown by ``--trace``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(render_trace(trace), encoding="utf-8")
    temporary.replace(output)


def _indent(text: str, prefix: str = "         ") -> str:
    return "\n".join(prefix + ln for ln in (text or "").split("\n"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="vuln_agent: C/C++ vulnerability review (LangGraph) -"
                    " the single scan entry point")
    ap.add_argument("target", help="C/C++ file to review, or a project root"
                                " directory (granularity is derived automatically)")
    ap.add_argument("--mode", choices=["file", "project"], default=None,
                    help="omitted = derive from target (file -> file,"
                         " directory -> project); given = force override")
    ap.add_argument("--profile", choices=["bench", "scan"], default=None,
                    help="bench = the full evaluation chain (default, protects"
                         " recall); scan = the deployment profile, which enables"
                         " the cheap file-level prescreen short-circuit to save cost")
    ap.add_argument("--workers", type=int, default=1,
                    help="file-level concurrency in project mode (<=4"
                         " recommended; ignored in file mode)")
    ap.add_argument("--progress", action="store_true",
                    help="live output: print progress to stderr as each node and"
                         " each file finishes")
    ap.add_argument("--json", dest="json_out",
                    help="write the full report to this JSON file")
    ap.add_argument("--resume", action="store_true",
                    help="project mode only: resume from the complete report given"
                         " by --json, skipping succeeded files and re-running"
                         " failed ones")
    ap.add_argument(
        "--trace", action="store_true",
        help="record and print each LLM call's full input/output plus token"
             " statistics",
    )
    ap.add_argument(
        "--trace-summary", action="store_true",
        help="record/print only the token and timing summary, without the full"
             " prompt/response text",
    )
    ap.add_argument("--trace-out",
                    help="write the trace detail to this JSON file")
    ap.add_argument("--full-log",
                    help="write the same complete LLM terminal trace as --trace"
                         " to this text file")
    ap.add_argument("--trajectory-out",
                    help="write the structured proof trajectory (used by the S3"
                         " side-channel audit / EGPR statistics) to this JSON file")
    ap.add_argument(
        "--llm-cache", metavar="DIR", default=None,
        help="cache and replay LLM responses in DIR, keyed by a hash of"
             " (model, prompt, sampling parameters): re-running the same target"
             " is byte-identical (the only means of guaranteeing reproducibility;"
             " changing a prompt or the model shifts the key automatically)",
    )
    args = ap.parse_args(argv)

    if args.resume and not args.json_out:
        ap.error("--resume requires --json")

    if args.llm_cache:
        # Write config rather than an environment variable: BudgetedLLM reads
        # config.LLM_CACHE_DIR fresh on every call.
        config.LLM_CACHE_DIR = args.llm_cache

    want_trace = args.trace or args.trace_summary or args.trace_out or args.full_log
    if want_trace:
        # --trace / --trace-out / --full-log record full text; only
        # --trace-summary records tokens and timing.
        full = bool(args.trace or args.trace_out or args.full_log)
        enable_trace(full=full)
    if args.trajectory_out:
        tj.enable_trajectory()

    report = run(args.target, args.mode, args.profile, progress=args.progress,
                 workers=args.workers, checkpoint_path=args.json_out,
                 resume=args.resume)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"report written to {args.json_out}")
    _print_human(report)

    if want_trace:
        trace = get_trace()
        if args.trace_out:
            with open(args.trace_out, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)
            print(f"\ntrace written to {args.trace_out} ({len(trace)} calls)")
        if args.full_log:
            _write_full_log(args.full_log, trace)
            print(f"complete LLM terminal trace written to {args.full_log}")
        # Terminal printing: --trace prints full text; --trace-summary prints
        # only the summary.
        if args.trace:
            _print_trace(trace)
        else:
            _print_trace([{k: v for k, v in r.items()
                           if k not in ("system", "user", "response")}
                          for r in trace])
    if args.trajectory_out:
        steps = tj.get_trajectory()
        with open(args.trajectory_out, "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        print(f"\nproof trajectory written to {args.trajectory_out}"
              f" ({len(steps)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
