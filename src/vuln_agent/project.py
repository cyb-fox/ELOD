"""Project-level index + runner (the cross-file branch).

Single-file mode is the MVP; the cross-project branch is hooked in here: the
single-file graph structure is unchanged, and it is realised through "a shared
project index + re-running the single-file graph per file", so file-mode
behaviour changes not at all.

- build_project_index(root): walks out all C/C++ source files, reads each in
  full, and builds a lightweight structured symbol index with tree-sitter
  (definitions/declarations/calls/includes/spans). It keeps the
  'function name -> [files...]' field for existing consumers. Read once, then
  injected into each file's initial state, feeding context_search's cross-file
  grep and project_dataflow's producer/caller lookup.
- run_project(root): re-runs the existing single-file graph per file (with
  project_index injected) and aggregates into a project-level report (grouped by
  file + a global vulnerability list). A file-level failure does not stop the
  whole project.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import config
from .tools.symbol_index import build_symbol_index

# Source-file suffixes aligned with segmenter._EXT_LANG (used when walking).
_SRC_EXTS = (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx")
# Directory names skipped while walking (build artifacts / dependencies / version
# control, so unrelated code never enters the index).
_SKIP_DIRS = {".git", ".svn", "build", "cmake-build-debug", "node_modules",
              "third_party", "vendor", ".cache", "__pycache__"}


def walk_source_files(root: str) -> list[str]:
    """Recursively walk the project root and return the absolute paths of all
    C/C++ source files (order-preserving, build directories excluded)."""
    base = Path(root)
    out: list[str] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _SRC_EXTS:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(str(p.resolve()))
    return out


def build_project_index(root: str) -> dict:
    """Build the project index:
    - files:   [absolute paths...] (order-preserving)
    - sources: {abs_path -> full text}
    - symbols: {func_name -> [abs_path...]}  compatibility field
    - symbol_index: structured definitions/declarations/calls/includes/spans index
    - root:    absolute path of the project root
    """
    files = walk_source_files(root)
    sources: dict[str, str] = {}
    for path in files:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        sources[path] = text
    symbol_index = build_symbol_index(sources, str(Path(root).resolve()))
    symbols = {
        name: sorted({item["path"] for item in spans})
        for name, spans in symbol_index.get("definitions", {}).items()
    }
    return {
        "root": str(Path(root).resolve()),
        "files": files,
        "sources": sources,
        "symbols": symbols,
        "symbol_index": symbol_index,
    }


def _rel(index: dict, path: str) -> str:
    """Turn an absolute path into a short path relative to the project root, for
    report readability."""
    try:
        return str(Path(path).resolve().relative_to(index["root"]))
    except Exception:
        return path


def _record_from_checkpoint(row: dict, path: str) -> dict:
    """Restore a persisted project-report file entry into run_project's internal
    record.

    The checkpoint deliberately reuses successful entries only; failed entries
    are requested again on the next run, so that a single network error is never
    frozen in as "already scanned". Budget/degradation fields absent from the
    report are tolerated as empty values.
    """
    return {
        "file": row["file"],
        "path": path,
        "error": None,
        "elapsed_s": row.get("elapsed_s"),
        "report": {
            "partial": bool(row.get("partial", False)),
            "prescreened_clean": bool(row.get("prescreened_clean", False)),
            "summary": {
                "vulnerabilities_found": row.get("vulnerabilities_found", 0),
                "sinks_examined": row.get("sinks_examined"),
            },
            "budget": row.get("budget") or {},
            "degraded": row.get("degraded") or [],
            "findings": row.get("findings") or [],
        },
    }


def _load_project_checkpoint(path: Path, root: str, index: dict) -> dict[str, dict]:
    """Read and validate the project checkpoint, returning
    ``relative_file -> internal_record``."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read project checkpoint {path}: {e}") from e
    if saved.get("mode") != "project":
        raise ValueError(f"{path} is not a project scan report and cannot be"
                         f" used for --resume")
    expected_root = str(Path(root).resolve())
    if saved.get("target") != expected_root:
        raise ValueError(
            f"checkpoint target mismatch: {saved.get('target')!r} !="
            f" {expected_root!r}")

    by_rel = {_rel(index, p): p for p in index["files"]}
    restored: dict[str, dict] = {}
    for row in saved.get("files", []):
        rel = row.get("file")
        if not isinstance(rel, str) or row.get("error") or rel not in by_rel:
            continue
        restored[rel] = _record_from_checkpoint(row, by_rel[rel])
    return restored


def _write_project_checkpoint(path: Path, root: str, index: dict,
                              per_file: list[dict]) -> None:
    """Write the project progress atomically, so the JSON stays usable as input
    for the next --resume even after Ctrl-C."""
    report = _aggregate_report(root, index, per_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def run_project(root: str, workers: int = 1,
                targets: Optional[set[str]] = None,
                profile: Optional[str] = None,
                progress: bool = False,
                checkpoint_path: Optional[str] = None,
                resume: bool = False) -> dict:
    """Re-run the single-file graph per file (sharing the project index) and
    aggregate into a project-level report.

    workers>1 enables file-level concurrency (bounded by the gateway's TPM rate
    limit; ≤4 recommended).

    targets: run the LLM pipeline only for these files (paths relative to the
    project root); all other files are still read into project_index in full (for
    cross-file grep / project_dataflow producers) but do not run the pipeline on
    their own. None = run every file. Files that a slice or pair set depends on
    are not scored, and this parameter compresses LLM cost from O(all files) to
    O(judged files) -- the largest speedup / cost lever.

    profile: the 'scan' profile runs the cheap prescreen on every file (no danger
    indication short-circuits the LLM chain); 'bench'/None runs the full
    evaluation chain. Pass 'scan' for real project scans to save cost.

    progress: live output -- prints one line per finished file to stderr
    (vulnerability count / elapsed / progress).

    checkpoint_path: after each file completes, the complete project JSON report
    is written atomically. When resume=True, files already completed
    successfully are skipped from that report and failed files are re-run.
    """
    import sys
    from .run import run_one_state  # deferred import, keeps graph construction
                                     # cost out of paths like --score-only

    index = build_project_index(root)
    files = index["files"]
    # Judged-file set: targets are relative paths, matched against the absolute
    # paths in this run's index.
    judged: Optional[set[str]] = None
    if targets is not None:
        judged = {p for p in files if _rel(index, p) in targets}
        skipped_ctx = len(files) - len(judged)
        print(f"[run_project] {Path(root).name}: judged files {len(judged)}/"
              f"{len(files)} ({skipped_ctx} used only as cross-file context,"
              f" no pipeline)")
    # Core-module filter for the scan profile: with no explicit targets, files
    # that are non-core (test/testdata/example/generated code) are excluded from
    # the judged set (they all remain in project_index for context_search
    # cross-file tracing). "Scan the core first, then search related variables
    # if something turns up." The bench profile and explicit targets do not
    # trigger this (the view is unchanged).
    elif config.prescreen_enabled(profile) and config.SCAN_SKIP_NONCORE:
        judged = {p for p in files if not config.is_noncore_path(_rel(index, p))}
        skipped_ctx = len(files) - len(judged)
        if skipped_ctx:
            print(f"[run_project] {Path(root).name}: core-module filter ->"
                  f" judged files {len(judged)}/{len(files)}"
                  f" ({skipped_ctx} non-core files used only as context)")

    run_list = [p for p in files if judged is None or p in judged]
    n_total = len(run_list)
    per_file: list[dict] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if resume:
        if checkpoint is None:
            raise ValueError("resume=True requires checkpoint_path")
        cached = _load_project_checkpoint(checkpoint, root, index)
        for path in run_list:
            rel = _rel(index, path)
            if rel in cached:
                per_file.append(cached[rel])
        run_list = [p for p in run_list if _rel(index, p) not in cached]
        print(f"[run_project] resumed {len(cached)} completed files,"
              f" {len(run_list)} still to process")
    if progress:
        print(f"▶ project scan {Path(root).name}: {n_total} judged files"
              + (f" (+{len(files) - n_total} context)"
                 if len(files) != n_total else ""),
              file=sys.stderr, flush=True)
    _done = [len(per_file)]  # closure counter (a coarse hint when concurrent)

    def _one(path: str) -> dict:
        t0 = time.time()
        rec = {"file": _rel(index, path), "path": path, "error": None}
        # Non-judged files: context only, pipeline skipped (saves the entire LLM
        # chain).
        if judged is not None and path not in judged:
            rec["skipped_context_only"] = True
            return rec
        try:
            report = run_one_state(path, "project", index, profile)
            rec["report"] = report
        except Exception as e:  # a single file's failure does not stop the project
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_s"] = round(time.time() - t0, 1)
        if progress:
            _done[0] += 1
            rep = rec.get("report") or {}
            vn = (rep.get("summary") or {}).get("vulnerabilities_found", 0)
            clean = " (prescreen clean)" if rep.get("prescreened_clean") else ""
            if rec.get("error"):
                mark, tail = "✗", rec["error"]
            else:
                # Three values: vuln -> 🔴 / underdetermined -> 🟡 / safe_proved -> 🟢.
                # A plain binary on vulnerabilities_found would print an
                # undetermined file as 🟢 (looking "clean") -- and undetermined
                # is precisely the state that must never be read as safe, so the
                # progress line takes its mark from resolution too.
                res = str(rep.get("file_resolution") or "")
                if not res:  # reports with no such field fall back to vuln-or-not
                    res = "vuln" if vn else "safe_proved"
                if res == "vuln":
                    mark, tail = "🔴", f"vulnerable {vn}{clean}"
                elif res == "underdetermined":
                    ud = (rep.get("summary") or {}).get("underdetermined_points", 0)
                    mark, tail = "🟡", f"undetermined {ud}{clean}"
                else:
                    mark, tail = "🟢", f"proved safe{clean}"
            print(f"  {mark} [{_done[0]:>3}/{n_total}] {rec['file']:<40} "
                  f"{rec['elapsed_s']:5.1f}s  {tail}", file=sys.stderr, flush=True)
        return rec

    def _accept(rec: dict) -> None:
        per_file.append(rec)
        if checkpoint is not None:
            _write_project_checkpoint(checkpoint, root, index, per_file)

    if workers > 1 and len(run_list) > 1:
        n = max(1, min(workers, len(run_list)))
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = {ex.submit(_one, p): p for p in run_list}
            for fut in as_completed(futs):
                _accept(fut.result())
    else:
        for p in run_list:
            _accept(_one(p))

    # Preserve order (by the files order), and aggregate the global vulnerability
    # list.
    order = {p: i for i, p in enumerate(files)}
    per_file.sort(key=lambda r: order.get(r["path"], 1 << 30))
    report = _aggregate_report(root, index, per_file)
    if checkpoint is not None:
        _write_project_checkpoint(checkpoint, root, index, per_file)
    return report


def _aggregate_report(root: str, index: dict, per_file: list[dict]) -> dict:
    """Project-level aggregation: grouping by file + the global confirmed
    vulnerability list + budget totals."""
    global_vulns: list[dict] = []
    files_out: list[dict] = []
    tok = calls = 0
    n_partial = 0
    n_failed = n_badjson = 0   # project-wide degradation counts (call failures /
                               # unparseable output)
    for rec in per_file:
        if rec.get("skipped_context_only"):
            continue  # context-only files stay out of the report (not scored,
                      # not in budget totals)
        rep = rec.get("report") or {}
        if rec.get("error"):
            files_out.append({
                "file": rec["file"], "error": rec["error"],
                "elapsed_s": rec.get("elapsed_s"),
            })
            continue
        summary = rep.get("summary", {})
        budget = rep.get("budget", {})
        tok += int(budget.get("tokens_used") or 0)
        calls += int(budget.get("llm_calls") or 0)
        n_failed += int(budget.get("failed_calls") or 0)
        n_badjson += int(budget.get("json_failures") or 0)
        if rep.get("partial"):
            n_partial += 1
        # This file's confirmed vulnerabilities (output has passed the threshold
        # gate and suspected entries are dropped).
        vulns = [f for f in rep.get("findings", [])
                 if f.get("is_vuln") and not f.get("suspected")]
        for v in vulns:
            global_vulns.append({**v, "file": rec["file"]})
        files_out.append({
            "file": rec["file"],
            "partial": bool(rep.get("partial", False)),
            "prescreened_clean": bool(rep.get("prescreened_clean", False)),
            # The key in the single-file report is file_resolution; reading
            # rep.get("resolution") here would always yield None and silently
            # drop the third state from every project-level prediction, which
            # downstream readers would then see as "no conclusion".
            "resolution": rep.get("file_resolution"),
            "underdetermined_risks": rep.get("underdetermined_risks", []),
            "unresolved_note": rep.get("unresolved_note"),
            "vulnerabilities_found": len(vulns),
            "sinks_examined": summary.get("sinks_examined"),
            "budget": budget,
            "degraded": rep.get("degraded", []),
            "findings": rep.get("findings", []),
            "elapsed_s": rec.get("elapsed_s"),
        })

    return {
        "mode": "project",
        "target": str(Path(root).resolve()),
        "summary": {
            "files_scanned": len([r for r in per_file if not r.get("error")]),
            "files_error": len([r for r in per_file if r.get("error")]),
            "files_partial": n_partial,
            "vulnerabilities_found": len(global_vulns),
        },
        "budget": {"tokens_used": tok, "llm_calls": calls,
                   "failed_calls": n_failed, "json_failures": n_badjson},
        "vulnerabilities": global_vulns,
        "files": files_out,
    }
