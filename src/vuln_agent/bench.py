"""PrimeVul batch evaluation: run vulnerability review file by file over the
dataset and compute accuracy/recall/F1 plus pair correctness.

This module only calls the existing run() (it does not change any agent logic).
CWE "relaxed matching" lives on the evaluation side -- merging by the top-level
family in families.json (e.g. CWE-457/909/665 all belong to
input-validation-and-state-integrity and count as hits for one another); when the
GT CWE has no matching family it falls back to exact matching.

Three separable stages:
  1) predict: run the LLM and write each file's prediction together with the GT
     to predictions.jsonl (supports resuming).
  2) score  : recompute metrics from predictions.jsonl without re-running the LLM
     (--score-only).
  3) freeze : freeze the manifest selected this run and the final predictions into
     a run artifact (--run-dir), later recomputed with --rescore-run, independent
     of the mutable resume journal.

────────────────────────── Usage ──────────────────────────
# smoke test: sample 20 pairs (40 files) to validate the chain, file-level
# concurrency 3
python3 -m vuln_agent.bench --sample 20 --workers 3 --out runs/pred.jsonl

# only run the first N manifest rows (unpaired, quick try)
python3 -m vuln_agent.bench --limit 10 --out runs/pred.jsonl

# full run (hours; resuming: repeating the command skips already-succeeded files)
python3 -m vuln_agent.bench --workers 4 --out runs/pred.jsonl

# retry errored files (--retry-errors re-runs files whose error != null)
python3 -m vuln_agent.bench --workers 4 --out runs/pred.jsonl --retry-errors

# recompute metrics only (no LLM), and export a per-file detail CSV
python3 -m vuln_agent.bench --score-only --out runs/pred.jsonl --per-file-out runs/detail.csv

# specify manifest / data root (default data/primevul/manifest.jsonl)
python3 -m vuln_agent.bench --manifest data/primevul/manifest.jsonl --out runs/pred.jsonl

──────────────── Full PrimeVul run (870 files = 435 pairs) ────────────────
# 1) smoke test 30 pairs first to validate the chain / rate limits / paths
python3 -m vuln_agent.bench --manifest data/primevul/manifest.jsonl \
    --sample 30 --workers 5 --out runs/pred_pv_smoke.jsonl

# 2) full run (hours; resuming: re-running the same command after an interruption
#    continues from where it stopped). Keep --workers within the gateway's TPM
#    rate limit; frequent 429 (RateLimit backoff) means lowering it.
python3 -m vuln_agent.bench --manifest data/primevul/manifest.jsonl \
    --workers 5 --out runs/pred_primevul.jsonl

# 3) after the run, recompute metrics only (no LLM re-run) + export per-file detail
python3 -m vuln_agent.bench --score-only \
    --out runs/pred_primevul.jsonl --per-file-out runs/detail_pv.csv
──────────────────────────────────────────────────────────────────────────────

Metrics are emitted under two views:
  · pure detection: a file counts as "predicted vulnerable" if any is_vuln is
    reported.
  · CWE family-level (cwe-family): used only for the lenient analysis of
    vulnerable-side CWE attribution; the file-level confusion matrix is kept for
    compatibility. P-C/P-V/P-B/P-R are always computed on detection: any
    confirmed vulnerability on the patched side is an FP.
Each view reports accuracy/precision/recall/F1/FPR; pair metrics are reported only
under the M1 detection view.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from . import config


ARTIFACT_SCHEMA_VERSION = 1
SCORER_VERSION = 1

# ===================== Family mapping (CWE relaxed matching) =====================


def _load_family_map() -> dict[str, str]:
    """cwe_id -> top-level family name. Reads families.json; returns empty when
    absent (the evaluation falls back to exact matching)."""
    p = config.KB_DIR / "families.json"
    try:
        fams = json.loads(p.read_text(encoding="utf-8")).get("families", {})
    except Exception:
        return {}
    m: dict[str, str] = {}
    for fam, info in (fams or {}).items():
        for cwe in (info or {}).get("members", []) if isinstance(info, dict) else []:
            m[cwe] = fam
    return m


_FAMILY_MAP = _load_family_map()


def _norm_cwe(raw: Any) -> str:
    """Normalize a GT/predicted CWE to the 'CWE-123' form; returns '' when
    unrecognized."""
    if not raw:
        return ""
    s = str(raw).strip().upper()
    if not s:
        return ""
    if s.startswith("CWE-"):
        return s
    if s.startswith("CWE"):
        s = s[3:].lstrip("-")
    return f"CWE-{s}" if s.isdigit() else ""


def _cwe_key(cwe: str) -> str:
    """Relaxed-matching key: the family name when one exists, otherwise the CWE
    number itself (equivalent to exact matching)."""
    return _FAMILY_MAP.get(cwe, cwe)


def _cwe_family_hit(gt_cwe: str, pred_cwes: list[str]) -> bool:
    """Whether any predicted CWE shares the GT's top-level family (requiring an
    exact number match when the GT has no family)."""
    gt = _norm_cwe(gt_cwe)
    if not gt:
        return bool(pred_cwes)  # GT unlabelled: any reported CWE counts as a hit
    gk = _cwe_key(gt)
    return any(_cwe_key(_norm_cwe(pc)) == gk for pc in pred_cwes if _norm_cwe(pc))


# ===================== predict: run the LLM =====================


def _extract_pred(report: dict) -> dict:
    """Extract the fields the evaluation needs from the run() report.

    Only "confirmed vulnerabilities" count as a predicted positive -- the output
    threshold gate marks low-confidence findings with suspected=True, and such
    doubts are kept in the report for human review but do not carry the
    file-level positive (consistent with summary.vulnerabilities_found).

    Three-valued verdict: `file_resolution ∈ {vuln, safe_proved, underdetermined}`
    is persisted alongside. The binary metric view is **unchanged** -- `pred_vuln`
    is still determined solely by confirmed positives, and `underdetermined`
    counts as negative (otherwise recall/FPR could not be compared with the
    existing main table). `file_resolution` is a **side field** used to report
    "how many negatives are actually undetermined rather than proved safe" and to
    recompute the abstain view offline.

    Cost accounting: `report.budget`'s tokens/calls are persisted too, as a
    second source independent of the trace. The two are cross-checked by the
    summarize script, which errors out on any mismatch, so a run whose cost
    accounting silently dropped records cannot pass unnoticed."""
    findings = report.get("findings", []) or []
    vulns = [f for f in findings if f.get("is_vuln") and not f.get("suspected")]
    pred_cwes = sorted({str(f.get("cwe_id") or "") for f in vulns if f.get("cwe_id")})
    # Persist (cwe,conf) for every is_vuln finding, suspected included, so that
    # --sweep can recompute thresholds offline.
    all_vuln = [f for f in findings if f.get("is_vuln")]
    detail = [{"cwe": str(f.get("cwe_id") or ""),
               "conf": round(float(f.get("confidence") or 0.0), 3)}
              for f in all_vuln]
    summary = report.get("summary") or {}
    budget = report.get("budget") or {}
    resolution = str(report.get("file_resolution") or "")
    if not resolution:
        # Records predating the three-valued verdict. **A zero output must never
        # fall back to safe_proved**: an empty findings list in such a record can
        # mean either "genuinely found nothing" or "the chain ran without
        # inspecting anything", and the two are indistinguishable -> always
        # underdetermined (same view as output's empty-output gate, preventing a
        # silent miss from being recorded as proved safety).
        if vulns:
            resolution = "vuln"
        elif findings and int(summary.get("sinks_examined", 0) or 0):
            resolution = "safe_proved"
        else:
            resolution = "underdetermined"
    return {
        "pred_vuln": bool(vulns),
        "pred_cwes": pred_cwes,
        "vuln_findings": detail,
        "n_findings": len(vulns),
        "file_resolution": resolution,
        "n_underdetermined": int(summary.get("underdetermined_points", 0) or 0),
        "sinks_examined": int(summary.get("sinks_examined", 0) or 0),
        # Risk CWEs reported by underdetermined points (side field, **excluded**
        # from pred_cwes and from the binary view): lets an offline analysis ask
        # whether abstention at least named a risk category.
        "underdetermined_cwes": sorted({str(r.get("cwe_id") or "")
                                for r in (report.get("underdetermined_risks") or [])
                                if r.get("cwe_id")}),
        "partial": bool(report.get("partial", False)),
        # secure_diff comparison outcome (side field): the only legitimate basis
        # for recording an empty output as safe_proved, letting check_regression
        # R1 distinguish "comparison completed, no difference" from "chain ran
        # without inspecting".
        "secure_diff_outcome": str((summary or {}).get("secure_diff_outcome") or ""),
        # Second cost source (independent of the trace, for cross-checking):
        # true usage accumulated in place by BudgetedLLM.
        "budget_tokens": int(budget.get("tokens_used") or 0),
        "budget_calls": int(budget.get("llm_calls") or 0),
        "budget_failed_calls": int(budget.get("failed_calls") or 0),
        "budget_json_failures": int(budget.get("json_failures") or 0),
    }


def _network_blank_error(report: dict, rec: dict) -> Optional[str]:
    """Degraded blank-run detection: this run is a product of "the chain
    succeeded but inspected nothing" -> do not persist, leave it for a re-run.

    Two forms, both of which turn pred_vuln=False into a false negative while
    looking perfectly normal:
    · `budget.failed_calls>0` with zero is_vuln candidates (suspected included)
      -- network outage / timeouts exhausted on retry;
    · `budget.json_failures>0` with `sinks_examined==0` -- the call succeeded but
      the output was unparseable (mostly truncation during generation), so
      secure_diff yielded zero candidates and the rule floor was 0 as well,
      leaving not a single dangerous operation inspected.
    The caller should **not persist** such a record: persisting it as a success
    would make the resume logic treat error=None as completed and permanently
    freeze the false negative. Records that degraded but still produced output or
    still inspected sinks are persisted as usual."""
    b = report.get("budget") or {}
    failed = int(b.get("failed_calls", 0) or 0)
    if failed and not rec.get("vuln_findings"):
        return (f"NetworkDegraded: {failed} LLM calls failed with zero output"
                " (suspected blank run; not persisted, pending a re-run)")
    bad_json = int(b.get("json_failures", 0) or 0)
    if bad_json and not int(rec.get("sinks_examined", 0) or 0):
        return (f"OutputDegraded: {bad_json} LLM outputs unparseable and"
                " sinks_examined=0 (no dangerous operation inspected; not"
                " persisted, pending a re-run)")
    return None


def _load_manifest(manifest: Path, data_root: Path) -> list[dict]:
    """Read manifest.jsonl and add each file's absolute path (the file field is
    relative to data_root)."""
    rows = []
    for ln in manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        d["_path"] = str((data_root / d["file"]).resolve())
        rows.append(d)
    return rows


def _select_rows(rows: list[dict], args) -> list[dict]:
    """Select a subset according to --pairs / --sample / --limit."""
    if args.sample:
        # Sample K pairs at random, keeping pairs intact (otherwise P-C/P-V are
        # meaningless)
        by_pair: dict[str, list[dict]] = {}
        for r in rows:
            by_pair.setdefault(r.get("pair_id", r["file"]), []).append(r)
        pids = list(by_pair.keys())
        random.Random(args.seed).shuffle(pids)
        chosen = pids[: args.sample]
        return [r for pid in chosen for r in by_pair[pid]]
    if args.limit:
        return rows[: args.limit]
    return rows


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _artifact_rows(rows: list[dict]) -> list[dict]:
    """Drop runtime absolute paths, keeping the original manifest rows selected
    by this run."""
    return [{k: v for k, v in row.items() if k != "_path"} for row in rows]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_fingerprint() -> dict[str, str]:
    """Record the code version; stays usable outside a git checkout, and a
    failure here is not treated as an evaluation failure."""
    try:
        root = Path(__file__).resolve().parents[1]
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        working_dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        staged_dirty = subprocess.call(
            ["git", "diff", "--cached", "--quiet"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        return {"head": head, "dirty": str(working_dirty or staged_dirty).lower()}
    except (OSError, subprocess.SubprocessError):
        return {"head": "unknown", "dirty": "unknown"}


def _load_done(out_path: Path) -> dict[str, dict]:
    """Read an existing predictions.jsonl and return file -> record (for
    resuming)."""
    done: dict[str, dict] = {}
    if not out_path.exists():
        return done
    for ln in out_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
            done[rec["file"]] = rec
        except Exception:
            continue
    return done


def _trace_file(trace_dir: Path, file_name: str) -> Path:
    """Map a manifest file to a stable per-file trace path."""
    return trace_dir / f"{file_name}.trace.json"


def _write_trace(trace_dir: Path, file_name: str, trace: list[dict]) -> str:
    """Atomically write complete LLM I/O for one file and return its relative path."""
    path = _trace_file(trace_dir, file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return str(path.relative_to(trace_dir))


def _full_log_file(full_log_dir: Path, file_name: str) -> Path:
    """Map a manifest file to its isolated, human-readable LLM transcript."""
    return full_log_dir / f"{file_name}.full.log"


def _write_full_log(full_log_dir: Path, file_name: str, trace: list[dict]) -> str:
    """Atomically persist exactly the detailed trace format used by ``run --trace``."""
    from .run import render_trace

    path = _full_log_file(full_log_dir, file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_trace(trace), encoding="utf-8")
    os.replace(temporary, path)
    return str(path.relative_to(full_log_dir))


def _predict(rows: list[dict], args) -> None:
    from .run import run  # deferred import: keeps graph building out of
                          # --score-only

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path)

    todo = []
    for r in rows:
        prev = done.get(r["file"])
        if prev and prev.get("error") is None:
            continue  # already succeeded
        if prev and prev.get("error") and not args.retry_errors:
            continue  # errored last time and no retry requested
        todo.append(r)

    print(f"[predict] to run {len(todo)} / selected {len(rows)}"
          f" (already done {len(rows) - len(todo)})")
    if not todo:
        print("[predict] nothing to run (all already done).")
        return

    write_lock = threading.Lock()
    fh = out_path.open("a", encoding="utf-8")
    counter = {"ok": 0, "err": 0, "i": 0}

    def _one(r: dict) -> dict:
        trace_dir = Path(args.trace_dir) if getattr(args, "trace_dir", None) else None
        full_log_dir = Path(args.full_log) if getattr(args, "full_log", None) else None
        if trace_dir is not None or full_log_dir is not None:
            from .llm import clear_trace
            clear_trace()
        rec = {
            "file": r["file"], "pair_id": r.get("pair_id"),
            "is_vulnerable": bool(r.get("is_vulnerable")),
            "gt_cwe": r.get("cwe"), "error": None,
        }
        t0 = time.time()
        try:
            if not os.path.exists(r["_path"]):
                raise FileNotFoundError(r["_path"])
            report = run(r["_path"], "file")
            rec.update(_extract_pred(report))
            blank = _network_blank_error(report, rec)
            if blank:
                rec["error"] = blank
                rec["_no_persist"] = True
        except Exception as e:  # a single file's failure does not stop the batch
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            if trace_dir is not None or full_log_dir is not None:
                from .llm import get_trace
                trace = get_trace()
            if trace_dir is not None:
                try:
                    rec["trace_file"] = _write_trace(trace_dir, r["file"], trace)
                except Exception as e:
                    trace_error = f"{type(e).__name__}: {e}"
                    rec["trace_error"] = trace_error
            if full_log_dir is not None:
                try:
                    rec["full_log_file"] = _write_full_log(full_log_dir, r["file"], trace)
                except Exception as e:
                    rec["full_log_error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_s"] = round(time.time() - t0, 1)
        return rec

    def _emit(rec: dict) -> None:
        with write_lock:
            # A blank run is not persisted: the file stays in the "never run"
            # state so the next resume retries it automatically (no need for
            # --retry-errors).
            if not rec.pop("_no_persist", False):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            counter["i"] += 1
            if rec["error"]:
                counter["err"] += 1
                tail = f"ERROR {rec['error'][:60]}"
            else:
                counter["ok"] += 1
                tail = f"pred_vuln={rec['pred_vuln']} cwes={rec.get('pred_cwes')}"
            print(f"  [{counter['i']}/{len(todo)}] {rec['file']} "
                  f"(gt_vuln={rec['is_vulnerable']}) {tail}")

    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_one, r): r for r in todo}
                for fut in as_completed(futs):
                    _emit(fut.result())
        else:
            for r in todo:
                _emit(_one(r))
    finally:
        fh.close()
    print(f"[predict] done: succeeded {counter['ok']}, failed {counter['err']},"
          f" written to {out_path}")


# ===================== score: compute metrics =====================


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "N": n,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "fpr": round(fpr, 4),
    }


def _predicted_positive(rec: dict, mode: str) -> bool:
    """Whether a file counts as "vulnerable" (one positive prediction) under the
    given view.
    detection: any reported is_vuln makes it positive.
    cwe-family: the same rule for vulnerable and patched files -- both require the
      predicted CWE to hit "the top-level family of this sample's gt_cwe".
      Patched-file semantics matter: what the patch fixes is "the original
      vulnerability's CWE family". If the model still reports that family for the
      patched file -> judged a false positive; if it reports only other families
      -> the original vulnerability is considered fixed and is not held to the
      original CWE, so it counts as negative (it may be a different problem
      introduced by the patch, which this view does not charge as an FP). A
      patched file's gt_cwe is the original CWE of its paired vulnerable file
      (identical within a pair in the manifest), so rec.gt_cwe can be used
      directly."""
    if not rec.get("pred_vuln"):
        return False
    if mode == "detection":
        return True
    # cwe-family: for both sides, positive means "the prediction hits this
    # sample's gt_cwe family".
    return _cwe_family_hit(rec.get("gt_cwe"), rec.get("pred_cwes") or [])


def _confusion(recs: list[dict], mode: str) -> dict:
    tp = fp = tn = fn = 0
    for rec in recs:
        pos = _predicted_positive(rec, mode)
        gt = rec.get("is_vulnerable")
        if gt and pos:
            tp += 1
        elif gt and not pos:
            fn += 1
        elif not gt and pos:
            fp += 1
        else:
            tn += 1
    return _metrics(tp, fp, tn, fn)


def _pair_metrics(recs: list[dict]) -> dict:
    """PrimeVul pair metrics are computed strictly under M1 detection, with no
    reference to CWE families.

    Any confirmed vulnerability on the patched side is an FP; P-C both sides
    correct, P-V vulnerable side only, P-B patched side only, P-R both wrong.
    pair-acc = P-C / number of pairs."""
    by_pair: dict[str, list[dict]] = {}
    for rec in recs:
        by_pair.setdefault(rec.get("pair_id") or rec["file"], []).append(rec)
    pc = pv = pb = pr = incomplete = 0
    for items in by_pair.values():
        v = [x for x in items if x.get("is_vulnerable")]
        p = [x for x in items if not x.get("is_vulnerable")]
        if len(v) != 1 or len(p) != 1:
            incomplete += 1
            continue
        v_ok = _predicted_positive(v[0], "detection")
        p_ok = not _predicted_positive(p[0], "detection")
        if v_ok and p_ok:
            pc += 1
        elif v_ok and not p_ok:
            pv += 1
        elif not v_ok and p_ok:
            pb += 1
        else:
            pr += 1
    total = pc + pv + pb + pr
    return {
        "pairs_scored": total, "pairs_incomplete": incomplete,
        "P-C": pc, "P-V": pv, "P-B": pb, "P-R": pr,
        "pair_accuracy": round(pc / total, 4) if total else 0.0,
    }


def _apply_threshold(rec: dict, thr: float) -> tuple[bool, list[str]]:
    """Recompute (pred_vuln, pred_cwes) from vuln_findings at the given
    threshold. Records lacking that field fall back to the ready-made
    pred_vuln/pred_cwes."""
    detail = rec.get("vuln_findings")
    if detail is None:
        return bool(rec.get("pred_vuln")), rec.get("pred_cwes") or []
    kept = [d for d in detail if float(d.get("conf") or 0.0) >= thr]
    cwes = sorted({d.get("cwe") for d in kept if d.get("cwe")})
    return bool(kept), cwes


def _sweep(args) -> None:
    """Sweep VULN_CONF_MIN offline to find the best cwe-family pair_accuracy,
    without running the LLM."""
    recs = [r for r in _load_done(Path(args.out)).values() if r.get("error") is None]
    if not recs:
        print("No valid predictions; run predict first.")
        return
    print(f"\n=== Threshold sweep ({args.out}, file=mode=cwe-family,"
          f" pair=M1 detection, N={len(recs)}) ===")
    print("thr    TP FP TN FN  fpr    P-C P-V P-B P-R  pair_acc")
    for thr in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        adj = []
        for r in recs:
            pv, pc = _apply_threshold(r, thr)
            adj.append({**r, "pred_vuln": pv, "pred_cwes": pc})
        m = _confusion(adj, "cwe-family")
        pm = _pair_metrics(adj)
        print(f"{thr:<6} {m['TP']:>2} {m['FP']:>2} {m['TN']:>2} {m['FN']:>2} "
              f"{m['fpr']:<6} {pm['P-C']:>3} {pm['P-V']:>3} {pm['P-B']:>3} "
              f"{pm['P-R']:>3}  {pm['pair_accuracy']}")


def _score_summary(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("error") is None]
    err = [r for r in recs if r.get("error")]
    return {
        "scorer_version": SCORER_VERSION,
        "records": {"total": len(recs), "valid": len(ok), "errors": len(err)},
        "detection": _confusion(ok, "detection") if ok else {},
        "cwe_family": _confusion(ok, "cwe-family") if ok else {},
        "pair_detection": _pair_metrics(ok) if ok else {},
    }


def _score(args) -> None:
    out_path = Path(args.out)
    recs = list(_load_done(out_path).values())
    ok = [r for r in recs if r.get("error") is None]
    err = [r for r in recs if r.get("error")]
    print(f"\n=== Evaluation ({out_path}) ===")
    print(f"records {len(recs)}: valid {len(ok)}, errored {len(err)}"
          + (" (errored files excluded from metrics)" if err else ""))
    if not ok:
        print("No valid predictions; run predict first.")
        return

    for mode, label in (("detection", "pure detection"),
                        ("cwe-family", "CWE family-level")):
        print(f"\n── {label} (mode={mode}) ──")
        m = _confusion(ok, mode)
        print(f"  file level N={m['N']}  TP={m['TP']} FP={m['FP']} "
              f"TN={m['TN']} FN={m['FN']}")
        print(f"    accuracy={m['accuracy']}  precision={m['precision']}  "
              f"recall={m['recall']}  f1={m['f1']}  fpr={m['fpr']}")
        if mode == "detection":
            pm = _pair_metrics(ok)
            print(f"  Pair (M1 detection only) count={pm['pairs_scored']}"
                  + (f" (incomplete pairs {pm['pairs_incomplete']} skipped)"
                     if pm["pairs_incomplete"] else ""))
            print(f"    P-C={pm['P-C']} P-V={pm['P-V']} P-B={pm['P-B']} "
                  f"P-R={pm['P-R']}  pair_accuracy={pm['pair_accuracy']}")

    if args.per_file_out:
        _write_per_file(ok, Path(args.per_file_out))
        print(f"\nper-file detail written to {args.per_file_out}")


def _write_per_file(recs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "pair_id", "gt_vulnerable", "gt_cwe", "pred_vuln",
            "pred_cwes", "detection_correct", "cwe_family_correct",
            "n_findings", "elapsed_s",
        ])
        for r in recs:
            det_pos = _predicted_positive(r, "detection")
            fam_pos = _predicted_positive(r, "cwe-family")
            gt = r.get("is_vulnerable")
            w.writerow([
                r["file"], r.get("pair_id"), gt, r.get("gt_cwe"),
                r.get("pred_vuln"), "|".join(r.get("pred_cwes") or []),
                det_pos == gt, fam_pos == gt,
                r.get("n_findings"), r.get("elapsed_s"),
            ])


def _final_records(rows: list[dict], journal: Path) -> list[dict]:
    """Take each file's last attempt from the mutable journal for the currently
    selected set, and pin the GT fields."""
    done = _load_done(journal)
    missing = [row["file"] for row in rows if row["file"] not in done]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"cannot freeze an incomplete run: {len(missing)}"
                         f" selected file records are missing:"
                         f" {preview}{suffix}")

    final: list[dict] = []
    for row in rows:
        rec = dict(done[row["file"]])
        rec.update({
            "file": row["file"],
            "pair_id": row.get("pair_id"),
            "is_vulnerable": bool(row.get("is_vulnerable")),
            "gt_cwe": row.get("cwe"),
        })
        final.append(rec)
    return final


def _source_digests(rows: list[dict]) -> dict[str, Optional[str]]:
    digests: dict[str, Optional[str]] = {}
    for row in rows:
        path = Path(row["_path"])
        digests[row["file"]] = _sha256_file(path) if path.exists() else None
    return digests


def _artifact_args(args) -> dict[str, Any]:
    """CLI arguments carry no secrets; the model identity is listed separately so
    a reviewer can align inputs when reproducing."""
    return {
        key: value for key, value in vars(args).items()
        if key not in {"run_dir", "rescore_run"}
    }


def _finalize_run(run_dir: Path, rows: list[dict], args) -> Path:
    """Seal this run's selection into a non-overwritable, self-contained
    evaluation artifact."""
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists, refusing to"
                              f" overwrite: {run_dir}")

    frozen_rows = _artifact_rows(rows)
    predictions = _final_records(rows, Path(args.out))
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    _write_jsonl(manifest_path, frozen_rows)
    _write_jsonl(predictions_path, predictions)

    summary = _score_summary(predictions)
    _write_json(run_dir / "scores.json", summary)
    _write_per_file(
        [rec for rec in predictions if rec.get("error") is None],
        run_dir / "per_file.csv",
    )
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scorer_version": SCORER_VERSION,
        "journal_path": str(Path(args.out)),
        "manifest_path": str(Path(args.manifest)),
        "arguments": _artifact_args(args),
        "git": _git_fingerprint(),
        "model": {
            key: getattr(config, key, None)
            for key in ("LLM_MODEL", "LLM_TEMPERATURE", "LLM_SEED", "LLM_TOP_P")
        },
        "source_sha256": _source_digests(rows),
        "files": {
            "manifest.jsonl": _sha256_file(manifest_path),
            "predictions.jsonl": _sha256_file(predictions_path),
        },
    }
    _write_json(run_dir / "run.json", metadata)
    return run_dir


def _read_frozen_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not an object")
        rows.append(row)
    return rows


def _load_run_artifact(run_dir: Path) -> tuple[dict, list[dict]]:
    metadata_path = run_dir / "run.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run artifact: {metadata_path}") from exc
    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact schema version is incompatible")

    expected = metadata.get("files") or {}
    for name in ("manifest.jsonl", "predictions.jsonl"):
        path = run_dir / name
        if not path.is_file() or expected.get(name) != _sha256_file(path):
            raise ValueError(f"artifact integrity check failed: {name}")

    manifest = _read_frozen_jsonl(run_dir / "manifest.jsonl")
    predictions = _read_frozen_jsonl(run_dir / "predictions.jsonl")
    manifest_files = [row.get("file") for row in manifest]
    prediction_files = [row.get("file") for row in predictions]
    if manifest_files != prediction_files or len(set(prediction_files)) != len(prediction_files):
        raise ValueError("the artifact's manifest and predictions are not in"
                         " one-to-one correspondence")
    return metadata, predictions


def _rescore_run(run_dir: Path) -> Path:
    metadata, predictions = _load_run_artifact(run_dir)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    index = 1
    while (metrics_dir / f"scores-v{SCORER_VERSION}-{index:03d}.json").exists():
        index += 1
    score_path = metrics_dir / f"scores-v{SCORER_VERSION}-{index:03d}.json"
    detail_path = metrics_dir / f"per_file-v{SCORER_VERSION}-{index:03d}.csv"
    _write_json(score_path, {
        "artifact_schema_version": metadata["artifact_schema_version"],
        "scorer_version": SCORER_VERSION,
        "prediction_sha256": metadata["files"]["predictions.jsonl"],
        "summary": _score_summary(predictions),
    })
    _write_per_file(
        [rec for rec in predictions if rec.get("error") is None], detail_path,
    )
    print(f"metrics recomputed from the frozen artifact: {score_path}")
    return score_path


# ===================== CLI =====================


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="PrimeVul batch evaluation: accuracy/recall/F1 + pair"
                    " correctness (two CWE views)")
    ap.add_argument("--manifest", default="data/primevul/manifest.jsonl",
                    help="path to manifest.jsonl")
    ap.add_argument("--data-root", default=None,
                    help="data root (the file field is relative to it; defaults"
                         " to the manifest's directory)")
    ap.add_argument("--out", default="runs/pred.jsonl",
                    help="predictions JSONL (predict appends, score reads)")
    ap.add_argument("--limit", type=int, default=0,
                    help="take only the first N manifest rows")
    ap.add_argument("--sample", type=int, default=0,
                    help="sample K pairs at random (keeping pairs intact)")
    ap.add_argument("--seed", type=int, default=0, help="sampling random seed")
    ap.add_argument("--workers", type=int, default=1,
                    help="file-level concurrency (mind the gateway's TPM rate"
                         " limit; ≤4 recommended)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run files whose error != null last time")
    ap.add_argument("--score-only", action="store_true",
                    help="do not run the LLM; recompute metrics from --out"
                         " (resume-journal view)")
    ap.add_argument("--run-dir", default=None,
                    help="freeze the selected results into a new artifact"
                         " directory (cannot be overwritten)")
    ap.add_argument("--trace-dir", default=None,
                    help="save the full LLM trace per file (system/user/response),"
                         " useful for diagnosing misses")
    ap.add_argument("--full-log", default=None,
                    help="save, per file, the same readable full LLM terminal"
                         " trace that run --trace prints, into this directory")
    ap.add_argument("--rescore-run", default=None,
                    help="recompute metrics only from a frozen artifact"
                         " directory, without running the LLM")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the confidence threshold offline (no LLM) to"
                         " find the best pair_accuracy")
    ap.add_argument("--per-file-out", default=None,
                    help="export a per-file correctness detail CSV")
    args = ap.parse_args(argv)

    if args.rescore_run:
        if args.score_only or args.sweep or args.run_dir:
            ap.error("--rescore-run cannot be combined with --score-only,"
                     " --sweep or --run-dir")
        _rescore_run(Path(args.rescore_run))
        return 0

    if args.run_dir and (args.score_only or args.sweep):
        ap.error("--run-dir can only be used with a normal predict run")

    if (args.trace_dir or args.full_log) and (args.score_only or args.sweep):
        ap.error("--trace-dir / --full-log can only be used with a normal"
                 " predict run")

    if args.sweep:
        _sweep(args)
        return 0

    if args.score_only:
        _score(args)
        return 0

    if args.trace_dir or args.full_log:
        from .llm import enable_trace
        enable_trace(full=True)

    manifest = Path(args.manifest)
    data_root = Path(args.data_root) if args.data_root else manifest.parent
    rows = _load_manifest(manifest, data_root)
    rows = _select_rows(rows, args)
    _predict(rows, args)
    if args.run_dir:
        run_dir = _finalize_run(Path(args.run_dir), rows, args)
        print(f"frozen evaluation artifact written to {run_dir}")
        # The terminal shows scores for the frozen selection only; history from
        # the resume journal must not leak in.
        original_out = args.out
        args.out = str(run_dir / "predictions.jsonl")
        _score(args)
        args.out = original_out
    else:
        _score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

