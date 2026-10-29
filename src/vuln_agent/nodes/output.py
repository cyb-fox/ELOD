"""output (deterministic): aggregate verdicts and emit the structured result;
partial marks budget exhaustion.

Root-cause de-duplication happens before aggregation (_dedup.dedup_findings):
several propagation points sharing one root cause collapse into a single primary
finding, so vulnerabilities_found reflects the number of independent root causes
while raw_verdicts keeps the original count for reconciliation.

File-level threshold gate: only findings with confidence >=
config.VULN_CONF_MIN count as a "confirmed vulnerability"; those below the
threshold are downgraded to suspected (kept in the report for human review but
not carrying the file-level positive). This gate is what keeps low-confidence
suspicions out of the file-level verdict, which is what bounds the
false-positive rate.

Three-valued verdict: non-positive points are separated into `safe_proved`
(safety positively proved) and `underdetermined` (a key obligation not proved,
often because the caller or the callee implementation is invisible from a single
file). The report's `file_resolution` aggregates over points: any confirmed
positive -> vuln; otherwise any undetermined point -> underdetermined; all proved
safe -> safe_proved. `underdetermined` is not "safe": for the metric view see
bench/metrics (it counts as negative by default, with a separate count).

**Hard rule: an empty output must distinguish "comparison completed, no
difference" from "the chain ran without inspecting"**. `safe_proved` is a
conclusion that requires positive evidence. `sinks_examined == 0` has two sources
that must not be conflated:

  · **secure_diff (model-comparison safe baseline) ran to completion and its zero
    differences passed the gate** (`_secure_diff_outcome == "no_diff"`): the
    model's generated safe version is identical to the real code under the
    deterministic diff gate, and neither the rule floor nor the triage bridge hit
    -- the comparison is itself positive safety evidence -> `safe_proved`;
  · **the comparison did not complete** (LLM output unusable / counterfactual gate
    off / blank budget run) -> always `underdetermined`.

The other two forms of "inspected nothing" are unchanged and still always
`underdetermined`:
  · `findings` is empty (no point reached a verdict);
  · there are points but **none** is `safe_proved` (all downgraded or
    undetermined).

Additionally, a classifier (select_knowledge) with no candidate CWE is not a safe
short-circuit condition -- it only affects knowledge loading, and the comparison
baseline must still run. This closes off the "classifier short-circuits -> zero
sinks -> assumed safe" path.
"""
from __future__ import annotations

from .. import config
from ..state import VulnState
from ._dedup import dedup_findings


def _resolution_of(f: dict) -> str:
    """The finding's three-valued verdict label; records with no such field fall
    back to inference from is_vuln."""
    r = str(f.get("resolution") or "")
    if r:
        return r
    return "vuln" if f.get("is_vuln") else "safe_proved"


def output(state: VulnState) -> dict:
    raw_verdicts = list(state.get("verdict", []) or [])
    budget = state.get("budget", {})
    # Root-cause de-duplication: fold propagation intermediate points, one entry
    # per independent root cause
    findings = dedup_findings(raw_verdicts, state.get("sinks", []) or [])
    thr = config.VULN_CONF_MIN
    # Threshold gate: meeting it makes a confirmed vulnerability; a point judged
    # vuln with insufficient confidence is downgraded to suspected (not carrying
    # the file-level positive).
    confirmed = []
    for f in findings:
        if not f.get("is_vuln"):
            continue
        if float(f.get("confidence", 0.0) or 0.0) >= thr:
            confirmed.append(f)
        else:
            f["suspected"] = True  # in doubt: below threshold, kept for display
                                     # but not counted in vulnerabilities_found
    vulns = confirmed
    underdetermineds = [f for f in findings
                if not f.get("is_vuln") and _resolution_of(f) == "underdetermined"]
    safe_proved = [f for f in findings
                   if not f.get("is_vuln") and _resolution_of(f) == "safe_proved"]
    n_sinks = len(state.get("sinks", []) or [])
    # Empty-output gate: safe_proved requires positive evidence, so with no
    # proved point we may not claim "proved safe".
    unresolved_note = ""
    if vulns:
        file_resolution = "vuln"
    elif underdetermineds:
        file_resolution = "underdetermined"
    elif n_sinks == 0:
        # Empty-scan gate: zero sensitive operations means underdetermined.
        # safe_proved needs at least one reviewed sensitive operation as positive
        # evidence (Eq. 6); a comparison baseline that completed with no
        # differences shows only that "the generated safe version matches the
        # original", which cannot substitute for reviewing the original's
        # sensitive operations.
        file_resolution = "underdetermined"
        unresolved_note = "No sensitive operation was reviewed (sinks_examined=0): " \
                          "an empty scan is not evidence of safety"
    elif not findings:
        file_resolution = "underdetermined"
        unresolved_note = ("Suspicious points exist but none reached a verdict"
                           " (findings is empty): the chain did not complete,"
                           " manual review required")
    elif not safe_proved:
        file_resolution = "underdetermined"
        unresolved_note = ("No point was positively proved safe (mostly"
                           " low-confidence downgrades / undetermined),"
                           " manual review required")
    else:
        file_resolution = "safe_proved"
    excluded = [
        f["sink_id"] for f in (state.get("forensics") or [])
        if f.get("screen") == "exclude"
    ]
    # underdetermined must surface its risk: the undetermined points' CWE plus
    # the reason they are undetermined are aggregated at the report's top level
    # for human review and downstream statistics (these points do not count in
    # vulnerabilities_found and do not carry the file-level positive).
    underdetermined_risks = [
        {
            "sink_id": f.get("sink_id"),
            "cwe_id": f.get("cwe_id", ""),
            "line": (f.get("location") or {}).get("line_start"),
            "risk": f.get("risk", ""),
            "unresolved_reason": f.get("unresolved_reason", ""),
        }
        for f in underdetermineds
    ]

    report = {
        "target": state.get("target_path"),
        "partial": bool(state.get("partial", False)),
        "file_resolution": file_resolution,
        "summary": {
            "sinks_examined": n_sinks,
            "screened_out": len(excluded),
            "verdicts": len(findings),
            "raw_verdicts": len(raw_verdicts),
            "vulnerabilities_found": len(vulns),
            "underdetermined_points": len(underdetermineds),
            "safe_proved_points": len(safe_proved),
            "conf_threshold": thr,
            "families_considered": state.get("family_candidates", []),
            # secure_diff comparison outcome (no_diff/diff/llm_unavailable/
            # disabled): the only legitimate basis for recording an empty output
            # as safe_proved, persisted in the report for check_regression R1.
            "secure_diff_outcome": str(state.get("_secure_diff_outcome") or ""),
        },
        "budget": {
            "tokens_used": budget.get("tokens_used"),
            "tokens_max": budget.get("tokens_max"),
            "llm_calls": budget.get("llm_calls"),
            "calls_max": budget.get("calls_max"),
            "failed_calls": budget.get("failed_calls", 0),
            # The call succeeded but the output was unparseable -> some node
            # silently degraded. When two runs disagree, read this number first:
            # non-zero means the report is a degraded product rather than a
            # stable conclusion.
            "json_failures": budget.get("json_failures", 0),
        },
        "findings": findings,
        "errors": state.get("errors", []) or [],
    }
    if underdetermined_risks:
        report["underdetermined_risks"] = underdetermined_risks
    if unresolved_note:
        report["unresolved_note"] = unresolved_note
    # Degradation trace: which nodes got no usable LLM output in this run and
    # which fallback they took. It **does not affect the verdict**; it only makes
    # "this report is a degraded product" visible, so when two runs disagree,
    # read this list first.
    degraded = list(state.get("degraded") or [])
    if degraded:
        report["degraded"] = degraded
    # Files short-circuited by the scan-profile prescreen: marked as cheaply
    # clean (LLM chain not run), so the savings can be counted.
    if state.get("_prescreen_clean"):
        report["prescreened_clean"] = True
        report["summary"]["prescreen_reasons"] = state.get("_prescreen_reasons", [])
    if report["partial"]:
        report["note"] = "Budget exhausted, the result is incomplete (partial)."
    return {"report": report}
