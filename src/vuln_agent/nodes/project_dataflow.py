"""project_dataflow (cross-file taint-chain forensics, project mode only; a single
LLM call).

Runs after arbiter and before output. In file mode (no project_index) it passes
straight through with zero LLM calls.

The blind spot of the single-file view: when the taint source lives in another
file, arbiter can only call those points underdetermined / safe because the source
is invisible. This node recovers exactly those points:
1. Candidates = sinks with an external taint token that are not currently
   confirmed as vulnerabilities (safe / below threshold).
2. Use the project index to grep those taint tokens across files, retrieving
   file:line evidence; and feed in, as "producer snippets", the bodies of callees
   invoked on the candidate lines but defined in other files.
3. The single LLM call **only proposes evidence-backed candidate facts**
   (obligation_id + direction + a concrete execution certificate), emitting no
   is_vuln/confidence; this node hands the candidate facts to the single
   discharger `obligation_checker.discharge` for deterministic rechecking, then
   `oc.resolve` aggregates them into a verdict -- the same evidence boundary as
   arbiter (S1).
4. One-sided: a verdict is written back only when `resolve` rules `vuln`
   (recovering missed detections). A safe_proved / underdetermined resolve never
   overrides an existing verdict -- a guard already present on the producer side
   (a patch shape) should not be re-judged, and insufficient cross-file evidence
   must never repaint an existing conclusion as safe.

When the budget is blown, budget_guard short-circuits and goes straight to output.
"""
from __future__ import annotations

from .. import config
from ..llm import prompts as P
from ..state import Verdict, VulnState
from ..tools import obligation_checker as oc
from ..tools import obligations as ob
from ..tools import segmenter as sg
from ..tools.context_search import ContextSearcher
from .arbiter import _duty_text
from ._common import cfg_for_line, make_llm, risk_window

# Confidence ceiling for candidate inclusion: a point that already passed the
# threshold gate (a confirmed vulnerability) needs no cross-file rechecking.
_CANDIDATE_CONF_MAX = config.VULN_CONF_MIN
# Cap on the number of cross-file producer snippets, to bound prompt size.
_MAX_PRODUCERS = 6


def project_dataflow(state: VulnState) -> dict:
    index = state.get("project_index")
    if not index:
        return {}  # file mode: pass through, no LLM call

    sinks = state.get("sinks", []) or []
    verdicts = state.get("verdict", []) or []
    candidates = _select_candidates(sinks, verdicts)
    if not candidates:
        return {}

    llm = make_llm(state, "project_dataflow")
    current = state.get("target_path", "")

    # Cross-file grep: gather every candidate's taint tokens, then search once for
    # file:line evidence.
    searcher = ContextSearcher(
        state["source_text"], scope_label=index.get("root", "<project>"),
        project_index=index, current_path=current,
    )
    all_tokens: list[str] = []
    for c in candidates:
        for t in c.get("tainted_tokens") or []:
            if t not in all_tokens:
                all_tokens.append(t)
    cross_evidence = searcher.search(all_tokens) if all_tokens else ""

    # Producer snippets: callees invoked on the candidate lines but defined in
    # other files (cross-file sources / propagators).
    producers = _collect_producers(state, index, candidates, current)

    data = llm.complete_json(
        P.PROJECT_DATAFLOW_SYSTEM,
        P.project_dataflow_user(
            _rel(index, current), candidates, cross_evidence, producers,
            list((state.get("knowledge") or {}).keys()),
            family_context=state.get("family_context") or [],
        ),
    )

    sink_by_id = {s["sink_id"]: s for s in sinks}
    new_findings = _to_findings(
        data, state, sink_by_id, {c["sink_id"] for c in candidates}, cross_evidence,
    )
    if not new_findings:
        return {}
    merged = _merge_verdicts(verdicts, new_findings)
    return {"verdict": merged}


def _select_candidates(sinks, verdicts) -> list[dict]:
    """Pick the cross-file recheck candidates: sinks with an external taint that
    are not currently confirmed as vulnerabilities."""
    conf_by_id: dict[str, float] = {}
    isvuln_by_id: dict[str, bool] = {}
    res_by_id: dict[str, str] = {}
    for v in verdicts:
        sid = v.get("sink_id", "")
        conf_by_id[sid] = float(v.get("confidence", 0.0) or 0.0)
        isvuln_by_id[sid] = bool(v.get("is_vuln"))
        res_by_id[sid] = str(v.get("resolution") or "")
    out: list[dict] = []
    for s in sinks:
        # contract has already been cross-project verified by risk_forensics
        # before arbiter; this is the traditional taint recovery branch, and it
        # must not bypass the hard constraints on a positive call
        # (contract_verified + violated_contract).
        if s.get("change_kind") == "contract":
            continue
        toks = [t for t in (s.get("tainted_tokens") or []) if isinstance(t, str) and t]
        if not toks and s.get("sink_type") not in {"arith", "deref"}:
            continue  # no external taint, so there is no cross-file chain to speak of
        sid = s["sink_id"]
        conf = conf_by_id.get(sid, 0.0)
        confirmed = isvuln_by_id.get(sid, False) and conf >= _CANDIDATE_CONF_MAX
        if confirmed:
            continue  # already a confirmed vulnerability, no cross-file recovery needed
        # The three-valued verdict passes through unchanged
        # (vuln/safe_proved/underdetermined): `underdetermined` is exactly the
        # "source invisible from one file" class this node recovers, and writing
        # it as safe would mislead the proposer into thinking this point is
        # already safe.
        prior = res_by_id.get(sid) or ("vuln" if isvuln_by_id.get(sid) else "underdetermined")
        out.append({
            "sink_id": sid, "func_name": s["func_name"], "line": s["line"],
            "snippet": s.get("snippet", ""), "tainted_tokens": toks,
            "prior": f"{prior}(conf={round(conf, 2)})",
        })
    return out


def _collect_producers(state, index, candidates, current) -> str:
    """Assemble the bodies (with file:line) of callees invoked on the candidate
    risky lines but defined in other files, as the producer snippets."""
    from ..tools import sink_locator as sl

    symbols: dict = index.get("symbols", {})
    structured = index.get("symbol_index") or {}
    definitions: dict = structured.get("definitions", {})
    sources: dict = index.get("sources", {})
    # The return-producing call often precedes the sink; inspect the bounded
    # risk window rather than only the division/dereference line itself.
    cand_lines = {line for c in candidates
                  for line in range(max(1, c["line"] - config.RISK_WINDOW), c["line"] + 1)}
    # Find callee names near the candidate lines in this file.
    called: list[str] = []
    for func in state.get("functions", []):
        for _node, name, _arg, line in sl._iter_calls(state["source_text"], func):
            if line in cand_lines and name not in called:
                called.append(name)

    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    # Also show callers: a certificate may start outside the consumer file and
    # carry the unsafe value through actual parameters into this function.
    for candidate in candidates:
        for call in (structured.get("call_sites", {}).get(candidate.get("func_name"), []) or []):
            caller = str(call.get("caller") or "")
            path = str(call.get("path") or "")
            if not caller or not path or path == current or (caller, path) in seen:
                continue
            seen.add((caller, path))
            spans = [d for d in definitions.get(caller, []) if d.get("path") == path]
            snippet = _func_snippet(sources.get(path, ""), caller, path, index, spans)
            if snippet:
                blocks.append(snippet)
            if len(blocks) >= _MAX_PRODUCERS:
                return "\n\n".join(blocks)
    for name in called:
        paths = [item.get("path", "") for item in definitions.get(name, [])]
        if not paths:
            paths = symbols.get(name, [])
        for path in sorted(set(paths)):
            if path == current:
                continue  # only cross-file definitions wanted
            key = (name, path)
            if key in seen:
                continue
            seen.add(key)
            spans = [item for item in definitions.get(name, [])
                     if item.get("path") == path]
            snippet = _func_snippet(sources.get(path, ""), name, path, index, spans)
            if snippet:
                blocks.append(snippet)
            if len(blocks) >= _MAX_PRODUCERS:
                return "\n\n".join(blocks)
    return "\n\n".join(blocks)


def _func_snippet(src: str, func_name: str, path: str, index: dict,
                  spans: list[dict] | None = None) -> str:
    """Get the line-numbered source of a named function in a given file (compact
    feed form), prefixed with its path relative to the project root."""
    if not src:
        return ""
    lang = sg.detect_lang(path)
    for span in spans or []:
        start = int(span.get("line_start") or 0)
        end = int(span.get("line_end") or 0)
        if start <= 0 or end < start:
            continue
        func = {
            "func_name": func_name,
            "line_start": start,
            "line_end": end,
            "lang": span.get("lang", lang),
        }
        unit = sg.prepare_unit(src, func)
        return f"[{_rel(index, path)}] function {func_name}:\n{unit['content']}"
    try:
        funcs = sg.split_functions(src, lang)
    except Exception:
        return ""
    for f in funcs:
        if f["func_name"] == func_name:
            unit = sg.prepare_unit(src, f)
            return f"[{_rel(index, path)}] function {func_name}:\n{unit['content']}"
    return ""


def _to_findings(
    data, state, sink_by_id: dict, candidate_ids: set[str], cross_evidence: str,
) -> list[Verdict]:
    """Hand the LLM's proposed cross-file candidate facts to the single
    discharger for rechecking and aggregate them into a verdict (S1 evidence
    boundary).

    The LLM is only a proposer here: it supplies obligation_id + direction +
    file:line anchors, `oc.discharge` performs the deterministic rechecking
    (actual and formal parameters on trusted source, return values, assignments,
    branches and the actual unsafe operation), and `oc.resolve` aggregates them
    as a pure function. **Only `vuln` is written back**: safe_proved /
    underdetermined never override an existing verdict -- insufficient
    cross-file evidence must never repaint an existing conclusion as safe (I3).
    """
    if not isinstance(data, list):
        return []
    file = state.get("target_path", "")
    index = state.get("project_index") or {}
    source_snapshot = {_rel(index, path): source for path, source in index.get("sources", {}).items()}
    target = _rel(index, file)
    if target not in source_snapshot or source_snapshot[target] != state.get("source_text"):
        return []  # Never replay a stale producer/consumer snapshot.
    facts_by_sink: dict[str, list[dict]] = {}
    cwe_by_sink: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("sink_id") or "")
        if sid not in candidate_ids:
            continue  # only points sent for rechecking this round are accepted;
                      # no sink may be conjured up out of thin air
        oid = str(item.get("obligation_id") or "")
        direction = str(item.get("direction") or "").upper()
        anchors = [a for a in (item.get("anchors") or []) if isinstance(a, str)]
        if not oid or direction not in ("VIOLATED", "SATISFIED") or (not anchors and not isinstance(item.get("cross_file_certificate"), dict)):
            continue
        fact: dict = {
            "obligation_id": oid, "direction": direction, "anchors": anchors,
            # The evidence in this node is cross-file by nature: it must be
            # rechecked by replaying the certificate against the source.
            "cross_file": bool(item.get("cross_file", True)),
        }
        if isinstance(item.get("cross_file_certificate"), dict):
            fact["cross_file_certificate"] = item["cross_file_certificate"]
        if item.get("derived_from"):
            fact["derived_from"] = str(item["derived_from"])
        facts_by_sink.setdefault(sid, []).append(fact)
        if item.get("cwe_id") and sid not in cwe_by_sink:
            cwe_by_sink[sid] = str(item["cwe_id"])

    out: list[Verdict] = []
    for sid, facts in facts_by_sink.items():
        sink = sink_by_id.get(sid)
        if sink is None:
            continue
        line = int(sink.get("line") or 0)
        window = risk_window(state, line) if line else ""
        # The project-level recheck reuses the same candidate--verification
        # boundary as the single-file path: the secure_diff candidates in the
        # sink snapshot must pass through discharge before becoming a
        # rule-able state.
        all_facts = oc.seeded_candidate_facts(sink.get("obligations") or []) + facts
        obs, refs = oc.discharge(
            sink, all_facts,
            consumer_source=window, searched_source=cross_evidence,
            cfg=cfg_for_line(state, line),
            project_sources=source_snapshot,
            target_path=target,
        )
        resolution, why = oc.resolve(obs)
        if resolution != "vuln":
            continue  # one-sided: only missed detections are recovered; no
                      # point is ever re-judged as safe / undetermined
        violated = oc.critical_violated(obs)
        duty = _duty_text(violated[0]) if violated else ""
        sink_type = sink.get("sink_type", "")
        cwe = cwe_by_sink.get(sid) or str(sink.get("cwe_id") or "") \
            or ob.cwe_hint(sink_type)
        certificates = [check["replay"]["certificate"] for obligation in obs
                        for check in obligation.get("verification_audit", {}).get("checks", [])
                        if (check.get("replay") or {}).get("accepted")]
        assumption = (f"Conditional C API counterexample, entry point and"
                      f" concrete inputs: {certificates}")
        basis = [duty, assumption] + [f"Cross-file replay evidence: {r}"
                                      for r in refs[:2] + refs[-2:]]
        out.append(Verdict(
            sink_id=sid,
            is_vuln=True,
            resolution="vuln",
            cwe_id=cwe,
            location={"file": file, "line_start": line, "line_end": line},
            # Source-replayed certificate; confidence is fixed by the checker.
            confidence=config.CONF_WITNESS,
            basis=basis,
            fix_suggestion="",
            missing_guard=duty if sink.get("change_kind") != "contract" else "",
            violated_contract=duty if sink.get("change_kind") == "contract" else "",
            origin="project",
            root_cause=f"{cwe} conditional cross-file C API execution counterexample",
            obligations=obs,
        ))
    return out


def _merge_verdicts(verdicts, new_findings) -> list[Verdict]:
    """Merge the **positive** conclusions found by cross-file rechecking into the
    existing verdicts: a matching sink_id is overwritten (the cross-file view is
    more complete and exists precisely to recover misses whose source is
    invisible from one file), otherwise the finding is appended. _to_findings
    already guarantees only resolution=="vuln" is produced, so an overwrite here
    can only lift underdetermined/safe to vuln, never the reverse."""
    by_id = {v.get("sink_id"): v for v in verdicts}
    for nf in new_findings:
        by_id[nf["sink_id"]] = nf
    # Preserve order: the original order first, with overwrites updated in place;
    # appends go afterwards.
    merged: list[Verdict] = []
    seen: set = set()
    for v in verdicts:
        sid = v.get("sink_id")
        merged.append(by_id[sid])
        seen.add(sid)
    for nf in new_findings:
        if nf["sink_id"] not in seen:
            merged.append(nf)
            seen.add(nf["sink_id"])
    return merged


def _rel(index: dict, path: str) -> str:
    from pathlib import Path
    try:
        return str(Path(path).resolve().relative_to(index["root"]))
    except Exception:
        return path
