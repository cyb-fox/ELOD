"""arbiter (**deterministic verdict aggregation**, zero LLM calls).

The single decision point after the topology slimming down: a verdict is no
longer produced by a third LLM call, but by aggregating the evidence-bearing
obligation snapshots that obligation_checker (the only discharger) wrote during
risk_forensics.

Why this node makes no LLM call:
- Its output is already fully dominated by the S1 evidence boundary gate:
  `is_vuln = LLM says so AND checker proved it`.  In that conjunction the LLM
  term can only re-negate a positive that machine checking has already proved,
  and the grounds it negates on ("no null check is visible", "how many lines
  there are") are precisely the superficial statistics that are symmetric in
  paired evaluation -- the source of pair collapse rather than a remedy.
- Everything a verdict needs (obligation status, evidence anchors, witness
  check conclusions, whether a guard dominates) is already a **deterministic
  fact** on entry.  Spending another whole-batch prompt on restating those facts
  only adds token cost and server-side non-determinism (MoE at temp=0 can still
  flip) without adding information.
- Once the decision is deterministic the flip rate is identically 0, so the
  boundary vote (re-rolling a majority over the knife-edge interval) and the
  metamorphic self-consistency filter (replaying a self-reported repair against
  the original code and judging it a second time) are not part of the pipeline.

Verdict rules (direct implementation of I2/I3, all recomputable):
    1. `oc.resolve(obligations)` three-valued resolution: `vuln` (a critical
       VIOLATED that passed re-checking exists and no proven mitigating guard
       covers it) / `safe_proved` (safety positively proved) / `underdetermined`
       (neither proved).  **The LLM's screen=="exclude" carries no veto** -- it
       is only a forensic summary for human reading; see below.
    2. confidence is tiered by **evidence-source strength**
       (machine_witness > micro_check > llm_anchor); a singleton positive (view
       B produced no hit signal at all) is pushed below the threshold gate and
       downgraded to suspected.
    3. missing_guard / violated_contract are generated directly from the
       predicate of the violated obligation (for human reading);
       fix_suggestion is view A's `secure_lines` indexed by function (no longer
       improvised by the model).

Why the LLM holds no veto:
- `screen=="exclude"` is the proposer's self-assessment, yet the reasons it
  gives for excluding are typically a matter of **missing information** ("the
  caller's source evidence is not reachable from the current window ->
  exclude").  Missing caller information should only lead to "undetermined",
  never to "proved safe".  The direction would be inverted: uncertainty treated
  as proof of innocence.
- The three-valued resolution states this plainly: failing to prove a
  vulnerability != proving safety.  `underdetermined` is an independent
  conclusion, it is recorded in the report, and UNDERDETERMINED critical
  obligations are never repainted as SATISFIED.

The node stays on the graph (name unchanged) to preserve the responsibility
split of sinks -> verdict; it consumes no budget.
"""
from __future__ import annotations

from .. import config
from ..state import Verdict, VulnState
from ..tools import obligation_checker as oc
from ..tools import obligations as ob
from ..tools import trajectory as tj


def _index(items, key="sink_id") -> dict:
    return {it[key]: it for it in (items or [])}


def arbiter(state: VulnState) -> dict:
    """Aggregate evidence-bearing obligation snapshots into verdicts
    deterministically (no LLM, no budget consumption)."""
    forensics = _index(state.get("forensics"))
    baselines = _index(state.get("baseline"), key="func_name")
    verdict: list[Verdict] = []
    for sink in state.get("sinks", []) or []:
        verdict.append(_adjudicate(
            sink, forensics.get(sink["sink_id"]),
            baselines.get(sink["func_name"]), state,
        ))
    return {"verdict": verdict}


def _obligations_of(sink, forensic) -> list:
    """The obligation snapshot a verdict rests on: prefer what the checker wrote
    in forensics (the only ground truth), then secure_diff's candidate list, and
    finally instantiate from sink_type (all UNDERDETERMINED -> must resolve
    safe)."""
    base = (forensic.get("obligations") if isinstance(forensic, dict) else None) \
        or sink.get("obligations")
    return [dict(o) for o in (base or ob.instantiate(sink.get("sink_type", "")))]


def _confidence(violated: list, forensic) -> float:
    """Tier by evidence-source strength (deterministic, recomputable).

    machine_witness (the witness of the four-step machine check on the CFG) is
    the strongest -- it cannot be faked by superficial statistics; micro_check
    (deterministic syntax-level abstract interpretation) comes next; llm_anchor
    (a verbatim anchor that re-verification hit) is the weakest.  Singleton
    down-weighting: when view B has neither a danger-signal hit nor a verified
    contract conflict, a positive supported by the anchor channel alone is
    pushed below VULN_CONF_MIN -> output naturally downgrades it to suspected
    (downgraded, never deleted, and recorded for auditing).  `is_false_defense`
    is not a corroboration item: it is the model's self-assessment against an
    inducing list, and using it to raise confidence would let the prompt endorse
    itself.
    """
    provs = {str(o.get("provenance") or "") for o in violated}
    if "machine_witness" in provs:
        conf = config.CONF_WITNESS
    elif "micro_check" in provs:
        conf = config.CONF_MICRO
    else:
        conf = config.CONF_ANCHOR
    fo = forensic if isinstance(forensic, dict) else {}
    corroborated = bool(fo.get("hit_signals")) or bool(fo.get("contract_verified"))
    if not corroborated and "machine_witness" not in provs:
        conf = min(conf, config.VULN_CONF_MIN - 0.05)
    return round(max(0.0, min(1.0, conf)), 2)


def _duty_text(o: dict) -> str:
    """Render a violated obligation as the human-readable "the guard that is
    missing" (the wording is no longer improvised by the model)."""
    ev = ", ".join(str(e) for e in (o.get("evidence") or [])[:2])
    pred = str(o.get("predicate") or o.get("id") or "")
    return f"obligation {o.get('id', '')} not satisfied: {pred}" + (
        f" (evidence: {ev})" if ev else "")


def _basis(violated: list, forensic, mitigated: list) -> list[str]:
    """Verdict basis = obligation evidence anchors + the view B forensic summary
    (all re-checkable, no model self-report)."""
    out: list[str] = []
    for o in violated:
        out.append(_duty_text(o))
        for e in (o.get("evidence") or [])[:3]:
            ref = f"evidence anchor: {e}"
            if ref not in out:
                out.append(ref)
    for o in mitigated:
        out.append(
            f"mitigating guard proved ({o.get('provenance', '')}): "
            f"{o.get('predicate', '')}")
    fo = forensic if isinstance(forensic, dict) else {}
    ev = str(fo.get("evidence", "")).strip()
    if ev:
        out.append(f"view B forensics: {ev}")
    return out


def _adjudicate(sink, forensic, baseline, state) -> Verdict:
    file = state.get("target_path", "")
    line = int(sink.get("line") or 0)
    loc = {"file": file, "line_start": line, "line_end": line}
    sink_type = sink.get("sink_type", "")
    cwe = str(sink.get("cwe_id") or "") or ob.cwe_hint(sink_type)
    is_contract = sink.get("change_kind") == "contract"
    fo = forensic if isinstance(forensic, dict) else {}

    obs = _obligations_of(sink, forensic)
    violated = oc.critical_violated(obs)
    mitigated = oc.mitigation_proved(obs)
    resolution, why = oc.resolve(obs)
    is_vuln = resolution == "vuln"

    if is_vuln:
        conf = _confidence(violated, forensic)
        basis = _basis(violated, forensic, mitigated)
        duty = _duty_text(violated[0]) if violated else ""
        final_obs = obs
    else:
        conf = 0.0
        duty = ""
        basis = [("proved safe: " if resolution == "safe_proved"
                  else "undetermined (I3 conservative reading): ") + why]
        ev = str(fo.get("evidence", "")).strip()
        if ev:
            basis.append(f"view B forensics: {ev}")
        # I3: safe_proved keeps the SATISFIED statuses exactly as the checker
        # wrote them; underdetermined marks the critical obligations that are
        # still UNDERDETERMINED as UNRESOLVED for the record and **does not**
        # rewrite them as SATISFIED.
        final_obs = obs if resolution == "safe_proved" else oc.mark_unresolved(obs, why)

    fix = str((baseline or {}).get("secure_lines", "") or "")

    tj.record_step(
        node="arbiter", action="aggregate_obligations",
        sink_id=sink["sink_id"], bucket=ob.bucket_key(cwe, sink_type),
        inputs={"change_kind": sink.get("change_kind", "guard"),
                "screen": fo.get("screen", "")},
        observation={"critical_violated": [o.get("id") for o in violated],
                     "mitigation_proved": [o.get("id") for o in mitigated],
                     "resolution": resolution,
                     "confidence": conf},
        state_delta={o["id"]: {"to": o.get("status", "UNDERDETERMINED")}
                     for o in final_obs if o.get("critical")},
        evidence_refs=[e for o in final_obs for e in (o.get("evidence") or [])],
        decision=is_vuln,
    )
    # underdetermined must surface the risk: the CWE class is reported as usual,
    # together with wording that it cannot be confirmed for lack of information
    # outside this file and with the reason the obligation stayed unresolved.
    # This is not a positive verdict -- is_vuln stays False and confidence stays
    # 0 -- it only prevents "undetermined" from becoming silence.  safe_proved
    # carries none of these fields, since it is a grounded safety conclusion.
    risk = _risk_text(cwe, sink, obs) if resolution == "underdetermined" else ""
    return Verdict(
        sink_id=sink["sink_id"], is_vuln=is_vuln,
        # The CWE class is a deterministic label of **which risk category the
        # suspicious point belongs to**, independent of whether it is judged
        # positive.  Clearing it under underdetermined would leave an
        # underdetermined point without a residual risk category.
        cwe_id=cwe if (is_vuln or resolution == "underdetermined") else "",
        location=loc, confidence=conf, basis=basis,
        resolution=resolution,
        fix_suggestion=fix if is_vuln else "",
        missing_guard=duty if is_vuln and not is_contract else "",
        violated_contract=duty if is_vuln and is_contract else "",
        **({"risk": risk, "unresolved_reason": why} if resolution == "underdetermined" else {}),
        **({"contract_witness": sink["contract_witness"]}
           if is_vuln and is_contract and sink.get("contract_witness") else {}),
        origin="arbiter",
        **({"obligations": final_obs} if final_obs else {}),
    )


def _risk_text(cwe: str, sink: dict, obs: list) -> str:
    """Human-readable risk description for an `underdetermined` point: which risk
    category, plus why a single file cannot confirm it.

    The wording deliberately separates "cannot confirm it is a vulnerability"
    from "proved safe" -- the former requires manual review or cross-file
    evidence.
    """
    pend = [o for o in obs
            if o.get("critical") and o.get("status") not in ("SATISFIED",)]
    duties = ", ".join(
        str(o.get("predicate") or o.get("id") or "") for o in pend[:2])
    walk = str(sink.get("walk_kind") or "")
    shape = {
        "incr": "the pointer keeps advancing inside the loop (*p++), while the "
                "loop bound is unrelated to the readable extent of the buffer",
        "helper": "the pointer is advanced and written back through a helper "
                  "(p=Helper(...,p,...)), and the step is not visible in this file",
        "addassign": "the pointer accumulates by a fixed step inside the loop (p+=k)",
    }.get(walk, "")
    parts = [f"suspected {cwe} risk" if cwe else "suspected security risk"]
    if shape:
        parts.append(shape)
    if duties:
        parts.append(f"obligations to be proved are not proved: {duties}")
    parts.append("because information outside this file is missing (the buffer "
                 "size passed by the caller / the semantics of the callee), it "
                 "can neither be confirmed as a vulnerability nor proved safe "
                 "-- manual review or cross-file evidence is required")
    return "; ".join(parts)
