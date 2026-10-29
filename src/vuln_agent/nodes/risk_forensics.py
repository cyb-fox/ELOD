"""risk_forensics (view B, LLM + grep tool loop, aggregated per batch).

One LLM call per batch for forensics (each batch holds <=SINK_BATCH_MAX sinks
and may span several small functions): only the original-code risk window,
machine CFG facts and search results are fed in, and the model judges point by
point whether detection_signals hit and whether a dominating protection already
covers it.  The secure formulation produced by view A does not enter this node,
which avoids reading the expected repair as original code.  Batching amortizes
the fixed SYSTEM+CWE schema tax across sinks instead of charging it per
function.  The LLM can request `search_context` via need_more (merged across
risk points); the round count and variable count are bounded by the two hard
limits of ContextSearcher.

The `false_defense_traps` channel is absent: it handed the model a list of
"protections that look right but are insufficient", whose practical effect was
to let genuine existing protections be read as false defenses -- the opposite of
the direction of I7, where equivalent protection is induced and judged by the
model -- and the `is_false_defense` it produced was one of the arbiter's
singleton corroboration items, i.e. an inducing prompt raising confidence.
Judging whether a protection suffices therefore keeps a single path: candidate
facts + checker re-verification.

`screen` semantics: it is only a **human-readable forensic summary label**, no
longer an early-exit switch and without veto power.  Adjudication is fully
deterministic, aggregated at arbiter from the evidence-bearing obligation
snapshots written by obligation_checker (three-valued resolution vuln /
safe_proved / underdetermined).  The reason: the LLM's exclude reasons are
often of the form "the caller is not visible inside the window", which grounds
**underdetermined**, not safety.

Conservative fallback for missed points: if the findings returned by the model
omit a sink_id, a `keep` item pending review is appended for it, so batch-level
aggregation never drops a report silently.
"""
from __future__ import annotations

from ..llm import prompts as P
from ..state import ForensicItem, VulnState
from ..tools import micro_check as mc
from ..tools import obligation_checker as oc
from ..tools import obligations as ob
from ..tools import trajectory as tj
from ..tools.context_search import ContextSearcher
from ._common import make_llm, parallel_map, risk_window, pack_batches, cfg_for_line
from .secure_diff import _aggregate


def _cfg_brief(state: VulnState, line: int) -> str:
    """Branch structure of the function owning the risk line (so the model can
    cite node ids when submitting a witness).  Empty when no graph is built."""
    g = cfg_for_line(state, int(line or 0))
    return g.brief(int(line or 0)) if g is not None else ""


def _machine_guard_facts(state: VulnState, line: int) -> str:
    """State only the local structure that the CFG has verified; never infer the
    failure semantics of unknown calls."""
    g = cfg_for_line(state, int(line or 0))
    if g is None:
        return ("CFG unavailable: no local machine guard facts could be extracted"
                " (this is not a safety conclusion).")

    facts: list[str] = []
    for guard in g.dominating_guards(int(line or 0)):
        if guard.get("must_take") is None:
            continue
        side = "true" if guard.get("must_take") else "false"
        facts.append(
            "branch {id}@{line} always takes the {side} side: {text}".format(
                id=guard["id"], line=guard["line"], side=side,
                text=guard["text"],
            )
        )
    for stmt in g.dominating_statements(int(line or 0)):
        facts.append("dominating data-relevant non-branch statement @{0}: {1}".format(
            stmt.line, stmt.text,
        ))
    if not facts:
        return ("No local machine guard facts related to this sink could be"
                " extracted (this is not a safety conclusion).")
    return "\n".join(facts)


def _verified_rationale(f: dict, window: str) -> str:
    """Keep only the brief model reasoning that anchors verbatim into the same
    risk window; for human reading, not part of adjudication."""
    rationale = str(f.get("rationale") or "").strip()
    anchors = f.get("rationale_anchors") or []
    if not rationale or len(rationale) > 400 or not isinstance(anchors, list):
        return ""

    window_lines: dict[int, str] = {}
    for raw in window.splitlines():
        prefix, sep, text = raw.partition("|")
        if sep and prefix.strip().isdigit():
            window_lines[int(prefix.strip())] = text.strip()

    verified: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for anchor in anchors:
        if not isinstance(anchor, dict):
            return ""
        try:
            line = int(anchor.get("line"))
        except (TypeError, ValueError):
            return ""
        text = str(anchor.get("text") or "").strip()
        source_line = window_lines.get(line)
        if not source_line or not oc.verify_local(text, source_line):
            return ""
        key = (line, " ".join(text.split()))
        if key not in seen:
            seen.add(key)
            verified.append((line, text))
    if not verified:
        return ""
    refs = "; ".join(f"{line}: {text}" for line, text in verified)
    return f"{rationale} (source anchors: {refs})"


def risk_forensics(state: VulnState) -> dict:
    llm = make_llm(state, "risk_forensics")
    knowledge = state.get("knowledge", {})
    scope = state.get("target_path", "<file>")
    batches = pack_batches(state.get("sinks", []))

    results = parallel_map(
        lambda batch: _forensic_batch(llm, state, batch, knowledge, scope),
        batches,
    )
    forensics: list[ForensicItem] = []
    for group_items in results:
        forensics.extend(group_items)
    degraded = list(state.get("degraded") or [])
    # Points not covered by forensics = the LLM output of that batch was
    # unavailable or omitted the sink_id.  The preset hypotheses of such points
    # are voided to UNDERDETERMINED (I7), so "underdetermined in this run" and
    # "vuln in an earlier run" may differ by exactly this degradation -- record
    # it.
    uncovered = sum(
        1 for f in forensics
        if not f.get("candidate_facts")
        and "conservatively kept pending review" in str(f.get("evidence", "")))
    if uncovered:
        degraded.append(
            f"risk_forensics: {uncovered} point(s) not covered by forensics; "
            "preset hypotheses voided to UNDERDETERMINED")
    return {"forensics": forensics, "degraded": degraded}


def _forensic_batch(llm, state, sinks, knowledge, scope) -> list[ForensicItem]:
    # Whole-family knowledge injection: det/negs are aggregated over all loaded
    # CWE families; recall first, no narrowing.
    det = _aggregate(knowledge, "detection_signals")
    # basis for "already covered by a guard"
    negs = _aggregate(knowledge, "negative_signals")
    searcher = ContextSearcher(
        state["source_text"], scope_label=scope,
        project_index=state.get("project_index"),
        current_path=state.get("target_path", ""),
    )
    # The risk window of every sink in this batch is packed into the brief (with
    # function names, so cross-function cases stay distinguishable) for the model
    # to investigate point by point.  cfg_brief is a **deterministic structure**
    # (branch node ids + the verbatim condition + whether it truly dominates the
    # sink), which lets the model submit witnesses by node id instead of the most
    # fragile link of all, copying source code verbatim; it costs zero token.
    sinks_brief = [
        {
            "sink_id": s["sink_id"], "func_name": s["func_name"], "line": s["line"],
            "tainted_tokens": s["tainted_tokens"],
            "change_kind": s.get("change_kind", "guard"),
            "contract_witness": s.get("contract_witness"),
            "snippet": risk_window(state, s["line"]),
            "cfg_brief": _cfg_brief(state, s["line"]),
            "machine_guard_facts": _machine_guard_facts(state, s["line"]),
        }
        for s in sinks
    ]
    # Contract candidates are searched proactively with the tokens given at
    # generation time; retrieving original code is independent evidence, while
    # the witness itself remains a hypothesis pending verification.
    contract_tokens: list[str] = []
    for s in sinks_brief:
        if s.get("change_kind") != "contract":
            continue
        for token in s.get("tainted_tokens") or []:
            if token not in contract_tokens:
                contract_tokens.append(token)
    search_summary = searcher.search(contract_tokens) if contract_tokens else ""

    findings: dict = {}
    for _ in range(searcher.rounds_left + 1):
        data = llm.complete_json(
            P.FORENSICS_SYSTEM,
            P.forensics_user(
                sinks_brief, det, negs, search_summary, searcher.rounds_left,
                family_context=state.get("family_context") or [],
            ),
        )
        if not isinstance(data, dict):
            break
        for f in (data.get("findings") or []):
            if isinstance(f, dict) and f.get("sink_id"):
                findings[str(f["sink_id"])] = f
        need = data.get("need_more")
        if isinstance(need, dict) and searcher.can_search():
            tokens = [t for t in (need.get("tokens") or []) if isinstance(t, str)]
            if tokens:
                more = searcher.search(tokens)
                search_summary = "\n".join(x for x in (search_summary, more) if x)
                continue
        break

    # One ForensicItem per sink.  Evidence boundary: the LLM only proposes
    # candidate_facts; every obligation advance is written after deterministic
    # re-verification by obligation_checker (this node no longer assembles
    # state_delta itself).  The consumer window used for re-verification is the
    # same line-numbered risk window fed to the LLM (sinks_brief.snippet) -- the
    # window text the model cites and the text the checker re-verifies must be
    # the same one, and the line numbers are what make a dominance judgement
    # possible.
    windows = {b["sink_id"]: b["snippet"] for b in sinks_brief}
    out: list[ForensicItem] = []
    for s in sinks:
        window = windows.get(s["sink_id"]) or s["snippet"]
        f = findings.get(s["sink_id"])
        sink_type = s.get("sink_type", "")
        bucket = ob.bucket_key("", sink_type)
        if isinstance(f, dict):
            is_contract = s.get("change_kind") == "contract"
            citations = [
                x for x in (f.get("contract_evidence") or []) if isinstance(x, str)
            ]
            contract_verified = (
                is_contract
                and bool(f.get("contract_verified"))
                and oc.verify_cross_file(citations, window, search_summary)
            )
            # screen is only a human-readable forensic summary and carries no
            # veto power (see the arbiter docstring).  A contract candidate that
            # produced no independent citation also stays keep -- missing
            # cross-file information can only lead to "undetermined", never to
            # "proved safe"; whether it is judged positive is still decided by
            # the obligation snapshot of obligation_checker.
            screen = "exclude" if str(f.get("screen", "keep")) == "exclude" else "keep"
            signals = [x for x in (f.get("hit_signals") or []) if isinstance(x, str)]
            if contract_verified and "contract_violation" not in signals:
                signals.append("contract_violation")

            # -- evidence boundary adjudication: candidate facts go to
            # obligation_checker for re-verification, the only place where an
            # obligation advances -- secure_diff only stores the candidate
            # direction; here it is handed to the same checker channel and is
            # never read directly as the VIOLATED status of an obligation.
            raw_facts = oc.seeded_candidate_facts(f.get("obligations") or [])
            raw_facts.extend(c for c in (f.get("candidate_facts") or []) if isinstance(c, dict))
            # If both-end citations of a contract item already passed
            # verify_cross_file, append one attacker_control candidate so that a
            # response that carried no candidate_facts can still advance the
            # obligation.
            if contract_verified and not any(
                str(c.get("obligation_id")) == "attacker_control" for c in raw_facts
            ):
                raw_facts.append({
                    "obligation_id": "attacker_control", "direction": "VIOLATED",
                    "anchors": citations, "cross_file": True,
                })
            # (5) Semantic micro-check: candidate facts produced by the
            # deterministic proposer merge with the LLM proposals on equal terms
            # and go together to the single discharger for re-verification (size
            # arithmetic overflow + no guard -> target obligation candidate
            # VIOLATED).
            raw_facts.extend(mc.propose_facts(s, window))
            # The deterministic structure needed for witnesses and true
            # dominance: the CFG of the function owning this sink (zero token,
            # cached per process).
            g = cfg_for_line(state, int(s.get("line") or 0))
            proved_obs, verified_refs = oc.discharge(
                s, raw_facts,
                consumer_source=window, searched_source=search_summary, cfg=g,
            )
            # Deterministic anchors for the advanced obligations are added: the
            # forensically examined risk line itself (func:line, the same format
            # as sink_id).
            refs = list(verified_refs)
            advanced = oc.critical_violated(proved_obs)
            if advanced and s["sink_id"] not in refs:
                refs.append(s["sink_id"])

            # Side-channel trace: forensics is the evidence-retrieval step for
            # obligations.  action is tagged by candidate type (contract =
            # variable-origin lookup; guard = local guard inspection).
            # state_delta reflects the obligations obligation_checker actually
            # advanced.
            action = ("resolve_variable_origin" if is_contract
                      else "inspect_local_guard")
            delta = {
                o["id"]: {"from": "UNDERDETERMINED", "to": o["status"]}
                for o in advanced
            }
            tj.record_step(
                node="risk_forensics", action=action,
                sink_id=s["sink_id"], bucket=bucket,
                inputs={"tainted_tokens": s.get("tainted_tokens", []),
                        "change_kind": s.get("change_kind", "guard")},
                observation={"screen": screen, "hit_signals": signals},
                state_delta=delta,
                evidence_refs=refs,
            )
            out.append(ForensicItem(
                sink_id=s["sink_id"],
                hit_signals=signals,
                # Show reasoning only when all of it anchors into the risk
                # window; otherwise fall back to the citations the checker has
                # re-verified.
                evidence=(
                    _verified_rationale(f, window)
                    or "; ".join(refs)
                    or "no reviewable forensic anchor was provided"
                ),
                screen=screen,
                candidate_facts=raw_facts,
                obligations=proved_obs,
                **({
                    "contract_verified": contract_verified,
                    "contract_evidence": citations,
                } if is_contract else {}),
            ))
        else:
            is_contract = s.get("change_kind") == "contract"
            # This point was not covered by forensics in this round: the preset
            # VIOLATED of secure_diff never passed any re-verification and
            # cannot enter adjudication as it stands (I7: a formal difference is
            # not a missing protection).  Downgraded to UNDERDETERMINED for the
            # record -- undetermined rather than proved safe, and output's
            # underdetermined_risks still surfaces the risk CWE.
            out.append(ForensicItem(
                sink_id=s["sink_id"], hit_signals=[],
                evidence=("(contract candidate produced no independent evidence,"
                          " conservatively kept pending review)" if is_contract
                          else "(batch-level forensics did not cover this point,"
                          " conservatively kept pending review)"),
                screen="keep",
                candidate_facts=[],
                obligations=oc.void_unaudited_presets(
                    [dict(o) for o in (s.get("obligations") or
                                       ob.instantiate(sink_type))],
                    "batch-level forensics did not cover this point; the preset "
                    "hypothesis of secure_diff was not re-verified (I7)",
                ),
                **({"contract_verified": False, "contract_evidence": []}
                   if is_contract else {}),
            ))
    return out
