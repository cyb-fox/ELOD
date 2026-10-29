"""secure_diff (view-A candidate gate: secure-version contrast + deterministic
diff; the LLM generates, the code adjudicates the difference).

Both modes share this implementation (the locate_sinks of the file mode and the
secure_baseline chain are both folded in here).  Core mechanism: **the reference
version is the agent-generated secure version**, but a "difference" is decided
deterministically by code and the model's self-report is never taken on trust:

  1. For every place the model believes is insecure, it reports a **line number
     plus the locating snippet of that line** (original) together with a secure
     implementation (secure).  `original` serves only to pin the model's chosen
     position onto the real source: once anchoring succeeds, the code retrieves
     the ground truth for that line from the source file and **overwrites**
     `original` with it, so a mis-transcribed or paraphrased line no longer
     pollutes downstream comparison.  Not restricted to the loaded CWE families:
     suspicions such as uninitialized state, object lifetime, cleanup paths,
     races and contract mismatches are all reported (creativity belongs to the
     hypothesis stage, discipline to the proof stage -- every suspicion has to
     pass the same deterministic evidence chain).
  2. [Deterministic diff gate -- decided by code, not by the model]  Given
     (line, original, secure), the code performs two anchoring steps:
       (i) `original` must anchor-hit inside the **real source** in the
           neighbourhood of that line (whole-line equality after whitespace
           normalization, or containment of a sufficiently long snippet).  If the
           line the model points at does not contain that code at all, the
           candidate is discarded (this forecloses hallucinating a vulnerability
           on a line that does not exist in the source).  After a hit, `original`
           is always replaced by the real source line.
       (ii) Only if the normalized `secure` differs from the anchored real source
            line is it a real difference; if `secure` is verbatim identical to the
            real original (the model claimed a change but copied the same line),
            the candidate is discarded.  This is exactly the behavior wanted when
            the input file is already a secure version: secure ~ original -> empty
            diff -> no candidates -> no over-reporting.
  This replaces the discriminative signal "ask the model whether there is a
  discrepancy" with a deterministic text diff between the **real original
  retrieved by code** and the model's secure version: the original side leaves
  the model no room to fabricate, so a byte-identical secure line yields no
  candidate whatever threat narrative the model attaches to it.

[Filtering gate -- route by difference axis after anchoring (suppresses
over-reporting up front, zero token)]
  (iii) guard-class changes go through the counterfactual guard gate
        (_guard_gate): the same guard criterion is re-checked on both sides --
        `original` lacks the construct (VIOLATED side) AND `secure` has the shape
        of a real protection acting on the same object (SATISFIED side).  "Secure"
        is thus defined as a **verifiable repair relative to the target
        obligation**, not something the model claims out loud.
  (iv) contract-class changes go through the contract evidence gate
       (_contract_gate): removing an error return, replacing a state criterion or
       adding a semantic parameter is allowed, but a complete behavioral
       counterexample, producer/consumer citations and cross-project-searchable
       evidence_tokens are required.
  (iv') type_fix-class changes go through the type gate (_type_fix_gate): the
        repair takes the form of a **width / signedness change** rather than a new
        branch, so it can never pass the guard gate.  The criterion is bilateral
        as well: the type name is visible on both sides, and `secure` really
        widens the type (by language-level width rank) or corrects signedness,
        acting on the same object.  Alias-only renames and edits to unrelated
        objects fail.  Truncation / signedness mismatch is one of the most common
        repair shapes in patches (`int` -> `int64_t`) and must not be swallowed by
        the guard gate.
  (iv") api_swap-class changes go through the API-swap gate (_api_swap_gate): the
        repair replaces an aborting / unvalidated call with a safe API that returns
        a status code (or wraps the original call with a status check).  Criterion:
        `secure` introduces a callee name absent from `original` AND (the original
        callee is gone OR a check shape was added) AND it acts on the same data
        object.  Merely wrapping the same call in another layer, or swapping in an
        unrelated call, fails.
  (v) all four classes share the three threat questions (_threat_gate, positive
      criteria): `source` and `reachable` must name at least one identifier that
      really occurs in the source (grep-verifiable), and `harm` must be non-empty
      (harm is a semantic description and is not required to cite identifiers).
      All three answers are passed through to forensics/arbiter.

[Cross-line defects: anchors expansion]  One defect may span several lines (a
narrow type at the declaration plus an accumulation at the use site), so the
prompt may attach {line, original, secure} records in `anchors`.  At parse time
each anchor is **expanded into an independent change of the same family**
(inheriting cwe_id / change_kind / the three threat answers / the contract
witness) and passes the same gate set one by one -- expansion rather than merging,
because downstream forensics and obligation instantiation both work per line, and
merging would leave the auxiliary lines without their own risk window.

Outputs:
- sinks: real difference candidates that passed the gates (source="secure_diff")
  UNION the rule_locate_sinks rule floor (recall guard); if both are empty, a
  bounded mechanism hypothesis anchored to the source produces source="triage"
  points pending verification.
- baseline: BaselineItem back-filled only for functions **that have a real
  difference** (already_safe=False, secure_lines = the concatenation of the
  difference points of that function), serving the view-A references and
  missing_guard identification of the downstream risk_forensics/arbiter;
  functions hit only by the rule floor produce no baseline, which leaves the
  arbiter to judge them without an early exit and lets its missing_guard hard
  constraint do the filtering.

Candidates are handed to risk_forensics, which traces context starting from the
difference point (reverse grep over tainted tokens, checking whether guards cover
it), and then to arbiter for the final verdict.  On budget exhaustion
budget_guard short-circuits.
"""
from __future__ import annotations

import re

from .. import config
from ..llm import prompts as P
from ..state import BaselineItem, ContractWitness, Sink, VulnState
from ..tools import obligation_checker as oc
from ..tools import obligations as ob
from ..tools import sink_locator as sl
from ..tools import trajectory as tj
from ._common import bounded_source, make_llm

# Deterministic diff anchor radius: model-reported line numbers are often off by
# a few lines (especially after structured compression), so the real hit for
# `original` is searched this many lines above and below the given line; on a hit
# the line number is corrected to the real line.
_ANCHOR_RADIUS = 4
# Minimum quote length required for the relaxed "snippet containment" match: short
# snippets ("len", "if (") would almost surely hit some line in the
# neighbourhood, which degrades anchoring into noise.  Below this length only
# exact whole-line matching is allowed.
_MIN_QUOTE_CHARS = 8

# Gate (iii) guard-construct regex: the deterministic criterion by which `secure`
# adds a "defensive construct" relative to `original` -- if / validation /
# comparison / error return / bound / length / locking / escaping and the like.
# Absent from the original and present in the secure version is what counts as
# real hardening.
_GUARD_TOKENS = re.compile(
    r"\b(if|else|return|throw|goto|assert|continue|break|"
    r"sizeof|strn?len|snprintf|strncpy|strncat|"
    r"lock|guard|mutex|escape|sanitize|valid|check|bound|clamp|min|max)\b"
    r"|[<>]=?|==|!=|&&|\|\||\?",
)
# Bounded / locking / escaping safe APIs: not shaped like an `if` check
# themselves, yet equally valid repair forms for an obligation (e.g.
# strcpy->strncpy).  Either such an API or a guard shape (oc.looks_like_guard)
# satisfies the SATISFIED side.
_SAFE_BOUNDED_API = {
    "sizeof", "strlen", "strnlen", "snprintf", "strncpy", "strncat",
    "lock", "guard", "mutex", "escape", "sanitize", "valid",
    "check", "bound", "clamp", "min", "max",
}

# Gate (iv') type_fix: integer type name -> language-level width rank.
# **Language-level** rather than project-level, because the width relations of
# C/C++ standard types are a syntactic fact independent of any framework or
# library (the same property as _STOP_IDENTS collecting C keywords).  It has
# exactly one purpose: deciding whether `secure` really **widens or corrects
# signedness** relative to `original` instead of merely swapping in an alias.
_INT_TYPE_RE = re.compile(
    r"\b(u?int(?:8|16|32|64)_t|s?size_t|ssize_t|ptrdiff_t|u?intptr_t|"
    r"unsigned\s+(?:long\s+long|long|int|short|char)|"
    r"signed\s+(?:long\s+long|long|int|short|char)|"
    r"long\s+long|long|int|short|char)\b",
)
_TYPE_WIDTH = {
    "char": 8, "short": 16, "int": 32, "long": 64, "long long": 64,
    "int8_t": 8, "int16_t": 16, "int32_t": 32, "int64_t": 64,
    "uint8_t": 8, "uint16_t": 16, "uint32_t": 32, "uint64_t": 64,
    "size_t": 64, "ssize_t": 64, "ptrdiff_t": 64,
    "intptr_t": 64, "uintptr_t": 64,
}
# Callee name (for the API-swap gate): an identifier immediately followed by a
# left parenthesis.
_CALLEE_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _aggregate(knowledge: dict, field: str) -> list:
    """Aggregate one field over all loaded CWEs (deduplicated)."""
    out: list = []
    for d in knowledge.values():
        v = d.get(field)
        if isinstance(v, list):
            for x in v:
                if x not in out:
                    out.append(x)
        elif v is not None:
            out.append(v)
    return out


def _safe_examples(knowledge: dict) -> dict:
    out = {}
    for cid, d in knowledge.items():
        ex = (d.get("minimal_examples") or {}).get("safe")
        if ex:
            out[cid] = ex
    return out


def _func_of_line(functions: list, line: int) -> str:
    """Which function interval the line number falls into -> function name; not
    inside any function body (a top-level declaration) -> "<toplevel>"."""
    for f in functions:
        if f["line_start"] <= line <= f["line_end"]:
            return f["func_name"]
    return "<toplevel>"


def _norm(s: str) -> str:
    """Normalization: strip leading/trailing whitespace + collapse internal
    whitespace runs to a single space.  Used for text-diff anchoring comparisons;
    it tolerates indentation/formatting differences while preserving genuine
    lexical changes (an added if / validation / boundary condition, etc.)."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _anchor_change(ch: dict, src_lines: list[str]) -> dict | None:
    """Deterministic diff gate: the model-reported (line, original, secure) is
    checked for truthfulness by code, and **the original side is retrieved by code
    from the source file** -- the model is only responsible for "which line",
    never for "what the original looks like".

    Returns the change carrying the anchored line plus the real `original`
    retrieved by code (passes the gate), or None (does not pass):
      (i) Locate: first try exact matching (whole-line equality after
          normalization), then substring containment from nearest to farthest
          (model-reported line numbers are often off by a few lines, especially
          after structured compression).  Failing both means the code does not
          exist on the line the model pointed at -> discard (this forecloses
          hallucinating a vulnerability on a line that does not exist).  Snippets
          that are too short are not allowed to use containment matching, which
          avoids noisy hits on fragments such as "len" or "if".
      (ii) Ground truth: `original` is always **overwritten** with the real source
          line that was hit.  A mis-transcribed, truncated or paraphrased line no
          longer affects downstream work -- the counterfactual guard gate compares
          the "real original vs the model's secure version", not two model
          self-reports.
      (iii) Real difference: the normalized `secure` must differ from the real
          original line that was hit.  Equal means secure ~ original, i.e. no
          difference -> discard (patch files land on exactly this branch).
    """
    line = ch["line"]
    quote = _norm(ch.get("original", ""))
    norm_secure = _norm(ch.get("secure", ""))
    if not quote:
        return None
    n = len(src_lines)
    # Candidate lines ordered by distance (line, line-1, line+1, line-2, ...), so
    # a correction takes the nearest hit.
    cands = [line]
    for d in range(1, _ANCHOR_RADIUS + 1):
        cands.extend((line - d, line + d))
    cands = [c for c in cands if 1 <= c <= n]

    hit = next((c for c in cands if _norm(src_lines[c - 1]) == quote), None)
    if hit is None and len(quote) >= _MIN_QUOTE_CHARS:
        # Relax once: the model may have copied only the key snippet (the snippet
        # must be long enough, otherwise it cannot locate anything).
        hit = next((c for c in cands if quote in _norm(src_lines[c - 1])), None)
    if hit is None:
        return None  # cannot anchor onto real source -> model fabrication, discard
    real = src_lines[hit - 1].strip()
    # (iii) Real-difference test: the normalized secure version must differ from
    # the real source line that was hit.
    if not norm_secure or norm_secure == _norm(real):
        # secure ~ original -> no difference -> discard (patch files land here)
        return None
    out = dict(ch)
    out["line"] = hit
    # The original side takes the ground truth code read from the source file; the
    # model's retelling is not trusted.
    out["original"] = real
    return out


def _guard_tokens(text: str) -> set[str]:
    """Extract guard-construct tokens (taking the whole matched text, which keeps
    operators such as <=/==/&& distinguishable -- findall on a regex containing
    groups returns only the group contents and would collapse every operator into
    an empty string)."""
    return {m.group(0) for m in _GUARD_TOKENS.finditer(text)}


def _guard_gate(ch: dict) -> bool:
    """Gate (iii), the counterfactual guard gate (deterministic, bilateral
    re-verification) -- "original violates the target obligation AND secure
    satisfies the target obligation", with the same criterion checked once on
    each of the two texts:

      VIOLATED side: relative to `original`, `secure` must **add** a guard-construct
        token (if / comparison / error return / length / bound / locking /
        escaping, ...) -- the original has no such construct, i.e. the same
        criterion is unprovable on the original;
      SATISFIED side: `secure` must really have the shape of a protection -- a guard
        shape (if / comparison / assertion / bound macro; the same criterion as
        `looks_like_guard`, which obligation_checker uses to re-check guard
        anchors), or the addition of a bounded / locking / escaping safe API (a
        valid repair shape such as strcpy->strncpy);
      same-object constraint: `secure` must touch an identifier of `original` (the
        guard acts on the same object, which excludes an equivalent rewrite that
        introduces a brand-new object out of thin air).

    Renaming alone, reordering, or an equivalent rewrite fails the VIOLATED side;
    claiming hardening without any check shape or bounded API fails the SATISFIED
    side.  The "security" of a secure version therefore gains a checkable
    definition: a verifiable repair relative to the target obligation of that
    place, rather than something the model claims out loud."""
    orig = _norm(ch.get("original", ""))
    sec = _norm(ch.get("secure", ""))
    added = _guard_tokens(sec.lower()) - _guard_tokens(orig.lower())
    if not added:
        return False
    if not (oc.looks_like_guard(sec) or (added & _SAFE_BOUNDED_API)):
        return False
    orig_ids = oc.idents(orig)
    return (not orig_ids) or bool(orig_ids & oc.idents(sec))


def _threat_gate(ch: dict, src_idents: set[str]) -> bool:
    """Gate (v), the positive criterion of the three threat questions: `source`
    and `reachable` must name at least one identifier that really occurs in the
    source (grep-verifiable), and `harm` merely has to be non-empty (harm is a
    semantic description and is not required to cite identifiers).

    Positive criteria are used instead of a negative word list of "vague terms": a
    negative word list would falsely reject entirely specific descriptions such as
    "the attacker may control len through recv", which is a textbook way of
    choking the model's creativity; a positive criterion only looks at whether
    real code entities are named and does not penalize wording."""
    if not _norm(ch.get("threat_harm", "")):
        return False
    for k in ("threat_source", "threat_reachable"):
        v = ch.get(k, "")
        if not _norm(v):
            return False
        if not (oc.idents(v) & src_idents):
            return False
    return True


def _types(text: str) -> list[str]:
    """Extract integer type names (internal whitespace normalized),
    order-preserving and deduplicated."""
    out: list[str] = []
    for m in _INT_TYPE_RE.finditer(text or ""):
        t = re.sub(r"\s+", " ", m.group(1)).strip()
        if t not in out:
            out.append(t)
    return out


def _width(t: str) -> int:
    """Language-level width of a type (an unknown type gets 0, i.e. "widening
    cannot be established", which only lets fewer candidates through)."""
    return _TYPE_WIDTH.get(re.sub(r"^(un)?signed\s+", "", t).strip(), 0)


def _is_unsigned(t: str) -> bool:
    return t.startswith("unsigned") or t.startswith("u") and t.endswith("_t") \
        or t in ("size_t", "uintptr_t")


def _type_fix_gate(ch: dict) -> bool:
    """Gate (iv'), the type gate (deterministic, bilateral re-verification) --
    a truncation / signedness-mismatch repair adds no branch, so it always fails
    the guard gate.

      VIOLATED side: an integer type name must really occur in `original`
        (otherwise this line has no type to blame);
      SATISFIED side: the type in `secure` must **really be wider** (a larger
        language-level width rank) or **must have corrected signedness**
        (unsigned<->signed flip) relative to `original` -- an alias-only change
        (`int` -> `int32_t`) or a rename without a width change does not count
        as a repair;
      same-object constraint: `secure` must touch an identifier of `original` (it
        is the type of the same variable that is changed, which excludes an
        equivalent rewrite that introduces a new object out of thin air).

    The width table contains only C/C++ standard types (a syntactic fact, not a
    per-project or per-framework name table), which matches the principle that
    "the deterministic side only performs structural, name-table-free checking" --
    it verifies width relations, not "which API is safe"."""
    orig = _norm(ch.get("original", ""))
    sec = _norm(ch.get("secure", ""))
    ot, st = _types(orig), _types(sec)
    if not ot or not st:
        return False
    widened = max((_width(t) for t in st), default=0) > \
        max((_width(t) for t in ot), default=0)
    sign_fixed = any(_is_unsigned(a) != _is_unsigned(b) for a in ot for b in st)
    if not (widened or sign_fixed):
        return False
    orig_ids = oc.idents(orig)
    return (not orig_ids) or bool(orig_ids & oc.idents(sec))


def _api_swap_gate(ch: dict) -> bool:
    """Gate (iv"), the API-swap gate (deterministic, bilateral re-verification) --
    the repair form is to switch to a safe API that returns a status code (or to
    add a status check to the original call), which again need not add a branch.

      VIOLATED side: `original` must really be a call (there is a callee name to
        swap);
      SATISFIED side: `secure` introduces a callee name absent from `original`,
        and **either** the original callee is gone (the original call
        disappeared), **or** a check shape was added (the original call is kept
        but its return status is now caught);
      same-object constraint: `secure` must touch an identifier of `original` (it
        is the handling of the same data that is swapped).

    Merely wrapping the same call in one more layer, or swapping in a call
    unrelated to the original data, fails.  Every criterion is structural: no
    table of "which APIs are safe" is held -- safety is induced by the model, and
    this gate only checks that a change of form really happened."""
    orig = _norm(ch.get("original", ""))
    sec = _norm(ch.get("secure", ""))
    o_calls = set(_CALLEE_RE.findall(orig))
    s_calls = set(_CALLEE_RE.findall(sec))
    if not o_calls or not (s_calls - o_calls):
        return False
    if not ((o_calls - s_calls) or oc.looks_like_guard(sec)):
        return False
    orig_ids = oc.idents(orig)
    return (not orig_ids) or bool(orig_ids & oc.idents(sec))


def _contract_gate(ch: dict) -> bool:
    """Gate dedicated to the contract class: it verifies only structure and
    verifiable evidence, and does not require the repair form to add a guard."""
    witness = ch.get("contract_witness")
    if not isinstance(witness, dict):
        return False
    required = (
        "precondition", "expected_behavior", "actual_behavior",
        "security_consequence", "counterexample",
        "producer_evidence", "consumer_evidence",
    )
    if any(not _norm(witness.get(k, "")) for k in required):
        return False
    if _norm(witness["expected_behavior"]) == _norm(witness["actual_behavior"]):
        return False
    tokens = witness.get("evidence_tokens")
    return bool(isinstance(tokens, list) and any(_norm(t) for t in tokens))


def _change_gate(ch: dict) -> bool:
    """Route by difference axis; the shared real-difference and threat gates are
    executed at the caller.

    An unknown change_kind always goes to the guard gate (the strictest one)
    rather than being discarded outright: a new term can only be rejected because
    it fails the guard gate, and will never be silently missed merely because the
    parse layer does not recognize it."""
    kind = ch.get("change_kind")
    if kind == "contract":
        return _contract_gate(ch)
    if kind == "type_fix":
        return _type_fix_gate(ch)
    if kind == "api_swap":
        return _api_swap_gate(ch)
    return _guard_gate(ch)


def _rule_and_provisional_sinks(state: VulnState, src_lines: list[str]) -> list[Sink]:
    """Rule floor; when entirely empty, the bounded mechanism-hypothesis bridge
    runs.  No VIOLATED is preset."""
    source = state["source_text"]
    functions = state.get("functions", []) or []
    rule_sinks: list[Sink] = []
    for f in functions:
        rule_sinks.extend(sl.rule_locate_sinks(source, f))
    all_sinks = sl.merge_sinks(rule_sinks, [])
    if not all_sinks:
        all_sinks = sl.merge_sinks([], _provisional_sinks(state, src_lines))
    return all_sinks


def secure_diff(state: VulnState) -> dict:
    source = state["source_text"]
    functions = state.get("functions", []) or []
    src_lines = source.split("\n")
    if not config.SECURE_REFERENCE_ENABLED:
        degraded = list(state.get("degraded") or [])
        degraded.append("secure_diff: counterfactual secure version gate disabled, "
                        "no violation preset")
        return {"sinks": _rule_and_provisional_sinks(state, src_lines),
                "baseline": [], "degraded": degraded,
                "_secure_diff_outcome": "disabled"}

    llm = make_llm(state, "secure_diff")
    knowledge = state.get("knowledge", {})

    # Whole-file feeding: huge files go through the structured digest
    # (bounded_source); line numbers stay absolute and keep pointing back into
    # the original text.
    numbered = bounded_source(state)
    cwe_ids = list(knowledge.keys())
    negs = _aggregate(knowledge, "negative_signals")
    safes = _safe_examples(knowledge)

    data = llm.complete_json(
        P.SECURE_DIFF_SYSTEM,
        P.secure_diff_user(numbered, cwe_ids, negs, safes,
                           suspicions=state.get("_triage_hypotheses") or [],
                           family_context=state.get("family_context") or []),
        max_tokens=config.GEN_MAX_TOKENS,
        # On truncation complete_json raises the cap; for dirty JSON that is not
        # truncated, one schema correction is allowed.  Both recovery paths
        # trigger only on the failure path, so a normal successful request gains
        # no extra call.
        retry=True,
    )
    degraded = list(state.get("degraded") or [])
    if data is None:
        degraded.append("secure_diff: LLM output unavailable, no diff candidate "
                        "this run (only the rule floor remains)")
    changes = _parse_changes(data, len(src_lines))
    _record_defenses(_parse_defenses(data), src_lines)

    # Deterministic diff gate: the model-reported (original, secure) is not taken
    # on trust as a "difference"; code decides --
    # (i) original must anchor-hit the real source in the neighbourhood of the
    #     line (otherwise the model hallucinated a vulnerability on a line that
    #     does not exist -> discard);
    # (ii) the normalized secure must differ from the real source line that was
    #     hit (otherwise secure ~ original = no difference -> discard; patch files
    #     land here).
    changes = [ch for ch in (_anchor_change(ch, src_lines) for ch in changes) if ch]

    # The filtering gates route by difference axis: the guard class goes through
    # the counterfactual guard gate (original violates the obligation AND secure
    # satisfies it); the contract class verifies a complete behavioral
    # counterexample plus searchable producer/consumer evidence.  Both classes
    # pass the positive criterion of the three threat questions (source/reachable
    # must name real source identifiers), which keeps generic hardening advice
    # from becoming a candidate.
    src_idents = oc.idents(source)
    changes = [ch for ch in changes if _change_gate(ch) and _threat_gate(ch, src_idents)]

    # diff candidates: one change -> one sink (the function name is looked up by
    # walking back from the line).
    diff_sinks: list[Sink] = []
    diff_by_func: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for ch in changes:
        line = ch["line"]
        fname = _func_of_line(functions, line)
        subject = str(ch.get("subject") or "")
        mechanism = ch.get("mechanism_kind") or ""
        key = _candidate_id(fname, line, mechanism, subject)
        if key in seen:
            continue
        seen.add(key)
        sink_type = _sink_type_of(ch.get("cwe_id", ""), mechanism)
        # S1: the difference only writes the candidate direction and never
        # presets a verification status; all other obligations are
        # UNDERDETERMINED.  The differing line is a locator pending
        # verification; only checker acceptance may write VIOLATED.
        targeted = ob.guard_target(sink_type, ch["change_kind"])
        obligs = ob.instantiate(
            sink_type, targeted=targeted,
            candidate_direction="VIOLATED",
            candidate_evidence=[f"{fname}:{line}"],
            candidate_provenance="secure_diff",
        )
        sink_id = key
        tj.record_step(
            node="secure_diff", action="admit_candidate",
            sink_id=sink_id,
            observation={"line": line, "subject": subject, "mechanism_kind": mechanism},
        )
        # The side-channel trace only records candidate generation; it is not an
        # obligation state transition.
        tj.record_step(
            node="secure_diff", action="instantiate_obligation",
            sink_id=sink_id, bucket=ob.bucket_key(ch.get("cwe_id", ""), sink_type),
            inputs={"change_kind": ch["change_kind"], "cwe_id": ch.get("cwe_id", "")},
            observation={"line": line, "snippet": src_lines[line - 1].strip()
                         if 1 <= line <= len(src_lines) else "",
                         "candidate_direction": "VIOLATED"},
            state_delta={},
            evidence_refs=[f"{fname}:{line}"],
        )
        diff_sinks.append(Sink(
            sink_id=sink_id,
            func_name=fname, line=line,
            snippet=src_lines[line - 1].strip() if 1 <= line <= len(src_lines) else "",
            sink_type=sink_type,
            tainted_tokens=_evidence_tokens(ch), source="secure_diff",
            change_kind=ch["change_kind"], obligations=obligs,
            **({"cwe_id": str(ch.get("cwe_id") or "")} if ch.get("cwe_id") else {}),
            **({"mechanism_kind": mechanism} if mechanism else {}),
            **({"subject": subject} if subject else {}),
            **({"contract_witness": ch["contract_witness"]}
               if ch["change_kind"] == "contract" else {}),
        ))
        diff_by_func.setdefault(fname, []).append(ch)

    # Rule-floor backstop: the secure version may fail to fix a real
    # vulnerability, so rule-confirmed sensitive API lines are merged in (zero
    # token).
    rule_sinks: list[Sink] = []
    for f in functions:
        rule_sinks.extend(sl.rule_locate_sinks(source, f))
    all_sinks = sl.merge_sinks(rule_sinks, diff_sinks)
    # Triage hypotheses are extra investigation points, not verdicts. Distinct
    # mechanism/subject identities stay beside rule and diff sinks; duplicates merge.
    all_sinks = sl.merge_sinks(all_sinks, _provisional_sinks(state, src_lines))

    # baseline back-fill: only functions "with a diff" produce a BaselineItem
    # (already_safe=False), secure_lines = the "- original / + secure"
    # concatenation of that function's change lines, for arbiter view A +
    # missing_guard.  rationale carries the model's own answers to the three
    # threat questions (controllable source / concrete harm / reachable path), so
    # that the arbiter adjudicates on that basis instead of generically judging
    # a missing defense.
    baseline: list[BaselineItem] = []
    for fname, chs in diff_by_func.items():
        secure_lines = "\n".join(
            f"- {ch['original']}\n+ {ch['secure']}" for ch in chs
        )
        rationale = "; ".join(_threat_line(ch) for ch in chs if ch.get("rationale"))
        witnesses = [
            ch["contract_witness"] for ch in chs
            if ch["change_kind"] == "contract"
        ]
        baseline.append(BaselineItem(
            func_name=fname,
            secure_lines=secure_lines,
            rationale=rationale,
            already_safe=False,
            **({"contract_witnesses": witnesses} if witnesses else {}),
        ))

    # Contrast outcome flag: it lets the empty-output gate of output distinguish
    # "the secure-baseline contrast ran to completion and produced no difference
    # that passed the gates" (no_diff -- the contrast itself is positive evidence,
    # so the file level can be recorded as safe_proved) from "LLM
    # unavailable / degraded / contrast not completed" (the chain ran empty, and a
    # zero output must still be underdetermined).  When the rule floor or the
    # provisional bridge hits, sinks is non-empty and this flag does not enter
    # adjudication; output consumes it only in the empty-output branch where
    # n_sinks==0.
    if data is None:
        outcome = "llm_unavailable"
    elif not diff_sinks:
        # LLM output usable but zero differences passed the gates: secure ~
        # original (patch files land here)
        outcome = "no_diff"
    else:
        outcome = "diff"
    return {"sinks": all_sinks, "baseline": baseline, "degraded": degraded,
            "_secure_diff_outcome": outcome}


def _threat_line(ch: dict) -> str:
    """Join the rationale of one change and the three threat questions into a
    single sentence for downstream readers."""
    base = ch.get("rationale", "")
    parts = []
    if ch.get("threat_source"):
        parts.append(f"controllable source={ch['threat_source']}")
    if ch.get("threat_harm"):
        parts.append(f"harm={ch['threat_harm']}")
    if ch.get("threat_reachable"):
        parts.append(f"reachable={ch['threat_reachable']}")
    return base + (" (" + ", ".join(parts) + ")" if parts else "")


def _evidence_tokens(ch: dict) -> list[str]:
    witness = ch.get("contract_witness")
    if not isinstance(witness, dict):
        return []
    return [
        str(t).strip() for t in (witness.get("evidence_tokens") or [])
        if isinstance(t, str) and len(t.strip()) > 1
    ]


def _provisional_sinks(state: VulnState, src_lines: list[str]) -> list[Sink]:
    """Create bounded, checkable candidates from anchored mechanism hypotheses.

    A hypothesis chooses only a source location.  Its text and optional CWE never
    count as evidence, and every obligation begins UNDERDETERMINED.
    """
    if not config.MECHANISM_PROVISIONAL_SINKS:
        return []

    out: list[Sink] = []
    seen: set[tuple] = set()
    cap = max(0, config.MECHANISM_PROVISIONAL_SINKS_MAX)

    def _trace(status: str, hyp: dict, sink_id: str = "", reason: str = "") -> None:
        tj.record_step(
            node="secure_diff",
            action="triage_hypothesis",
            sink_id=sink_id,
            inputs={
                "line": hyp.get("line"),
                "mechanism": hyp.get("mechanism"),
                "anchor": hyp.get("anchor"),
            },
            observation={"candidate_status": status, **({"reason": reason} if reason else {})},
        )

    for hypothesis in state.get("_triage_hypotheses") or []:
        if not isinstance(hypothesis, dict):
            continue
        well_formed = "line" in hypothesis and bool(hypothesis.get("anchor"))
        if well_formed:
            _trace("raw", hypothesis)

        mechanism = _mechanism_kind(hypothesis.get("mechanism"))
        sink_type = _MECHANISM_SINK_TYPE.get(mechanism)
        try:
            line = int(hypothesis.get("line", 0))
        except (TypeError, ValueError):
            line = 0
        anchor = str(hypothesis.get("anchor") or "")
        subject = str(hypothesis.get("subject") or "")
        fname = _func_of_line(state.get("functions", []) or [], line) if line else ""
        key = (fname, line, sink_type or "", mechanism, subject)

        reason = ""
        if not mechanism or not sink_type:
            reason = "unmapped_mechanism"
        elif not (1 <= line <= len(src_lines)):
            reason = "invalid_line"
        if reason:
            if well_formed:
                _trace("rejected", hypothesis, reason=reason)
            continue
        if key in seen:
            _trace("deduplicated", hypothesis, sink_id=f"{fname}:{line}:{sink_type}:{mechanism}:{subject}")
            continue
        if len(out) >= cap:
            _trace("rejected", hypothesis, reason="cap")
            continue
        seen.add(key)
        sink_id = f"{fname}:{line}:{sink_type}:{mechanism}:{subject}"
        out.append(Sink(
            sink_id=sink_id, func_name=fname, line=line,
            snippet=src_lines[line - 1].strip(), sink_type=sink_type,
            tainted_tokens=sl.extract_tainted_tokens(f"{subject} {anchor}"),
            source="triage", change_kind="guard",
            obligations=ob.instantiate(sink_type), mechanism_kind=mechanism,
            subject=subject,
            **({"cwe_id": str(hypothesis["candidate_cwe"])}
               if hypothesis.get("candidate_cwe") else {}),
        ))
        _trace("accepted", hypothesis, sink_id=sink_id)
        tj.record_step(
            node="secure_diff", action="admit_triage", sink_id=sink_id,
            observation={"candidate_status": "accepted", "subject": subject},
        )
    return out


def _parse_contract_witness(raw) -> ContractWitness:
    if not isinstance(raw, dict):
        return ContractWitness()
    tokens = [
        str(t).strip() for t in (raw.get("evidence_tokens") or [])
        if isinstance(t, str) and len(t.strip()) > 1
    ]
    return ContractWitness(
        precondition=str(raw.get("precondition", "")),
        expected_behavior=str(raw.get("expected_behavior", "")),
        actual_behavior=str(raw.get("actual_behavior", "")),
        security_consequence=str(raw.get("security_consequence", "")),
        counterexample=str(raw.get("counterexample", "")),
        producer_evidence=str(raw.get("producer_evidence", "")),
        consumer_evidence=str(raw.get("consumer_evidence", "")),
        evidence_tokens=tokens,
    )


_MECHANISM_SINK_TYPE = {
    "memory_extent": "mem", "allocation_size": "alloc",
    "numeric_arithmetic": "arith", "type_conversion": "arith",
    "null_or_failure": "deref", "object_lifetime": "use",
    "command_input": "exec", "path_input": "file_op", "format_input": "format",
    "query_input": "sql",
    "state_contract": "state_transition",
    "trust_boundary": "boundary_validation",
    "initialization_cleanup": "resource_lifecycle",
    "concurrency_order": "concurrency_state",
}


def _candidate_id(func_name: str, line: int, mechanism_kind: str = "", subject: str = "") -> str:
    """Identity for one investigation candidate; same line can host several."""
    return f"{func_name}:{int(line)}:{mechanism_kind}:{subject}"


def _mechanism_kind(raw: object) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return value if value in {
        "memory_extent", "allocation_size", "numeric_arithmetic", "type_conversion",
        "null_or_failure", "object_lifetime", "command_input", "path_input",
        "format_input", "query_input", "initialization_cleanup", "state_contract",
        "trust_boundary", "concurrency_order",
    } else ""


def _sink_type_of(cwe_id: str, mechanism_kind: str = "") -> str:
    """Route by anchored mechanism, with CWE retained as the compatibility fallback."""
    by_mechanism = _MECHANISM_SINK_TYPE.get(_mechanism_kind(mechanism_kind))
    if by_mechanism:
        return by_mechanism
    c = str(cwe_id).upper()
    ids = set(re.findall(r"CWE-\d+", c))
    if ids & {"CWE-120", "CWE-125", "CWE-787", "CWE-119"}:
        return "mem"
    # CWE-190/191 are integer overflow / wraparound, not memory out-of-bounds ->
    # routed to arith and targeted at arithmetic_bound (already provided by
    # obligations.py), instead of misfiling an integer overflow under
    # readable_extent.
    if ids & {"CWE-190", "CWE-191"}:
        return "arith"
    if "CWE-134" in ids:
        return "format"
    if ids & {"CWE-78", "CWE-77"}:
        return "exec"
    if ids & {"CWE-22", "CWE-23"}:
        return "file_op"
    # Three families added to widen coverage (derived backwards from the test CWE
    # distribution): NULL dereference / object lifetime / numeric.
    if "CWE-476" in ids:
        return "deref"
    if ids & {"CWE-416", "CWE-415"}:
        return "use"
    if "CWE-369" in ids:
        return "arith"
    # Numeric conversion / truncation (the prompt's "cross-type assignment ->
    # truncation to a narrow type" signal): the obligation lands on the
    # arithmetic bound.
    if ids & {"CWE-681", "CWE-197"}:
        return "arith"
    # Uncontrolled allocation (the prompt's "result used for allocation" signal):
    # the obligation lands on the size-overflow check.
    if ids & {"CWE-770", "CWE-789", "CWE-400"}:
        return "alloc"
    # Everything else (uninitialized / leak / race / assertion DoS, ...) lands in
    # _FALLBACK (controllability + one dominating guard); we deliberately do not
    # force a template that does not fit -- a wrong template is worse than a
    # coarse one.
    return "other"


# The difference-axis vocabulary declared by the prompt (in one-to-one
# correspondence with the change_kind of SECURE_DIFF_SYSTEM).
_CHANGE_KINDS = {"guard", "contract", "type_fix", "api_swap"}


def _kind_of(raw: str, has_witness: bool) -> str:
    """Normalize change_kind: a term inside the vocabulary is used as is; a term
    outside it falls back to contract/guard depending on whether a contract
    witness exists (the guard gate is the strictest one, so an unknown term can
    only be rejected, and will never be silently missed because the parse layer
    does not recognize it)."""
    kind = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if kind in _CHANGE_KINDS:
        return kind
    return "contract" if has_witness else "guard"



def _parse_changes(data, n_lines: int) -> list[dict]:
    """Normalize the LLM output into a list of change entries; compatible with a
    bare array and with {"changes":[...]}.  Out-of-range line numbers are
    discarded.

    One output entry may cover a **cross-line defect** (the prompt's anchors: a
    narrow type at the declaration plus an accumulation at the use site).  Each
    anchor is expanded into an independent change of the same family, inheriting
    cwe_id / change_kind / the three threat questions / the contract witness, and
    then passes the same deterministic gate set one by one -- expansion rather
    than merging, because downstream forensics and obligation instantiation both
    work per line.
    """
    if isinstance(data, dict):
        data = data.get("changes") or data.get("diffs") or []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        threat = it.get("threat")
        threat = threat if isinstance(threat, dict) else {}
        raw_witness = it.get("contract_witness") or it.get("contract")
        kind = _kind_of(it.get("change_kind", ""), isinstance(raw_witness, dict))
        common = {
            "cwe_id": str(it.get("cwe_id", "")),
            "mechanism_kind": _mechanism_kind(it.get("mechanism_kind", "")),
            "change_kind": kind,
            "contract_witness": _parse_contract_witness(raw_witness),
            "rationale": str(it.get("rationale", "")),
            # The model's own answers to the three threat questions
            # (source = controllable data source / harm = concrete harm /
            # reachable = reachable path) are passed downstream, so that forensics
            # can reverse-grep from the controllable source and the arbiter can
            # adjudicate on the concrete harm.
            "threat_source": str(threat.get("source", "")),
            "threat_harm": str(threat.get("harm", "")),
            "threat_reachable": str(threat.get("reachable", "")),
            "subject": str(it.get("subject", "")),
        }
        # Primary anchor plus the additional anchors in `anchors`, all with the
        # same (line, original, secure) shape.
        spots = [it]
        for a in (it.get("anchors") or []):
            if isinstance(a, dict):
                spots.append(a)
        for spot in spots:
            try:
                line = int(spot.get("line", 0))
            except (TypeError, ValueError):
                continue
            if not (1 <= line <= n_lines):
                continue
            out.append({
                **common,
                "line": line,
                "original": str(spot.get("original", "")),
                "secure": str(spot.get("secure", "")),
                "subject": str(spot.get("subject") or common.get("subject") or ""),
            })
    return out


def _parse_defenses(data) -> list[dict]:
    """Extract the defense anchors on which the model **suppresses** a point (the
    prompt's defense_at={line,snippet}).

    The prompt requires "even when judging something already safe, give a
    verifiable anchor", and this function receives it: compatible with a
    top-level defenses array and with the per-entry defense_at.  Returns
    [{line, snippet}], which the caller checks against the same anchoring
    criteria -- anchored = the suppression is grounded (written to the trace), not
    anchored = the suppression is ungrounded (also written to the trace, marked
    rejected).  This channel **does not participate in adjudication** (it
    produces no candidates and writes no obligations); it only prevents
    "already safe" from becoming an untraced silence.
    """
    items: list = []
    if isinstance(data, dict):
        items = list(data.get("defenses") or [])
        data = data.get("changes") or data.get("diffs") or []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and isinstance(it.get("defense_at"), dict):
                items.append(it["defense_at"])
    out: list[dict] = []
    for d in items:
        if not isinstance(d, dict):
            continue
        try:
            line = int(d.get("line", 0))
        except (TypeError, ValueError):
            continue
        snippet = str(d.get("snippet") or d.get("original") or "")
        if line >= 1 and _norm(snippet):
            out.append({"line": line, "snippet": snippet})
    return out


def _record_defenses(defenses: list[dict], src_lines: list[str]) -> None:
    """Re-check the suppression anchors against the same anchoring criteria and
    write the outcome to the trace (it does not participate in adjudication).

    Symmetry: claiming a vulnerability requires an `original` that can be
    anchored, and claiming the code is already safe likewise requires a
    `defense_at` that can be anchored.  Anchored -> accepted (the suppression is
    grounded); not anchored -> rejected (the model claimed a defense but the
    source has no such line).  Both outcomes are recorded so that "already safe"
    never becomes an untraced silence: abstention must not degrade into silence.
    """
    n = len(src_lines)
    for d in defenses:
        line, quote = d["line"], _norm(d["snippet"])
        cands = [line] + [c for dd in range(1, _ANCHOR_RADIUS + 1)
                          for c in (line - dd, line + dd)]
        cands = [c for c in cands if 1 <= c <= n]
        hit = next((c for c in cands if _norm(src_lines[c - 1]) == quote), None)
        if hit is None and len(quote) >= _MIN_QUOTE_CHARS:
            hit = next((c for c in cands if quote in _norm(src_lines[c - 1])), None)
        tj.record_step(
            node="secure_diff", action="check_defense_anchor",
            inputs={"claimed_line": line, "snippet": d["snippet"]},
            observation={"anchored_line": hit or 0,
                         "verdict": "accepted" if hit else "rejected"},
            evidence_refs=[f":{hit}"] if hit else [],
        )
