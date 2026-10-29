"""Deterministic obligation checker.

A textual citation only establishes existence; it never entails risk semantics. A
location string, a cross-file double citation, and a proposer-supplied provenance
are each insufficient on their own to authorize VIOLATED. The witness and guard
channels are still verified under their limited CFG / differential-constraint
rules and do not constitute a complete C/C++ proof.
"""
from __future__ import annotations

import re
from typing import Optional, TypedDict

from ..state import Obligation
from .. import config as vcfg
from . import obligations as ob
from . import witness as wit
from .cfg import CFG

_SETTLED = {"SATISFIED", "VIOLATED"}
_MIN_QUOTE_LEN = 6  # too short to anchor (avoids hits on noise like "if"/"len")

# "Guard obligations" (SATISFIED direction) that the mitigation gate can verify
# deterministically: set to true when a valid guard dominating the sink exists.
# Aligned with the value range of tools/obligations._GUARD_TARGET, plus the generic
# dominating_guard / non_negative_length.
_GUARD_OBLIGATIONS = {
    "readable_extent", "size_overflow_checked", "input_sanitized",
    "path_validated", "format_string_fixed", "query_parameterized",
    "null_guard", "object_live_at_use", "arithmetic_bound",
    "dominating_guard", "non_negative_length",
}

# Guard-line "shape" predicate: must look like a check (branch / assertion /
# comparison / boundary macro), otherwise it is not accepted as a guard anchor.
_GUARD_SHAPE = re.compile(
    r"\b(if|while|assert|BUG_ON|WARN_ON|OP_REQUIRES|CHECK|require|IS_ERR|"
    r"likely|unlikely|min|max|clamp)\b|[<>]=?|==|!=|!\s*[A-Za-z_(]"
)
# Window lines carrying a line-number prefix (number_lines yields "12| xxx";
# unit tests and compatibility also accept "12: xxx").
_NUM_LINE = re.compile(r"^\s*(\d+)\s*[|:]\s?(.*)$")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_STOP_IDENTS = {
    "if", "else", "for", "while", "return", "goto", "break", "continue",
    "int", "char", "void", "const", "unsigned", "long", "short", "size_t",
    "struct", "sizeof", "null", "case", "switch", "do", "static", "the",
}


def _idents(text: str) -> set[str]:
    """Extract identifiers (>2 chars, keywords removed) to relate guards to sink
    variables."""
    return {
        t for t in _IDENT_RE.findall(text or "")
        if len(t) > 2 and t.lower() not in _STOP_IDENTS
    }


def derives_from(cfg: Optional[CFG], sink_idents: set[str], src_var: str,
                 *, control: bool = False) -> bool:
    """Deterministic re-check of `derived_from`: is a variable the sink cares about
    derived from src_var along an *assignment chain*?

    The risky value is often a derived quantity (`N = shape_vec[0]`, `n = x.size()`,
    `p = &buf[k]`), and a protection acting on the source variable also protects it.
    The "who derives from whom" relation is not taken on the proposer's word: it is
    looked up through the assignment-dependency closure of `cfg.defs()`
    (flow-insensitive, no alias analysis, an over-approximation that can only accept
    *less*). When the CFG is unavailable nothing is accepted, consistent with the
    honest boundary of the other channels: a limitation only reduces what this
    module can prove.

    `control=True` folds control dependencies into the closure (flag-style
    validation: after `if (n > cap) bad = 1;` a later `if (bad) ...` disconnects bad
    from n in a pure assignment chain). The extension enlarges the closure and makes
    acceptance easier, so it is used *only* for voiding an assumption (I7's
    `protection_exists`); the strict re-check that writes SATISFIED keeps using the
    narrow closure.
    """
    if cfg is None or not cfg.nodes or not src_var:
        return False
    deps = cfg.defs(control=control)
    seen: set[str] = set()
    stack = [t for t in sink_idents if t]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur == src_var:
            return True
        for nxt in deps.get(cur, set()):
            if nxt not in seen:
                stack.append(nxt)
    return False


def idents(text: str) -> set[str]:
    """Public entry: extract identifiers (>2 chars, keywords removed). The positive
    criterion behind secure_diff's three threat questions shares this extraction
    rule with the module's guard re-check, so both apply the same notion of
    grep-able specificity."""
    return _idents(text)


def looks_like_guard(text: str) -> bool:
    """Public entry: the guard-shape predicate, i.e. whether the text looks like a
    check (branch / assertion / comparison / boundary macro). It is shared by
    secure_diff's counterfactual guard gate and this module's verify_mitigation."""
    return bool(_GUARD_SHAPE.search(text or ""))


def _parse_numbered(source: str) -> dict[int, str]:
    """Parse a numbered window into {line number: source line}. Unparsable lines are
    ignored."""
    out: dict[int, str] = {}
    for ln in (source or "").split("\n"):
        m = _NUM_LINE.match(ln)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


class CandidateFact(TypedDict, total=False):
    """A candidate fact proposed by an LLM proposer and consumed by this module. A
    proposer never writes an obligation status directly."""
    obligation_id: str                  # target obligation id (must exist in the
                                        # template for this sink_type)
    direction: str                      # "VIOLATED" | "SATISFIED"
    anchors: list[str]                  # source anchors (verbatim spans or file:line)
    cross_file: bool                    # True -> cross-file replay certificate
                                        # required; False -> local re-check
    witness: dict                       # optional machine-checked witness (see
                                        # state.Witness): the semantic channel
    cross_file_certificate: dict        # entry point and concrete input; call, return
                                        # and path are replayed from source
    derived_from: str                   # optional source variable the risky value
                                        # derives from (the guard acts on it)
    provenance: str                     # proposer class; used only for auditing and
                                        # I5's deterministic-proposer exemption


def seeded_candidate_facts(obligations: list[Obligation]) -> list[CandidateFact]:
    """Turn candidate material left by a proposer in the snapshot into facts pending
    re-check.

    This function neither reads nor modifies ``status``. The candidate direction
    kept by secure_diff is separate from the verification state; only after
    ``discharge`` accepts a fact may the obligation be written as VIOLATED or
    SATISFIED.
    """
    out: list[CandidateFact] = []
    for o in obligations or []:
        direction = str(o.get("candidate_direction") or "").upper()
        oid = str(o.get("id") or "")
        if direction not in _SETTLED or not oid:
            continue
        out.append(CandidateFact(
            obligation_id=oid,
            direction=direction,
            anchors=[a for a in (o.get("candidate_evidence") or []) if isinstance(a, str)],
            cross_file=False,
            provenance=str(o.get("candidate_provenance") or "").strip(),
        ))
    return out


def inferred_guard_facts(sink: dict, cfg: Optional[CFG]) -> list[CandidateFact]:
    """Produce a small number of ``SATISFIED`` candidate facts that the CFG can
    re-check independently.

    No safety is inferred here from "no vulnerability was reported", and no label
    or patch is read. Only two patterns whose semantics are fully determined by
    local syntax are covered: the null-pointer failure branch before a dereference,
    and the zero branch before a division or modulo. The guard must truly dominate
    the sink, and the must-take path to the sink must exclude the dangerous
    condition.

    These facts still pass through ``verify_fact``'s anchor, object-association and
    CFG-dominance checks like any other; the function only removes the unnecessary
    dependency on an LLM submitting an `unsafe` field for an existing guard to be
    visible to the machine at all.
    """
    if cfg is None or not cfg.entry:
        return []
    sink_line = int(sink.get("line") or 0)
    if sink_line <= 0:
        return []
    kind = str(sink.get("sink_type") or "")
    snippet = str(sink.get("snippet") or "")
    tokens = deref_ptr_candidates(sink) if kind == "deref" else [
        t for t in (sink.get("tainted_tokens") or []) if isinstance(t, str) and t
    ]
    if not tokens:
        return []
    out: list[CandidateFact] = []
    for guard in cfg.dominating_guards(sink_line):
        text = str(guard.get("text") or "")
        take = guard.get("must_take")
        if take is None or not text:
            continue
        for tok in tokens:
            t = re.escape(tok)
            if kind == "deref":
                # Reaching the sink must take the "non-null" side:
                # if (!p) return / if (p == NULL) return.
                null_on_true = bool(re.search(
                    rf"!\s*\(?\s*{t}\b|\b{t}\s*==\s*(?:NULL|nullptr|0)\b|"
                    rf"\b(?:NULL|nullptr|0)\s*==\s*{t}\b", text))
                null_on_false = bool(re.search(
                    rf"\b{t}\s*!=\s*(?:NULL|nullptr|0)\b|"
                    rf"\b(?:NULL|nullptr|0)\s*!=\s*{t}\b", text))
                safe = (null_on_true and take is False) or (null_on_false and take is True)
                oid = "null_guard"
            elif kind == "arith" and ("/" in snippet or "%" in snippet):
                # Only the zero branch of a denominator or modulus is handled;
                # overflow bounds still need a stronger arithmetic proof.
                zero_on_true = bool(re.search(
                    rf"\b{t}\s*==\s*0\b|\b0\s*==\s*{t}\b", text))
                zero_on_false = bool(re.search(
                    rf"\b{t}\s*!=\s*0\b|\b0\s*!=\s*{t}\b", text))
                safe = (zero_on_true and take is False) or (zero_on_false and take is True)
                oid = "arithmetic_bound"
            else:
                safe = False
                oid = ""
            if safe:
                # The CFG stores the condition subtree (e.g. ``(!p)``), while the
                # anchor re-check must be able to hit the original ``if (...)`` line;
                # restoring the keyword introduces no new semantics and is still
                # verified verbatim downstream.
                anchor = f"if {text}"
                out.append(CandidateFact(
                    obligation_id=oid, direction="SATISFIED", anchors=[anchor],
                    cross_file=False, provenance="deterministic_guard_inference",
                ))
                break
    return out


def _norm(text: str) -> str:
    """Normalize whitespace for substring anchoring."""
    return " ".join((text or "").split())


def verify_local(quote: str, source: str) -> bool:
    """Local anchor re-check: after normalization the quote must be a substring of
    the source and long enough."""
    q = _norm(quote)
    if len(q) < _MIN_QUOTE_LEN:
        return False
    return q in _norm(source)


def verify_cross_file(
    anchors: list[str], consumer_source: str, searched_source: str,
    *, certificate: Optional[dict] = None, project_sources: Optional[dict] = None,
    sink: Optional[dict] = None, target_path: str = "",
    obligation_id: str = "", direction: str = "VIOLATED",
) -> bool:
    """Cross-file semantic discharge. A citation is not a certificate: every relation
    is replayed from trusted source code."""
    from .cross_file_checker import check_certificate
    return check_certificate(certificate, project_sources, sink or {}, target_path,
                             obligation_id, direction)[0]


def cross_file_quotes_present(
    anchors: list[str], consumer_source: str, searched_source: str,
) -> bool:
    """Retrieval-only check; never authorizes an obligation transition."""
    normalized: list[str] = []
    for a in anchors:
        q = _norm(a)
        if len(q) >= _MIN_QUOTE_LEN and q not in normalized:
            normalized.append(q)
    if len(normalized) < 2:
        return False
    consumer = _norm(consumer_source)
    searched = _norm(searched_source)
    consumer_hits = {i for i, q in enumerate(normalized) if q in consumer}
    producer_hits = {i for i, q in enumerate(normalized) if q in searched}
    return any(c != p for c in consumer_hits for p in producer_hits)


def _dominates_sink(quote: str, consumer_source: str, sink: dict,
                    cfg: Optional[CFG]) -> bool:
    """Gate 4, "does the guard dominate the sink": only true dominance on the CFG is
    accepted.

    The true-dominance criterion is that some line hit by the anchor carries a
    branch node which dominates the sink line and whose must_take is not None (i.e.
    one side must be taken to reach the sink). When the CFG is unavailable, the sink
    is not in the graph, or the anchor cannot be located, this function rejects the
    mitigation claim; the caller then keeps UNDERDETERMINED instead of substituting
    textual line order for control-flow evidence.
    """
    sink_line = int(sink.get("line") or 0)
    numbered = _parse_numbered(consumer_source)
    norm_q = _norm(quote)
    hit_lines = [n for n, txt in numbered.items() if norm_q in _norm(txt)]
    if not (cfg is not None and cfg.entry and sink_line and hit_lines):
        return False
    guards = [g for g in cfg.dominating_guards(sink_line)
              if g.get("must_take") is not None]
    if not guards:
        return False   # no dominating guard with a must-take side was found
    return any(g["line"] in hit_lines for g in guards)


def _sink_idents(sink: dict) -> set[str]:
    """Identifiers of the objects the sink cares about: the tainted tokens (taken
    verbatim, single-character pointer names such as p included) unioned with the
    identifiers of the snippet."""
    tainted = {t for t in (sink.get("tainted_tokens") or []) if isinstance(t, str) and t}
    return set(tainted) | _idents(sink.get("snippet", ""))


def _same_object(
    quote: str, sink: dict, cfg: Optional[CFG] = None, derived_from: str = "",
) -> bool:
    """Whether the guard acts on the same object the sink cares about.

    A direct hit (the citation shares a tainted token or snippet identifier with the
    sink) passes. Otherwise **derivation lets it pass**: if the proposer declares
    `derived_from=<source variable>` and `derives_from` confirms it along the
    assignment chain, a guard acting on the source variable also counts as acting on
    the risky value (a bound check for `N = shape_vec[0]` sits on shape_vec). The
    derivation is re-checked against the CFG's assignment-dependency closure rather
    than taken from the proposer's assertion.
    """
    sink_idents = _sink_idents(sink)
    if not sink_idents:
        return True
    guard_idents = set(_IDENT_RE.findall(quote or ""))
    if guard_idents & sink_idents:
        return True
    src = str(derived_from or "").strip()
    return bool(
        src and src in guard_idents
        and derives_from(cfg, sink_idents, src)
    )


def _data_related(quote: str, sink: dict, cfg: Optional[CFG]) -> bool:
    """Whether the citation talks about the *same data* as the sink (a relatedness
    criterion looser than `_same_object`).

    The criterion is "citation identifiers intersect the dependency closure
    (including control dependencies) of the sink variables", computed by
    `cfg.dep_closure`. It is needed because an equivalent protection often shares no
    identifier with the sink at all: `OP_REQUIRES(ctx, !overflow_ocurred, ...)`
    validates a flag while the sink uses a quantity derived from
    `input_shape_vec`, so demanding a literally shared identifier systematically
    blinds the checker to exactly this kind of protection, which is the protection in
    question. Requiring the proposer to fill in `derived_from` does not help either,
    because the source variable often does not appear in the citation (the citation
    only contains the flag).

    The closure is an over-approximation (flow-insensitive, no alias analysis,
    undirected, includes control dependencies), so this criterion serves a
    *one-sided* purpose only: voiding a preset assumption of `secure_diff`
    (-> UNDERDETERMINED). Writing SATISFIED still goes through `_same_object`.
    """
    sink_idents = _sink_idents(sink)
    if not sink_idents:
        return True
    guard_idents = set(_IDENT_RE.findall(quote or ""))
    if guard_idents & sink_idents:
        return True
    if cfg is None or not cfg.nodes:
        return False
    return bool(guard_idents & cfg.dep_closure(sink_idents))


def verify_mitigation(
    quote: str, consumer_source: str, sink: dict, cfg: Optional[CFG] = None,
    derived_from: str = "",
) -> bool:
    """Mitigation anchor re-check (SATISFIED direction, the deterministic criterion
    behind `mitigation_gate`, no LLM).

    Symmetric with the strict re-check for VIOLATED: an anchor is accepted only when
    it really is a guard that *dominates the sink and acts on the same tainted
    token*, since accepting less would cost recall on the P-V -> P-B transition. All
    four hard gates must pass:
      1. the anchor hits the consumer window verbatim (reusing verify_local);
      2. the anchor line "looks like a check" (branch / assertion / comparison /
         boundary macro, _GUARD_SHAPE);
      3. the anchor mentions some tainted token of the sink or an identifier from
         the snippet (the guard acts on the same object), or `derived_from` confirms
         along the assignment chain that it is the source variable of this risky
         value (see `_same_object`);
      4. the anchor line *truly dominates* the sink on the CFG (the must-take
         criterion of `cfg.dominating_guards`).
         When the CFG is unavailable no mitigation is proven and the obligation stays
         UNDERDETERMINED.
    """
    q = _norm(quote)
    if len(q) < _MIN_QUOTE_LEN or q not in _norm(consumer_source):
        return False
    if not _GUARD_SHAPE.search(quote):
        return False
    if not _same_object(quote, sink, cfg, derived_from):
        return False
    return _dominates_sink(quote, consumer_source, sink, cfg)


# ---- I7: assumption-voiding gate (existence check for equivalent protections;
# no guard template, no API name table) ----
def _function_hit_lines(quote: str, cfg: Optional[CFG], consumer_source: str) -> list[int]:
    """The set of line numbers where the citation hits the *real source of that
    function*.

    With a CFG, the search runs over the node texts of the whole function (this
    matters: an equivalent protection often falls outside the +/- risk window, so
    searching only the window would systematically hide it). Without a CFG the
    search falls back to the numbered window. Node text has whitespace normalized
    and is truncated when long, so a substring test is used rather than equality.
    """
    q = _norm(quote)
    if len(q) < _MIN_QUOTE_LEN:
        return []
    if cfg is not None and cfg.nodes:
        hits = {n.line for n in cfg.nodes.values()
                if n.line and q in _norm(n.text)}
        if hits:
            return sorted(hits)
    return sorted(n for n, txt in _parse_numbered(consumer_source).items()
                  if q in _norm(txt))


def protection_exists(
    quote: str, sink: dict, consumer_source: str, cfg: Optional[CFG] = None,
    derived_from: str = "",
) -> bool:
    """Check whether "a protection that already exists in the original code" really
    exists and is effectively placed (the deterministic half of I7).

    Division of labour: *equivalence is judged inductively by the proposer (LLM)*,
    which states that a line does not look like the idealized safe if-check yet
    still makes the obligation hold. This function only re-checks three recomputable
    things:
      1. verbatim existence: the citation occurs in the real source of that function
         (see `_function_hit_lines`);
      2. effective placement: the hit line *truly dominates* the sink on the control
         flow graph and is not the sink line itself (copying back the accused line
         carries no new information);
      3. same object: the citation shares a tainted token or snippet identifier with
         the sink, or is connected to the sink variables through the dependency
         closure (`_data_related`: control dependencies included, covering equivalent
         protections such as "the validation writes a flag while the sink uses a
         derived quantity" where not a single identifier overlaps).
    Requiring "looks like an if / comparison / assertion" is deliberately *not* part
    of the criterion: a validation function returning a status code checked by the
    caller, a construction-time validation, and a length-capping API contain no
    branch literal yet still make the obligation hold. That shape gate is precisely
    the source of "an equivalent protection mistaken for a missing one", and it is
    why this project must not hard-code any table of validation function names (such
    a table fails on both unfamiliar projects and private APIs).
    One-sidedness: this criterion is only used to *void an assumption* (preset
    VIOLATED -> UNDERDETERMINED) and never to write SATISFIED, which still has to
    pass the strict guard re-check in `verify_mitigation`. A proposer that falsely
    claims "a protection already exists" therefore can at worst turn a point into an
    undetermined one; it cannot report a real vulnerability as proven safe.
    """
    sink_line = int(sink.get("line") or 0)
    hits = [n for n in _function_hit_lines(quote, cfg, consumer_source)
            if n != sink_line]
    if not hits:
        return False
    if not (_same_object(quote, sink, cfg, derived_from)
            or _data_related(quote, sink, cfg)):
        return False
    if not (cfg is not None and cfg.entry and sink_line):
        return False
    targets = [n.id for n in cfg.nodes_at_line(sink_line)]
    cand = [n.id for n in cfg.nodes.values() if n.line in hits]
    return any(cfg.dominates(a, b) for a in cand for b in targets)


# ---- (1) premise gate (deref / CWE-476 only, deterministic) ----
# An assertion such as "the value returned by an external API may be NULL" concerns a
# premise that is *not visible in a single file*, and an anchor that is merely the
# assignment line itself carries no information beyond that. A deref-type critical
# obligation may therefore go to VIOLATED only when both of the following hold:
#   (A) the nullable-source premise is discharged: the anchor names a nullable API in
#       the registry / the cross-file double-anchor re-check passes / the pointer
#       visible in the window is assigned from an API in the registry;
#   (B) an in-file dereference anchor: the tainted pointer is genuinely dereferenced
#       within the consumption window (*p / p-> / p[]). Merely passing the pointer
#       to an external function does not count: the dereference happens in another
#       file, so single-file evidence cannot support a positive verdict.
_DEREF_CRITICAL = {"attacker_control", "null_guard"}
_DEREF_USE = r"(\*\s*{t}\b)|(\b{t}\s*->)|(\b{t}\s*\[)"


def has_infile_deref(consumer_source: str, tainted: list[str]) -> bool:
    """(B) In-file dereference anchor: some tainted pointer occurs in the window in
    the form *p / p-> / p[]."""
    for tok in tainted or []:
        if not isinstance(tok, str) or not tok:
            continue
        t = re.escape(tok)
        if re.search(_DEREF_USE.format(t=t), consumer_source or ""):
            return True
    return False


def deref_ptr_candidates(sink: dict) -> list[str]:
    """Pointer candidates the premise gate looks at: tainted tokens first; failing
    that the left-hand side of the assignment in the sink snippet (guard-type
    secure_diff candidates often carry no tainted token); failing that, all
    identifiers of the snippet."""
    tainted = [t for t in (sink.get("tainted_tokens") or [])
               if isinstance(t, str) and t]
    if tainted:
        return tainted
    snippet = str(sink.get("snippet", ""))
    m = re.search(r"\b([A-Za-z_]\w*)\s*=(?!=)", snippet)
    if m:
        return [m.group(1)]
    return sorted(_idents(snippet))


# A citation where the producer explicitly returns NULL (a common form of
# cross-file retrieval evidence) also counts as nullable-source evidence.
_RETURN_NULL = re.compile(r"return[^;]{0,60}\bNULL\b|=\s*NULL\s*;", re.I)


def nullable_source_ok(fact: "CandidateFact", sink: dict, consumer_source: str) -> bool:
    """(A) Whether the nullable-source premise is discharged (three routes, see
    above). The re-check of the two independent cross_file anchors is performed by
    verify_fact; here only the route itself is accepted. An anchor that quotes the
    producer's "return / assign NULL" source (a common form of cross-file retrieval
    evidence) is accepted as well."""
    anchors = [a for a in (fact.get("anchors") or []) if isinstance(a, str)]
    if any(ob.mentions_nullable_api(a) or _RETURN_NULL.search(a) for a in anchors):
        return True
    if fact.get("cross_file"):
        return True
    for tok in deref_ptr_candidates(sink):
        m = re.search(
            rf"\b{re.escape(tok)}\s*=\s*([A-Za-z_]\w*)\s*\(", consumer_source or "",
        )
        if m and m.group(1) in ob.NULLABLE_APIS:
            return True
    return False


def deref_premise_ok(fact: "CandidateFact", sink: dict, consumer_source: str) -> bool:
    """The deref premise gate: (A) nullable source discharged AND (B) in-file
    dereference anchor."""
    return (nullable_source_ok(fact, sink, consumer_source)
            and has_infile_deref(consumer_source, deref_ptr_candidates(sink)))


# ---- I5 anchor-independence gate ----

def _sink_line_text(sink: dict, consumer_source: str) -> str:
    """The text of the sink line in the consumption window (falls back to
    sink.snippet)."""
    numbered = _parse_numbered(consumer_source)
    txt = numbered.get(int(sink.get("line") or 0), "")
    return txt or str(sink.get("snippet", ""))


def anchor_is_independent(anchor: str, sink: dict, consumer_source: str) -> bool:
    """I5: an anchor must carry information beyond the sink line.

    Copying back the accused line (or a substring of it) is not evidence: "there is a
    memcpy here" is both the accusation and the alleged evidence, so the proposer is
    vouching for itself. A file:line locator that points at the sink line is equally
    not independent.
    """
    a = _norm(anchor)
    if not a:
        return False
    if _is_locator(a):
        tail = a.rsplit(":", 1)[-1]
        return not (tail.isdigit() and int(tail) == int(sink.get("line") or -1))
    sink_txt = _norm(_sink_line_text(sink, consumer_source))
    if not sink_txt:
        return True
    return not (a in sink_txt or sink_txt in a)


# ---- I6 external-source decidability gate (a generalization of deref's
# nullable-source premise) ----
# For `x = SomeExternalCall(...)`, "what values can x take" is not visible in a
# single file, because the callee's range is not visible. Under I3 the conservative
# reading keeps the obligation UNDERDETERMINED instead of taking the assignment line
# as evidence for a positive verdict: that line only states "this value is invisible
# to me", which is a confession of missing information, not evidence of
# controllability or of an out-of-bounds condition.
#
# "The token is compared somewhere in the window" is deliberately *not* an accepting
# condition: a branch such as `if (packet_size == 1)` only handles cases one by one
# and says nothing about the range of the returned value. It is material for the
# SATISFIED direction (handled by verify_mitigation / guard_proves_safe) and cannot
# in turn make VIOLATED decidable.
_OPAQUE_ASSIGN = r"\b{t}\s*=\s*([A-Za-z_]\w*)\s*\("


def opaque_external_value(token: str, consumer_source: str) -> str:
    """Whether `token` holds a value returned by an external function and is
    therefore undecidable from a single file.

    Returns the callee name (undecidable) or "" (decidable: no external assignment in
    the window, or the callee is in the registry of well-known source / nullable APIs
    whose semantics the machine can look up).
    """
    if not token or not consumer_source:
        return ""
    t = re.escape(token)
    m = re.search(_OPAQUE_ASSIGN.format(t=t), consumer_source)
    if not m:
        return ""     # no external assignment in the window
    callee = m.group(1)
    if ob.mentions_source_api(callee) or ob.mentions_nullable_api(callee):
        return ""     # a well-known external source: the machine can look it up
    return callee


def external_premise_ok(
    fact: "CandidateFact", sink: dict, consumer_source: str,
) -> tuple[bool, str]:
    """I6 gate: if the tainted value a candidate fact relies on is externally
    returned and unconstrained within the window, it is undecidable from a single
    file.

    Accepting conditions (any one suffices): cross_file (the certificate has been
    verified by inter-procedural replay); an anchor names a source / nullable API in
    the registries; or the tainted value is not the return value of an external call
    within the window. Returns (accepted, reason when rejected). The witness channel
    is exempted at the call site.
    """
    if fact.get("cross_file"):
        return True, ""
    anchors = [a for a in (fact.get("anchors") or []) if isinstance(a, str)]
    if any(ob.mentions_source_api(a) or ob.mentions_nullable_api(a) for a in anchors):
        return True, ""
    tokens = [t for t in (sink.get("tainted_tokens") or [])
              if isinstance(t, str) and t]
    for tok in tokens:
        callee = opaque_external_value(tok, consumer_source)
        if callee:
            return False, (
                f"undecidable external source: {tok} comes from {callee}(), with no "
                f"definition and no constraint in the window; single-file evidence "
                f"is insufficient for a positive verdict (a cross-file anchor or a "
                f"machine-checked witness is required)"
            )
    return True, ""


def verify_witness(fact: CandidateFact, sink: dict,
                   cfg: Optional[CFG]) -> tuple[bool, list[str], str]:
    """Witness-channel re-check (semantic level). Returns (discharged, evidence list,
    reason).

    VIOLATED direction -> the four-step machine check of `witness.check_witness`;
    SATISFIED direction -> `witness.guard_proves_safe` (does a machine-computed
    dominating guard contradict the unsafe condition). A missing witness or an
    unavailable CFG yields (False, [], reason) and the obligation stays
    UNDERDETERMINED (I3).
    """
    w = fact.get("witness")
    if not isinstance(w, dict) or not w:
        return False, [], "no witness"
    if cfg is None or not cfg.entry:
        return False, [], "CFG unavailable: the witness cannot be machine-checked"
    w = dict(w)
    w.setdefault("sink_line", int(sink.get("line") or 0))
    if int(w.get("sink_line") or 0) != int(sink.get("line") or 0):
        return False, [], "witness location does not match the target sink"
    if getattr(cfg, "degraded", None):
        return False, [], "the CFG is degraded, so no obligation can be discharged"
    direction = str(fact.get("direction", "")).upper()
    if direction == "SATISFIED":
        out = wit.guard_proves_safe(cfg, int(w.get("sink_line") or 0),
                                    str(w.get("unsafe") or ""))
        if not out.get("proved"):
            return False, [], str(out.get("reason") or "guard not proven")
        refs = [f"witness:guard {a}" for a in (out.get("anchors") or [])]
        return True, refs, str(out.get("reason") or "")
    rep = wit.check_witness(cfg, w)
    if not rep.get("verified"):
        return False, [], str(rep.get("reason") or "witness did not pass")
    refs = [f"witness:{a}" for a in (rep.get("anchors") or [])]
    refs.append(f"witness:sat={rep.get('sat')}")
    return True, refs, str(rep.get("reason") or "")



def verify_fact(
    fact: CandidateFact, consumer_source: str, searched_source: str,
    sink: Optional[dict] = None, cfg: Optional[CFG] = None,
    project_sources: Optional[dict] = None, target_path: str = "",
) -> tuple[bool, list[str], str]:
    """Re-check a candidate: an existing citation does not mean the obligation
    holds, and any unsupported semantic relation stays undetermined.

    Local proof of safety still requires a guard re-check. Local risk additionally
    requires a machine-checked witness; a cross-file citation only supports triage
    and never authorizes VIOLATED.
    """
    if fact.get("cross_file"):
        from .cross_file_checker import check_certificate
        ok, refs, reason, _audit = check_certificate(
            fact.get("cross_file_certificate"), project_sources, sink or {},
            target_path, str(fact.get("obligation_id") or ""),
            str(fact.get("direction") or "").upper(),
        )
        return ok, refs, reason
    if isinstance(fact.get("witness"), dict) and fact.get("witness"):
        # A5: when the witness channel is off, a candidate fact carrying a witness
        # fails outright and does **not** fall back to the anchor channel (see the
        # no-fallback design in verify_witness: an LLM must not bypass the semantic
        # check with an unwitnessable witness plus a verbatim citation). The
        # affected scope is limited to witness-type facts; deterministic proposers
        # such as micro_check emit anchor-type facts, do not enter this branch, and
        # are unaffected by A5.
        if not vcfg.WITNESS_CHANNEL_ENABLED:
            return False, [], "witness channel is disabled"
        ok, refs, why = verify_witness(fact, sink or {}, cfg)
        return ok, refs, why
    # A4: when the anchor channel is off, no non-witness candidate fact is
    # discharged (VIOLATED and SATISFIED alike: verify_mitigation is also an anchor
    # re-check, so it is disabled together).
    if not vcfg.ANCHOR_CHANNEL_ENABLED:
        return False, [], "anchor channel is disabled"
    anchors = [a for a in (fact.get("anchors") or []) if isinstance(a, str) and a.strip()]
    if not anchors:
        return False, [], "no non-empty anchor"
    direction = str(fact.get("direction", "")).upper()
    if direction == "SATISFIED":
        # Mitigation discharge: at least one anchor is a real guard that dominates
        # the sink and acts on the tainted token (or on its derived source variable).
        # derived_from is declared by the proposer; the derivation itself is
        # re-checked by derives_from along the assignment chain.
        s = sink or {}
        src = str(fact.get("derived_from") or "")
        # A positional locator can only demonstrate provenance; it cannot supply the
        # source content that the guard-shape test, the object association and CFG
        # dominance need. It therefore must not alone write the obligation as
        # SATISFIED.
        verified = [a for a in anchors
                    if not _is_locator(a)
                    and verify_mitigation(a, consumer_source, s, cfg, src)]
        return ((len(verified) > 0), verified,
                "mitigation anchor passed the source, object-association and CFG "
                "dominance checks"
                if verified else "mitigation anchor did not pass the source, "
                "object-association and CFG dominance checks together")
    # Existence is necessary provenance, never semantic entailment. In particular
    # a caller-controlled provenance string must not grant a semantic exemption.
    verified = [a for a in anchors if not _is_locator(a) and verify_local(a, consumer_source)]
    return False, [], ("local anchor hits the source, but the risk origin, data flow "
                       "and path constraints of the obligation remain unproven"
                       if verified else "local anchor misses verifiable source; a "
                       "positional string alone cannot discharge an obligation")


def _is_locator(anchor: str) -> bool:
    """Whether the anchor is a file:line locator (e.g. "net.c:84" / "foo:12")."""
    a = anchor.strip()
    if ":" not in a:
        return False
    tail = a.rsplit(":", 1)[-1]
    return tail.isdigit()


def _provenance_of(fact: CandidateFact) -> str:
    """Provenance label of the evidence (an audit trail only, never part of the
    adjudication).

    Witness channel -> machine_witness; deterministic proposers (micro_check and
    others reporting their own provenance) keep the source they reported; everything
    else is the LLM anchor channel, llm_anchor. This supports the ex-post statistic
    "how many obligations were discharged by the machine".
    """
    if fact.get("cross_file"):
        return "cross_file_replay"
    if isinstance(fact.get("witness"), dict) and fact.get("witness"):
        return "machine_witness"
    src = str(fact.get("provenance") or "").strip()
    return src or "llm_anchor"


def _cfg_audit(cfg: Optional[CFG]) -> dict:
    """A CFG audit snapshot that does not participate in adjudication, for offline
    review and result stratification."""
    if cfg is None or not cfg.entry:
        return {
            "status": "unavailable",
            "degraded": list(getattr(cfg, "degraded", []) or []),
        }
    degraded = list(getattr(cfg, "degraded", []) or [])
    return {
        "status": "degraded" if degraded else "constructed",
        "degraded": degraded,
    }


def _append_audit(obligation: Obligation, event: dict) -> None:
    """Append one structured re-check record; the record does not affect the
    obligation's tri-valued status."""
    audit = obligation.setdefault("verification_audit", {
        "cfg": {"status": "unavailable", "degraded": []}, "checks": [],
    })
    checks = audit.setdefault("checks", [])
    checks.append(event)


def discharge(
    sink: dict,
    facts: list[CandidateFact],
    *,
    consumer_source: str,
    searched_source: str = "",
    base: Optional[list[Obligation]] = None,
    cfg: Optional[CFG] = None,
    project_sources: Optional[dict] = None,
    target_path: str = "",
) -> tuple[list[Obligation], list[str]]:
    """Advance discharged obligations on a sink's obligation list (the sole
    discharger).

    - base: the initial obligation list; defaults to sink.obligations, and failing
      that is instantiated from the sink_type.
    - cfg: the CFG of the function containing this sink (`tools/cfg.build`). It is a
      necessary condition for the witness channel, for SATISFIED guards and for the
      equivalent-protection check; when unavailable these channels stay
      UNDERDETERMINED.
    - Every candidate fact is re-checked deterministically; a pass sets the target
      obligation to `direction` and attaches the evidence (I1); a failure leaves the
      obligation untouched (usually UNDERDETERMINED, I3).
    - **Machine-self-proved guards** (when a cfg exists): even if the LLM proposed no
      SATISFIED fact at all, a guard that the machine computes to "dominate the sink
      and contradict the unsafe condition of some candidate fact" directly sets the
      corresponding guard obligation to SATISFIED. This step depends on no LLM output
      and is the deterministic exit on the patch side.
    Returns (the advanced obligation list, the evidence references that passed
    re-check).
    """
    sink_type = sink.get("sink_type", "")
    obs: list[Obligation] = [dict(o) for o in (base or sink.get("obligations") or
                                               ob.instantiate(sink_type))]
    cfg_audit = _cfg_audit(cfg)
    for o in obs:
        o.setdefault("verification_audit", {
            "cfg": dict(cfg_audit),
            "checks": [{
                "stage": "initial", "result": "seeded",
                "status": str(o.get("status") or "UNDERDETERMINED"),
                "provenance": str(o.get("provenance") or ""),
                "candidate_direction": str(o.get("candidate_direction") or ""),
                "candidate_provenance": str(o.get("candidate_provenance") or ""),
            }],
        })
    by_id = {o.get("id"): o for o in obs}
    verified_refs: list[str] = []
    # For two local guard shapes with unambiguous semantics the machine can propose
    # candidates from the CFG itself; every other pattern still accepts proposer
    # material only, so that a loose heuristic is never promoted to a safety
    # conclusion.
    all_facts = list(facts or []) + inferred_guard_facts(sink, cfg)

    # (1) The premise gate also applies to deref obligations carrying a *preset
    # VIOLATED* (secure_diff's initial candidates enter with initial_status=VIOLATED):
    # the candidate gate only proves that the diff is a real guard, not that the
    # nullable premise holds. A failed audit falls back to UNDERDETERMINED (I3), and
    # a later candidate fact can advance the obligation again if it discharges.
    if sink_type == "deref":
        for o in obs:
            if not (o.get("critical") and o.get("status") == "VIOLATED"
                    and o.get("id") in _DEREF_CRITICAL):
                continue
            probe: CandidateFact = {
                "anchors": [e for e in (o.get("evidence") or []) if isinstance(e, str)],
            }
            if not deref_premise_ok(probe, sink, consumer_source):
                o["status"] = "UNDERDETERMINED"
                o["unresolved_reason"] = (
                    "deref premise gate not passed: the nullable source is "
                    "undischarged, or no real dereference occurs in the window"
                )
                _append_audit(o, {
                    "stage": "deref_premise", "result": "rejected",
                    "reason": o["unresolved_reason"],
                })

    for fact in all_facts:
        oid = str(fact.get("obligation_id", ""))
        direction = str(fact.get("direction", "")).upper()
        if oid not in by_id or direction not in _SETTLED:
            continue
        # The mitigation gate (SATISFIED) only acts on obligations whose guard can
        # be re-checked; the VIOLATED direction is unrestricted.
        if direction == "SATISFIED" and oid not in _GUARD_OBLIGATIONS:
            continue
        replay_audit = None
        if fact.get("cross_file"):
            from .cross_file_checker import check_certificate
            ok, refs, why, replay_audit = check_certificate(
                fact.get("cross_file_certificate"), project_sources, sink,
                target_path, oid, direction,
            )
        else:
            ok, refs, why = verify_fact(
                fact, consumer_source, searched_source, sink=sink, cfg=cfg,
            )
        if not ok:
            _append_audit(by_id[oid], {
                "stage": "candidate", "result": "rejected",
                "direction": direction,
                "channel": "cross_file_replay" if fact.get("cross_file") else "witness" if isinstance(fact.get("witness"), dict) else "anchor",
                "replay": replay_audit,
                "reason": why,
            })
            continue  # I3: not discharged -> the obligation stays UNDERDETERMINED
        # (1) Premise gate: a deref-type critical obligation going to VIOLATED must
        # additionally pass the premise gate: the nullable source is discharged
        # (nullable-API registry / cross-file anchor / nullable assignment in the
        # window) and a real dereference occurs in the window. Failing this keeps the
        # obligation UNDERDETERMINED (I3), so a verbal nullable claim about an
        # external API no longer advances an obligation. Facts that have passed the
        # witness channel are exempt: the witness has already proven the
        # dereference and the path on the CFG, which is stronger than a heuristic.
        if (direction == "VIOLATED" and sink_type == "deref"
                and oid in _DEREF_CRITICAL
                and not fact.get("cross_file_certificate")
                and not isinstance(fact.get("witness"), dict)
                and not deref_premise_ok(fact, sink, consumer_source)):
            _append_audit(by_id[oid], {
                "stage": "deref_premise", "result": "rejected",
                "direction": direction,
                "reason": "the nullable-source premise is not discharged by source",
            })
            continue
        # (6) External-source decidability gate (I6, a generalization of the
        # nullable-source premise above): if unsafety depends on a value that is
        # "returned by an external function and has neither definition nor
        # constraint in the window", single-file evidence is insufficient for a
        # positive verdict; copying back that assignment only proves "this value is
        # invisible to me". The witness channel is exempt, having already checked
        # taint and constraints on the CFG. A2: this gate is disabled when
        # EXTERNAL_PREMISE_ENABLED=0 (ablation).
        if (direction == "VIOLATED" and vcfg.EXTERNAL_PREMISE_ENABLED
                and not isinstance(fact.get("witness"), dict)):
            ok_ext, why = external_premise_ok(fact, sink, consumer_source)
            if not ok_ext:
                o = by_id[oid]
                if o.get("status") == "UNDERDETERMINED":
                    o["unresolved_reason"] = why
                _append_audit(o, {
                    "stage": "external_premise", "result": "rejected",
                    "direction": direction, "reason": why,
                })
                continue
        o = by_id[oid]
        o["status"] = direction
        merged = list(o.get("evidence") or [])
        for r in refs:
            if r not in merged:
                merged.append(r)
        o["evidence"] = merged
        o["provenance"] = _provenance_of(fact)
        _append_audit(o, {
            "stage": "candidate", "result": "accepted",
            "direction": direction,
            "channel": "cross_file_replay" if fact.get("cross_file") else "witness" if isinstance(fact.get("witness"), dict) else "anchor",
            "reason": why,
            "evidence": list(refs),
            "replay": replay_audit,
        })
        for r in refs:
            if r not in verified_refs:
                verified_refs.append(r)

    # (7) Assumption-voiding gate (I7): the VIOLATED presets that `secure_diff`
    # derives by comparing against the model's ideal safe version are assumptions:
    # they only state that the original code does not look like that ideal version.
    # If the proposer points out that the original code already satisfies the
    # obligation in *another form* (a candidate fact with direction=SATISFIED) and
    # that form fails the guard-shape gate of verify_mitigation because it contains
    # no branch or comparison literal, the gate takes one step back: as soon as
    # protection_exists confirms that such a protection *really exists in the
    # function, truly dominates the sink, and acts on the same object*, the
    # assumption is voided -> UNDERDETERMINED. It does not write SATISFIED:
    # equivalence is an inductive judgement of the LLM while this module only checks
    # existence, which suffices to void an assumption but not to prove safety (I3).
    # The gate is therefore one-sided: it can only turn a positive into an
    # undetermined point and can never manufacture a safe_proved.
    # A3: this gate is disabled when EQUIVALENT_PROTECTION_ENABLED=0 (the assumption
    # does not fall back; the ablation measures its marginal contribution).
    voided: set[str] = set()
    if vcfg.EQUIVALENT_PROTECTION_ENABLED:
        voided = {
            str(f.get("obligation_id", ""))
            for f in all_facts
            if str(f.get("direction", "")).upper() == "SATISFIED"
            and any(protection_exists(a, sink, consumer_source, cfg,
                                      str(f.get("derived_from") or ""))
                    for a in (f.get("anchors") or []) if isinstance(a, str))
        }
    for o in obs:
        if (o.get("status") == "VIOLATED" and o.get("id") in voided
                and str(o.get("provenance") or "") == "secure_diff"):
            o["status"] = "UNDERDETERMINED"
            o["unresolved_reason"] = (
                "assumption voided (I7): a re-checked equivalent protection already "
                "exists in the original code (it dominates the sink and acts on the "
                "same object), so secure_diff's 'missing protection' is a formal "
                "difference, not an absence"
            )
            _append_audit(o, {
                "stage": "equivalent_protection", "result": "revoked",
                "reason": o["unresolved_reason"],
            })

    # Machine-self-proved guards (independent of any LLM output): take the unsafe
    # conditions that appeared in the candidate facts and ask, one by one, whether a
    # machine-computed dominating guard contradicts them. A contradiction implies the
    # corresponding guard obligation of that sink is deterministically SATISFIED.
    # This is the deterministic exit on the patch side: a patched function is judged
    # safe because the negative cycle is replayable, not because a model said so.
    if cfg is not None and cfg.entry:
        gid = ob.guard_target(sink_type, str(sink.get("change_kind") or ""))
        target = by_id.get(gid)
        if target is not None and target.get("status") != "SATISFIED":
            for cond in _unsafe_conditions(all_facts):
                out = wit.guard_proves_safe(cfg, int(sink.get("line") or 0), cond)
                if not out.get("proved"):
                    continue
                target["status"] = "SATISFIED"
                target["provenance"] = "machine_guard"
                refs = [f"witness:guard {a}" for a in (out.get("anchors") or [])] \
                    or [f"witness:guard refutes {cond}"]
                merged = list(target.get("evidence") or [])
                for r in refs:
                    if r not in merged:
                        merged.append(r)
                target["evidence"] = merged
                _append_audit(target, {
                    "stage": "machine_guard", "result": "accepted",
                    "reason": str(out.get("reason")
                                 or "a dominating guard refutes the unsafe condition"),
                    "evidence": list(refs),
                })
                for r in refs:
                    if r not in verified_refs:
                        verified_refs.append(r)
                break
    return obs, verified_refs


def _unsafe_conditions(facts: list[CandidateFact]) -> list[str]:
    """Collect every claimed unsafe condition (the witness `unsafe` field) from the
    candidate facts, deduplicated with order preserved."""
    out: list[str] = []
    for f in facts or []:
        w = f.get("witness")
        if not isinstance(w, dict):
            continue
        cond = str(w.get("unsafe") or w.get("overflow") or "").strip()
        if cond and cond not in out:
            out.append(cond)
    return out


def void_unaudited_presets(
    obligations: list[Obligation], reason: str = "",
) -> list[Obligation]:
    """Leave an undetermined audit record for secure_diff candidates that were never
    covered (I7).

    A candidate direction never enters ``status``, so such a point is already
    UNDERDETERMINED. This function only supplies the reason for the lack of coverage,
    and keeps the compatibility path for older artifacts: a leftover secure_diff
    preset VIOLATED is still downgraded to UNDERDETERMINED.
    """
    out: list[Obligation] = []
    for o in obligations or []:
        o = dict(o)
        is_new_candidate = bool(o.get("candidate_direction")) and \
            str(o.get("candidate_provenance") or "") == "secure_diff"
        is_legacy_preset = (o.get("critical") and o.get("status") == "VIOLATED"
                            and str(o.get("provenance") or "") == "secure_diff")
        if is_new_candidate or is_legacy_preset:
            o["status"] = "UNDERDETERMINED"
            o["unresolved_reason"] = reason or (
                "secure_diff candidate was not re-checked against evidence (I7): a "
                "formal difference from the ideal safe version is not a missing guard"
            )
            _append_audit(o, {
                "stage": "coverage", "result": "unresolved",
                "reason": o["unresolved_reason"],
            })
        out.append(o)
    return out


def critical_violated(obligations: list[Obligation]) -> list[Obligation]:
    """Return the critical obligations that are VIOLATED and carry evidence (I2: the
    necessary and sufficient basis for allowing a positive verdict)."""
    return [o for o in (obligations or [])
            if o.get("critical") and o.get("status") == "VIOLATED" and o.get("evidence")]


def mitigation_proved(obligations: list[Obligation]) -> list[Obligation]:
    """Return the "mitigation guard" obligations discharged by deterministic
    re-check (SATISFIED + evidence + belonging to the guard obligations).

    `mitigation_gate`: a true obligation of this kind means that *there exists a
    guard which dominates the sink and acts on the same taint* -- the most common
    shape of a patch. It symmetrically completes the evidence half of I3: a guard
    verified by the checker can veto a positive verdict even while the
    controllability obligation (attacker_control) is still VIOLATED. The semantics
    are "controllable AND unguarded -> vulnerability"; once the guard is discharged,
    that weakness is mitigated."""
    return [o for o in (obligations or [])
            if o.get("id") in _GUARD_OBLIGATIONS
            and o.get("status") == "SATISFIED" and o.get("evidence")]


def is_vuln_provable(obligations: list[Obligation]) -> bool:
    """I2 (including the mitigation gate): a positive verdict is allowed only when a
    re-checked critical VIOLATED obligation exists and no re-checked mitigation guard
    does. Once a guard is discharged (mitigation_proved is non-empty) the weakness is
    mitigated and the positive verdict is vetoed."""
    return bool(critical_violated(obligations)) and not mitigation_proved(obligations)


def safety_proved(obligations: list[Obligation]) -> bool:
    """Positive criteria for "proven safe": all critical obligations SATISFIED with
    evidence, **or** a mitigation guard discharged by deterministic re-check. Both
    are evidence-backed positive proofs, not a default of "no vulnerability was
    found"."""
    crits = [o for o in (obligations or []) if o.get("critical")]
    if crits and all(o.get("status") == "SATISFIED" and o.get("evidence") for o in crits):
        return True
    return bool(mitigation_proved(obligations))


def resolve(obligations: list[Obligation]) -> tuple[str, str]:
    """Tri-valued adjudication: returns (resolution, reason).

    - `vuln`             a re-checked critical VIOLATED exists and no mitigation
                         guard is discharged (I2).
    - `safe_proved`      safety is positively proven (all critical SATISFIED with
                         evidence, or a mitigation guard discharged).
    - `underdetermined`  neither holds -- some critical obligation is not discharged.

    `underdetermined` is deliberately *not* folded into safe: writing a still
    UNDERDETERMINED critical obligation as SATISFIED would report "no guard was
    observed" as "the guard is satisfied", an assertion contradicting the facts in
    the file. I3 requires "proven safe" and "not proven vulnerable" to be two
    distinct conclusions; this function is where that is enforced.
    """
    if is_vuln_provable(obligations):
        return "vuln", ""
    mit = mitigation_proved(obligations)
    if mit:
        return "safe_proved", (
            f"mitigation guard discharged by deterministic re-check "
            f"({mit[0].get('provenance', '')}): {mit[0].get('predicate', '')}"
        )
    crits = [o for o in (obligations or []) if o.get("critical")]
    if crits and all(o.get("status") == "SATISFIED" and o.get("evidence") for o in crits):
        return "safe_proved", ("every critical obligation was discharged as "
                               "SATISFIED by deterministic re-check (with evidence)")
    pending = [o for o in crits if o.get("status") != "SATISFIED"]
    if not crits:
        return "underdetermined", ("no critical obligation to adjudicate "
                                   "(template missing)")
    why = "; ".join(
        f"{o.get('id')}={o.get('status')}"
        + (f" ({o.get('unresolved_reason')})" if o.get("unresolved_reason") else "")
        for o in pending
    )
    return ("underdetermined",
            f"critical obligations not discharged; the conservative reading of I3 "
            f"keeps them undetermined: {why}")


def mark_unresolved(obligations: list[Obligation], reason: str = "") -> list[Obligation]:
    """Audit trail of an `underdetermined` adjudication: mark the critical
    obligations that are still UNDERDETERMINED as UNRESOLVED and attach a reason.

    Only the status label and unresolved_reason are changed; SATISFIED is **never**
    written."""
    out: list[Obligation] = []
    for o in obligations or []:
        o = dict(o)
        if o.get("critical") and o.get("status") == "UNDERDETERMINED":
            o["status"] = "UNRESOLVED"
            if reason and not o.get("unresolved_reason"):
                o["unresolved_reason"] = reason
        out.append(o)
    return out
