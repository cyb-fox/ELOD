"""select_knowledge (knowledge loading) -- three backends, defaulting to the
local family classification head.

**clf (default, zero token)**: `tools/select_classifier` reads the whole source
text and outputs the family names above threshold; what gets loaded are **family
knowledge cards** (`registry.family_card`: principle / trigger / examples /
patch / safe_pattern) instead of a per-CWE signal list.  `knowledge` is therefore
`{}` by design, and **this is not a degradation**: the invariant is restated
from "never let knowledge go empty" as "never let the family list go empty".

**llm (`SELECT_KNOWLEDGE_BACKEND=llm`)**: one call feeds a two-level menu
(top-level family -> member CWEs); the model reads the code, selects the
specific CWEs whose knowledge is needed, and writes down every suspicion it
noticed while reading (`suspicions`, passed through to secure_diff for deeper
digging).

**none (`SELECT_KNOWLEDGE_BACKEND=none`, K ablation)**: neither the classifier
head nor an LLM is run.  The family list comes only from the rule floor plus the
`MAX_FAMILIES` backstop, and **must not go empty**.  `=llm` is a different route,
not this one.

The three backends **share no helper**: `_select_via_llm` keeps its own
selection code.  The switch exists so that the other route stays reproducible,
and a shared helper is exactly how it would be silently modified.

The two backends differ in more than "who picks the family":
- `family_candidates` has different semantics -- clf yields **family names**,
  llm yields **CWE ids** (the CWE-id semantics are what the training-corpus
  builder depends on when it intersects the value with ground-truth CWEs;
  switching to family names would silently drive the hit rate to a constant 0,
  and the collect stage of that builder must in any case run with `=llm`);
- the clf path makes no positional assumption (masked average pooling erases
  position, the head is [9,768], and hypotheses were never a training target)
  -> `_triage_hypotheses` is always empty and the zero-sink bridge of
  secure_diff is consequently inactive;
- on the clf path the downstream prompts carry only family cards, with no
  CWE-level detection_signals / negative_signals.

Deterministic invariants (recall-first, independent of the prompt, present in
both backends):
- rule floor: CWEs confirmed by `families_from_ops` are merged in
  unconditionally (the clf path folds them to family granularity);
- empty backstop: when the candidate list is empty, fall back to the first
  `MAX_FAMILIES` families; the family list never goes empty.

On budget overrun -> raise BudgetExceeded, which graph routes at the top level
to output.  When the clf path makes zero calls `budget_guard` is a no-op, but
`=llm` needs it, so that wrapper is kept in graph.
"""
from __future__ import annotations

import re

from .. import config
from ..knowledge import (
    cwe_family,
    family_card,
    family_member_menu,
    family_names,
    load_cwe,
    top_family_menu,
    whitelist,
)
from ..llm import prompts as P
from ..state import VulnState
from ..tools import select_classifier as clf
from ..tools import sink_locator as sl
from ..tools.segmenter import strip_line_numbers
from ._common import bounded_source, make_llm

# Cap on how many CWEs are kept per family (node-local, not a config item).
_TOP_K = 4
_MIN_ANCHOR_CHARS = 8
_MECHANISMS = frozenset({
    "memory_extent", "allocation_size", "numeric_arithmetic", "type_conversion",
    "null_or_failure", "object_lifetime", "command_input", "path_input",
    "format_input", "query_input", "initialization_cleanup", "state_contract",
    "trust_boundary", "concurrency_order",
})
_CWE_RE = re.compile(r"CWE-?(\d+)$", re.IGNORECASE)


def _norm(text: object) -> str:
    return " ".join(str(text or "").split())


def _mechanism(raw: object) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return value if value in _MECHANISMS else ""


def _cwe(raw: object) -> str:
    match = _CWE_RE.fullmatch(str(raw or "").strip())
    return f"CWE-{match.group(1)}" if match else ""


def _validated_hypotheses(raw: object, source: str) -> list[dict]:
    """Forward only compact, single-line hypotheses anchored to the displayed source."""
    if not isinstance(raw, list):
        return []
    lines = source.splitlines()
    out: list[dict] = []
    for item in raw[:4]:
        if not isinstance(item, dict):
            continue
        try:
            line = int(item.get("line", 0))
        except (TypeError, ValueError):
            continue
        anchor = str(item.get("anchor") or "").strip()
        mechanism = _mechanism(item.get("mechanism"))
        required = str(item.get("required_condition") or "").strip()
        if not (mechanism and required and 1 <= line <= len(lines)):
            continue
        if len(_norm(anchor)) < _MIN_ANCHOR_CHARS or _norm(anchor) not in _norm(lines[line - 1]):
            continue
        hypothesis = {"line": line, "anchor": anchor, "mechanism": mechanism,
                      "subject": str(item.get("subject") or "").strip(),
                      "required_condition": required}
        candidate_cwe = _cwe(item.get("candidate_cwe"))
        if candidate_cwe:
            hypothesis["candidate_cwe"] = candidate_cwe
        out.append(hypothesis)
    return out


def _backend() -> str:
    """Current backend: explicit llm / none; everything else (including a
    misspelling) falls to clf."""
    value = str(config.SELECT_KNOWLEDGE_BACKEND).strip().lower()
    if value == "llm":
        return "llm"
    if value in ("none", "off", "fallback"):
        return "none"
    return "clf"


def _family_context(families: list[str]) -> list[dict]:
    """Turn a list of family names into the family knowledge cards used by
    downstream prompts (order follows the input order)."""
    return [family_card(fam) for fam in families]


def select_knowledge(state: VulnState) -> dict:
    backend = _backend()
    if backend == "llm":
        return _select_via_llm(state)
    if backend == "none":
        return _select_via_fallback(state)
    return _select_via_classifier(state)


def _select_via_fallback(state: VulnState) -> dict:
    """K ablation: no classifier head, only the rule floor plus the empty
    backstop.  The family list does not go empty."""
    fams_all = family_names()
    picked: list[str] = []
    for cwe_root in sl.families_from_ops(state["source_text"], state["functions"]):
        fam = cwe_family(cwe_root)
        if fam and fam in fams_all and fam not in picked:
            picked.append(fam)
    if not picked:
        picked = list(fams_all[: config.MAX_FAMILIES])
    selected = [fam for fam in fams_all if fam in picked]
    degraded = list(state.get("degraded") or [])
    degraded.append("select_knowledge: family routing node disabled, "
                    "using the rule floor / backstop")
    return {
        "family_candidates": selected,
        "knowledge": {},
        "family_context": _family_context(selected),
        "_triage_hypotheses": [],
        "degraded": degraded,
    }


def _select_via_classifier(state: VulnState) -> dict:
    """Local family classification head path (default, zero token)."""
    fams_all = family_names()
    # The training text is the line-number-free form of bounded_source, and
    # inference must use the same source and the same shape (the classification
    # head is sensitive to the first 512 tokens; strip_line_numbers deliberately
    # preserves the separating spaces, see the comment in segmenter).
    text = strip_line_numbers(bounded_source(state))
    result = clf.select_families(text)

    degraded = list(state.get("degraded") or [])
    picked: list[str] = []
    if result["ok"]:
        picked = list(result["families"])
        if result["truncated"]:
            # 512 tokens is about 51 numbered lines of C code, and the training
            # side is right-truncated the same way, so this **does not change
            # behavior**; it only makes "the family was decided from roughly the
            # first 50 lines of the file" visible in the report.
            degraded.append(
                f"select_knowledge: source exceeds the classification head input"
                f" limit ({config.SELECT_CLF_MAX_LEN} tokens), families were"
                " decided from the prefix only"
            )
    else:
        degraded.append(
            f"select_knowledge: family classifier unavailable"
            f" ({result['reason']}), using the rule floor / backstop"
        )

    # (1) Rule floor: CWEs confirmed by the operations are folded to family
    # granularity and merged in (points already confirmed by rules must not be
    # missed).
    for cwe_root in sl.families_from_ops(state["source_text"], state["functions"]):
        fam = cwe_family(cwe_root)
        if fam and fam in fams_all and fam not in picked:
            picked.append(fam)

    # (2) Family list backstop: classifier down and the floor empty -> the first
    # MAX_FAMILIES families.
    if not picked:
        picked = list(fams_all[: config.MAX_FAMILIES])

    # (3) Fixed-order normalization: reorder by family_names() (not by
    # prediction order) so that the downstream prompts of two runs over the same
    # file are byte-identical -- small probability jitter must not change the
    # prompt text.
    selected = [fam for fam in fams_all if fam in picked]

    return {
        "family_candidates": selected,   # clf path: family names (llm path: CWE ids)
        "knowledge": {},                 # empty by design: family-granularity loading
        "family_context": _family_context(selected),
        "_triage_hypotheses": [],        # explicitly empty: makes the disappearance
                                         # of the suspicion channel visible in the
                                         # state dump
        "degraded": degraded,
    }


def _select_via_llm(state: VulnState) -> dict:
    """LLM path (`SELECT_KNOWLEDGE_BACKEND=llm`): per-CWE knowledge loading, with
    family_context appended at the end."""
    llm = make_llm(state, "select_knowledge")
    fams = family_names()
    menus: dict[str, list[tuple[str, str]]] = {
        fam: family_member_menu(fam) for fam in fams
    }
    wl = set(whitelist())
    # Feed the source text directly: the operation digest from the rule pre-scan
    # would show a defect without sensitive APIs as "no sensitive call", which
    # hints in reverse that the LLM is safe and leads it to under-select.  Huge
    # files go through the structured digest (bounded_source).
    numbered = bounded_source(state)

    data = llm.complete_json(
        P.SELECT_KNOWLEDGE_SYSTEM,
        P.select_knowledge_user(
            numbered, top_family_menu(),
            [(fam, menus[fam]) for fam in fams if menus[fam]],
        ),
    )
    picks: dict = {}
    hypotheses: list[dict] = []
    if isinstance(data, dict):
        if isinstance(data.get("picks"), dict):
            picks = data["picks"]
        hypotheses = _validated_hypotheses(data.get("suspicions"), state["source_text"])
    degraded = list(state.get("degraded") or [])
    if not isinstance(data, dict):
        # Without a usable output the empty backstop below necessarily runs (the
        # family list goes from "the few entries the model picked" to
        # MAX_FAMILIES x TOP_K, i.e. dozens), so families_considered differs
        # drastically between two runs over the same file.  Adjudication does
        # not change because of this (the backstop only loads more knowledge),
        # but it must be recorded, otherwise the difference cannot be explained.
        degraded.append("select_knowledge: LLM output unavailable, "
                        "family list falls back to the empty backstop")

    # LLM picks filtered to that family's members ∩ whitelist, truncated to top-K
    # per family.
    selected: list[str] = []
    for fam in fams:
        members = {cid for cid, _ in menus[fam]}
        raw = picks.get(fam) if isinstance(picks.get(fam), list) else []
        chosen = [c for c in raw if isinstance(c, str) and c in members and c in wl]
        for c in list(dict.fromkeys(chosen))[:_TOP_K]:
            if c not in selected:
                selected.append(c)

    # Rule floor: CWEs confirmed by the operations are merged in unconditionally
    # (LLM selection must not miss points already confirmed by rules).
    for cwe_root in sl.families_from_ops(state["source_text"], state["functions"]):
        if cwe_root in wl and cwe_root not in selected:
            selected.append(cwe_root)

    # Empty backstop: the last line of defence when parsing fails and the rule
    # floor is empty; knowledge never goes empty.
    if not selected:
        for fam in fams[: config.MAX_FAMILIES]:
            for cid, _ in menus[fam][:_TOP_K]:
                if cid in wl and cid not in selected:
                    selected.append(cid)

    knowledge: dict = {}
    for cwe_id in selected:
        d = load_cwe(cwe_id)
        if d is not None:
            knowledge[cwe_id] = d
    # Family cards: fold the selected CWEs back to families, deduplicate, and
    # sort by family_names().  Downstream therefore has a single prompt shape
    # with no branching on the backend name (the llm path is a pure increment:
    # one extra family block, everything else unchanged).
    fams_hit = {f for f in (cwe_family(c) for c in selected) if f}
    family_context = _family_context([f for f in fams if f in fams_hit])
    # Mechanism hypotheses serve only as entry leads for secure_diff; no sink is
    # produced here and no obligation status is advanced.
    return {"family_candidates": selected, "knowledge": knowledge,
            "family_context": family_context,
            "_triage_hypotheses": hypotheses, "degraded": degraded}
