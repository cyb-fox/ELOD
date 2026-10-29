"""Centralized prompt templates.

Each LLM node exposes a ``SYSTEM`` prompt and a ``build_user`` helper. System
prompts define the role and hard constraints; user prompts contain only the
current input data. All model responses must be strict JSON and are parsed by
the deterministic ``complete_json`` helper.
"""
from __future__ import annotations

import json
from typing import Any

# The runtime whitelist is injected by registry; this module intentionally has
# no dependency on the knowledge package.


def _brief(obj: Any, limit: int = 1600) -> str:
    """Compactly serialize knowledge fields so prompts stay within budget."""
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + " ...(truncated)"


# ---------- family knowledge card block (the one shape both backends share) ----------
# Uniform rule: **a section is present iff the data is present**. With no card
# the whole section is not emitted; within a card, an empty field omits its line.
# Empty sections and empty `[]` lists are never emitted -- an empty list reads to
# the model as "this family has no safe form", which is the opposite implication.
_FAMILY_LIST_CAP = 3          # max list entries per card
_FAMILY_EXAMPLE_CAP = 2       # max vuln/safe pairs per card
FAMILY_BLOCK_BUDGET = 4200    # character budget for card details (family name +
                               # one-line description are not bound by it)

_FAMILY_HEAD = (
    "Vulnerability family reference (background knowledge; a hint for where to look, "
    "never a verdict and never exhaustive):\n"
)


def _family_lines(card: dict, full: bool) -> list[str]:
    """Render one card (a missing field omits its whole line). With full=False
    only the obligation and the trigger shape are kept."""
    out: list[str] = []
    for key, label in (("principle", "obligation"), ("trigger", "typical trigger")):
        value = card.get(key)
        if isinstance(value, str) and value.strip():
            out.append(f"    {label}: {value.strip()}")
    if not full:
        return out
    examples = card.get("examples")
    if isinstance(examples, list):
        for ex in examples[:_FAMILY_EXAMPLE_CAP]:
            if not isinstance(ex, dict):
                continue
            vuln = str(ex.get("vuln") or "").strip()
            safe = str(ex.get("safe") or "").strip()
            if vuln and safe:
                out.append(f"    vulnerable: {vuln}")
                out.append(f"    fixed:      {safe}")
    for key, label in (
        ("patch", "standard fix"),
        # safe_pattern is "the shape that already satisfies the obligation" and
        # serves only as an inductive clue: SATISFIED must still pass the
        # checker's anchor/witness channel, so the wording must rule out
        # "looks right therefore it is safe".
        ("safe_pattern", "shapes that already satisfy it (must still be quoted from source)"),
    ):
        items = card.get(key)
        if isinstance(items, list):
            picks = [str(i).strip() for i in items[:_FAMILY_LIST_CAP] if str(i).strip()]
            if picks:
                out.append(f"    {label}: " + "; ".join(picks))
    return out


def family_block(family_context: list[dict] | None,
                 budget: int = FAMILY_BLOCK_BUDGET) -> str:
    """Render the family knowledge cards into a prompt section; returns "" when
    there are no cards (the whole section is not emitted).

    Three degradation tiers, filled greedily in registry order (first come first
    served, the remaining budget goes to the next one): full card (obligation +
    trigger + contrast examples + fix means + safe shapes) -> brief card
    (obligation + trigger) -> name + one-line description only. A card is never
    truncated halfway -- half an example is worse than none. Registry order
    approximates prior frequency (memory-safety / numeric-errors first, the bulk
    of training labels), so greedy filling gives the full card to the most likely
    relevant families; when all 9 families pass the threshold (routing degenerates
    into no routing) the tail falls to the name tier.
    `budget` bounds only the card details; the name + description line is always
    emitted (~150 chars per family, ~1.3k for 9 families), so the whole section
    has a hard upper bound of ~5.5k chars ~= 1.4k token.

    Order is the input order (the node side already normalises via
    `family_names()`), so the same family_context renders byte-identically twice
    -- probability jitter must not change the prompt text.
    """
    if not family_context:
        return ""
    cards = [c for c in family_context
             if isinstance(c, dict) and str(c.get("family") or "").strip()]
    if not cards:
        return ""
    rendered: list[str] = []
    left = budget - len(_FAMILY_HEAD)
    for card in cards:
        name = str(card.get("family")).strip()
        desc = str(card.get("description") or "").strip()
        head = f"- {name}" + (f": {desc}" if desc else "")
        block = head
        for full in (True, False):
            candidate = "\n".join([head] + _family_lines(card, full))
            if len(candidate) <= left:
                block = candidate
                break
        left -= len(block) + 1
        rendered.append(block)
    return _FAMILY_HEAD + "\n".join(rendered) + "\n\n"


# ========== Module 2: select_knowledge (two-level menu + mechanism hypotheses) ==========
# The former family_triage, critique_family, and load_knowledge nodes are now
# one call. The deterministic recall floor remains in the node implementation.
# Feed source directly rather than an operation-only pre-scan: uninitialized,
# lifetime, and cleanup defects may have no sensitive API in their summary.
SELECT_KNOWLEDGE_SYSTEM = (
    "You are a vulnerability-knowledge loader. You receive numbered source code and a two-level "
    "menu: top-level vulnerability families -> member CWEs (ID and name). Complete two tasks.\n"
    "1) picks: use the menu to select CWEs whose knowledge should be loaded for review. Group by "
    "family, choose at most 4 per family, and select only member IDs with source-based reasons.\n"
    "The test is not whether code resembles a dangerous API, but whether it violates a security "
    "obligation. Any data or resource use that relies on an unproven precondition is a suspicion. "
    "Typical violations include: a value is used without proving its range, length, sign, non-null, "
    "initialization, liveness, ownership, or trusted origin; a resource is not acquired, released, "
    "or exclusively accessed correctly; data crossing a trust boundary is not neutralized before an "
    "interpreter, path, or memory operation; a required check, authorization, lock, or error path is "
    "missing or bypassable; or producer and consumer assumptions about state disagree.\n"
    "Ask first: what assumption is implicitly required here but not proved by the code? Then ask: "
    "what happens if that assumption is false? Do not limit suspicions to memory bugs or menu entries.\n"
    "2) suspicions: propose at most 4 mechanism hypotheses with source anchors. Each must quote one "
    "single source line and include line, subject, and required_condition. mechanism must be one of: "
    "memory_extent, allocation_size, numeric_arithmetic, type_conversion, null_or_failure, "
    "object_lifetime, command_input, path_input, format_input, query_input, initialization_cleanup, "
    "state_contract, trust_boundary, concurrency_order. candidate_cwe may be empty: describe the "
    "mechanism and missing precondition even when classification is uncertain. This step optimizes "
    "recall; downstream deterministic checks independently verify every hypothesis.\n"
    'Output strict JSON: {"picks":{"memory-safety":["CWE-119"]},"suspicions":[{"line":42,'
    '"anchor":"mixer->isActive()","mechanism":"object_lifetime","subject":"mixer",'
    '"required_condition":"mixer remains live before this call","candidate_cwe":"CWE-416"}]}'
)


def select_knowledge_user(
    numbered_source: str, top_menu: list[str],
    family_blocks: list[tuple[str, list[tuple[str, str]]]],
) -> str:
    blocks = []
    for fam, members in family_blocks:
        lines = "\n".join(f"    - {cid} - {name}" if name else f"    - {cid}"
                          for cid, name in members)
        blocks.append(f"  [{fam}]\n{lines}")
    return (
        f"Source under review (numbered):\n----\n{numbered_source}\n----\n\n"
        f"Top-level families:\n" + "\n".join(f"  - {m}" for m in top_menu) + "\n\n"
        f"Two-level menu (picks must come from the matching family):\n" + "\n".join(blocks) + "\n\n"
        "Return evidence-backed picks (possibly empty) and up to 4 mechanism hypotheses, each with "
        "a verbatim single-line anchor.\n"
        '{"picks":{"<family>":["CWE-xxx"]},"suspicions":[{"line":1,"anchor":"...",'
        '"mechanism":"null_or_failure","subject":"p","required_condition":"p is non-null",'
        '"candidate_cwe":""}]}'
    )


# ========== Module 5: secure_diff (candidate gate and counterfactual safe form) ==========
# The model proposes a location and a safe form. Code compares it with the
# source truth and decides whether a real difference exists.
SECURE_DIFF_SYSTEM = (
    "You are a secure code comparator. Read numbered C/C++ source and identify real, exploitable "
    "security defects. For each, give the absolute line and a safe replacement.\n"
    "[Signal-driven, family-agnostic] CWE IDs are hints only. Follow data flow when a signal appears; "
    "use the closest CWE (leave it empty if uncertain):\n"
    "  - arithmetic (multiply/add/shift) -> overflow, wraparound, or underflow, especially in allocation, dimensions, indices, or loop bounds;\n"
    "  - cross-type assignment or argument passing -> narrowing (int64_t/size_t to int), signedness mismatch, or negative-to-unsigned conversion;\n"
    "  - memcpy, loop writes, indexed writes, pointer arithmetic -> bounds, off-by-one, length, or offset problems;\n"
    "  - pointer/handle-returning calls (malloc, lookup, cast) -> NULL or failure used without checking;\n"
    "  - free/delete/destructor/ownership transfer -> use-after-free, double-free, or access after release;\n"
    "  - first read of a variable -> an uninitialized path;\n"
    "  - error branch/early return/goto cleanup -> leak, incomplete cleanup, or inconsistent error state;\n"
    "  - division/modulo -> divide by zero;\n"
    "  - check-then-use or concurrent shared-state access -> race, TOCTOU, or non-atomic read-modify-write;\n"
    "  - CHECK/abort/assert receiving external input -> assertion-based DoS;\n"
    "  - packet/file parser using its own length/count -> an untrusted field used for copy or allocation;\n"
    "  - cross-function call -> contract mismatch or an illegal state-machine order.\n"
    "[Unified criterion] A true defect requires an externally controllable risk value, no equivalent "
    "protection, and an externally reachable path; confirm all three by tracing data flow. Concrete "
    "harms include memory corruption, injection/traversal/authorization bypass/disclosure, overflow, "
    "CHECK/abort DoS, and uncontrolled allocation. Do not report generic missing defenses when data is "
    "uncontrollable, harm is unclear, or the path is unreachable.\n"
    "[Look for existing protection first] Before reporting a missing defense, inspect upward in the "
    "function for an equivalent obligation: status-return checks, construction/conversion validation, "
    "bounded APIs, assertions, or values excluded by types/constants all count. Trace derived values "
    "back to their source (N=shape(0), p=&buf[k]); protection on the source protects the derived value.\n"
    "[Evidence is also required for suppression] If you call a point safe because of an existing "
    "defense, provide defense_at={line,snippet} with at least 8 real source characters. An unanchored "
    "defense cannot suppress a finding. An anchored defense is a hard stop; do not report it just in "
    "case. If the patched version has no difference, return [].\n"
    "[Required fields] line=absolute source line; change_kind=guard|contract|type_fix|api_swap "
    "(guard=missing validation; contract=contract mismatch; type_fix=narrowing/signedness; "
    "api_swap=replace with a Status-returning safe API); original=verbatim location snippet from that "
    "line, at least 8 characters, preserving identifiers; secure=the expected safe form and different "
    "from source; mechanism_kind is one of memory_extent, allocation_size, numeric_arithmetic, "
    "type_conversion, null_or_failure, object_lifetime, command_input, path_input, format_input, "
    "query_input, initialization_cleanup, state_contract, trust_boundary, concurrency_order, or empty; "
    "cwe_id is optional root-cause attribution; threat={source,harm,reachable} must name real source "
    "identifiers; rationale is at most 2 sentences. Use anchors [{line,original,secure}] for multi-line "
    "defects. Contract changes also include contract_witness={precondition,expected_behavior,"
    "actual_behavior,security_consequence,counterexample,producer_evidence,consumer_evidence,evidence_tokens}.\n"
    "Output a strict JSON array, for example:\n"
    '[{"line":80,"change_kind":"type_fix","cwe_id":"CWE-190",'
    '"original":"int new_num_elements = 1;","secure":"int64_t new_num_elements = 1;",'
    '"anchors":[{"line":82,"original":"MultiplyWithoutOverflow(new_num_elements",'
    '"secure":"accumulate in int64_t without truncating the 64-bit product into int"}],'
    '"threat":{"source":"input_shape_vec from user sparse_shape",'
    '"harm":"truncated dimension product bypasses overflow checks and causes a TensorShape CHECK failure (DoS)",'
    '"reachable":"AddManySparseToTensorsMap reaches this on every Compute"},'
    '"rationale":"A 64-bit product is truncated to 32 bits, disabling overflow detection"}]'
    " (return [] when there is no defect)."
)


def secure_diff_user(
    numbered_source: str, cwe_ids: list[str],
    negative_signals: Any, safe_example: Any,
    suspicions: list[dict] | None = None,
    family_context: list[dict] | None = None,
) -> str:
    """Under the clf backend cwe_ids / negative_signals / safe_example are all
    empty and the family block appears alone; under the llm backend all three are
    present and the family block is a pure increment (one extra section, the rest
    unchanged). Empty sections are never emitted."""
    hint = (
        "Reconnaissance hypotheses (entry points only; re-check each and never treat them as findings or evidence):\n"
        f"{_brief(suspicions, 1800)}\n\n"
        if suspicions else ""
    )
    loaded = f"Loaded CWEs (not exclusive): {cwe_ids}\n\n" if cwe_ids else ""
    negs = f"Negative signals for safe code: {_brief(negative_signals)}\n" if negative_signals else ""
    safe = f"Safe example: {_brief(safe_example, 600)}\n" if safe_example else ""
    tail = "\n" if (negs or safe) else ""
    return (
        f"{family_block(family_context)}"
        f"{loaded}"
        f"Source to compare (absolute line numbers):\n----\n{numbered_source}\n----\n\n"
        f"{hint}"
        f"{negs}{safe}{tail}"
        "For each suspicion, first look upward in the function for an equivalent defense, including "
        "derived-value tracing and status-return checks. A safe judgment requires an auditable "
        "defense_at anchor. Report only points with no defense, using the required schema. Return [] for a patched file."
    )


# ========== Module 6: risk_forensics (evidence loop, function-level aggregation) ==========
FORENSICS_SYSTEM = (
    "You are a security forensic analyst processing a batch of risk points, possibly across functions. "
    "Submit only candidate facts with verbatim anchors; do not decide the verdict. A deterministic "
    "checker validates anchors and decides whether each obligation holds.\n"
    "[Neutral starting point] A line selected by the safe-baseline diff means only that an obligation "
    "may matter there. It is a hypothesis, not proof that protection is missing. Do not assume either "
    "absence or hardening.\n"
    "For every risk point answer three questions:\n"
    "  1) Where does the taint originate? Quote the source line. If it comes from an opaque external "
    "call, that line proves only that the implementation is unseen, not controllability or out-of-bounds "
    "behavior; use cross-file anchors or a witness, otherwise leave it unresolved.\n"
    "  2) Before the dangerous operation, is there a dominating defense applied to the same taint or its "
    "derived source? Any equivalent form counts: NULL/bounds/overflow checks, nulling after release, "
    "locking, checked status returns, construction-time validation, bounded APIs, assertions, or values "
    "excluded by types. If yes, submit direction=SATISFIED, quote the defense line, and set derived_from "
    "for a derived value.\n"
    "  3) If no defense exists, submit direction=VIOLATED. Anchors must be independent facts outside the "
    "risk line (source line, missing-guard context, or cross-file producer file:line); never quote the "
    "accused sink as self-evidence.\n"
    "[Uncertainty stays unresolved] Not seeing a caller or callee is not a reason to call code safe. "
    "Keep screen=keep and submit nothing unless you can quote concrete lines.\n"
    "[Allowed obligations] obligation_id must be one of attacker_control, non_negative_length, "
    "readable_extent, size_overflow_checked, arithmetic_bound, null_guard, object_live_at_use, "
    "input_sanitized, path_validated, format_string_fixed, query_parameterized, dominating_guard. "
    "Do not use input_sanitized for integer, size, or allocation defects.\n"
    "[Machine-checkable witness, required for local VIOLATED] Each risk point includes branch structure (node id@line + condition). "
    "For a vulnerability, prefer witness={\"sink_line\":line,\"path\":[{\"node\":\"n3\",\"take\":false}],"
    "\"precondition\":\"entry condition such as len>64\",\"unsafe\":\"machine-checkable expression such as n>64 or p==NULL\"}. "
    "path contains only node IDs and directions; unsafe uses comparison operators (< <= > >= == !=), "
    "optionally with constants, but no multiplication, division, bit operations, or prose. A witness "
    "does not fall back to the anchor channel.\n"
    "[SATISFIED also needs unsafe] When claiming existing protection, fill witness={\"unsafe\":\"the condition excluded by that defense\"}; "
    "for example n>cap for if(n>cap) return, or p==NULL for a non-null check. path/precondition may be omitted. "
    "The checker uses machine-computed dominating guards to refute the unsafe condition. Successful refutation "
    "is the only path to a proved-safe result; without unsafe, leave the point unresolved.\n"
    "[One finding per sink] Emit exactly one finding for every sink_id below; do not omit or merge points.\n"
    "[Concise rationale] rationale is at most 2 sentences and must be fully supported by rationale_anchors or "
    "machine_guard_facts. Anchors must quote source text with line numbers; otherwise leave rationale empty. "
    "screen is a human-readable label: exclude only when a defense covers the point or it clearly misses every "
    "danger signal; otherwise keep. contract_witness is only a hypothesis, and contract_verified=true requires "
    "independent evidence for both the risk window and search results.\n"
    '{"findings":[{"sink_id":"...","hit_signals":["..."],'
    '"contract_verified":false,"contract_evidence":[],'
    '"candidate_facts":[{"obligation_id":"size_overflow_checked","direction":"SATISFIED",'
    '"anchors":["verbatim defense line"],"cross_file":false,"derived_from":"shape_vec",'
    '"witness":null}],"screen":"keep","rationale":"anchored reasoning (possibly empty)",'
    '"rationale_anchors":[{"line":80,"text":"verbatim source snippet"}]}],'
    '"need_more":{"tokens":["var"]}}'
    " (set need_more to null when no search is needed; use [] for no candidate facts; omit derived_from when not applicable)."
)


def _sinks_brief(sinks_brief: list[dict]) -> str:
    lines = []
    for s in sinks_brief:
        cfg = s.get("cfg_brief") or ""
        guard_facts = s.get("machine_guard_facts") or "(no machine guard facts supplied)"
        lines.append(
            f"- sink_id={s.get('sink_id')!r} function={s.get('func_name','')} line={s.get('line')} "
            f"kind={s.get('change_kind','guard')} tainted={s.get('tainted_tokens')}\n"
            f"    source: {s.get('snippet','')}\n"
            + (f"    branch structure (machine-computed; cite these node IDs in witnesses): {cfg}\n" if cfg else "")
            + f"    machine_guard_facts (local machine facts only, not a safety verdict): {guard_facts}\n"
            + f"    contract_witness: {_brief(s.get('contract_witness'))}"
        )
    return "\n".join(lines)


def forensics_user(
    sinks_brief: list[dict], detection_signals: Any,
    negative_signals: Any, search_summary: str, rounds_left: int,
    family_context: list[dict] | None = None,
) -> str:
    """Under the clf backend det/negs are empty and forensics relies on the family
    cards plus machine facts; under the llm backend both are present."""
    ctx = (f"search_context from the previous round:\n{search_summary}\n\n"
           if search_summary else "(search_context has not been called yet)\n\n")
    more = (f"You may request {rounds_left} more search_context round(s)." if rounds_left > 0
            else "The search limit is reached; provide a conclusion now and set need_more to null.")
    det = f"Danger signals (detection_signals): {_brief(detection_signals)}\n" if detection_signals else ""
    negs = f"Safe-code signals (negative_signals): {_brief(negative_signals)}\n" if negative_signals else ""
    tail = "\n" if (det or negs) else ""
    return (
        f"{family_block(family_context)}"
        f"Risk points to investigate one by one (do not omit; may span functions):\n{_sinks_brief(sinks_brief)}\n\n"
        f"{det}{negs}{tail}"
        f"{ctx}"
        "For each point, trace whether a dominating defense covers it (trace derived values and set "
        "derived_from). If so, return SATISFIED with a verbatim defense line; otherwise judge whether the "
        "source exhibits the obligation violation described for its family and, if it does, submit "
        "VIOLATED with independent anchors. Existing defenses may be quoted only from source, "
        "search_context, or "
        "machine_guard_facts. Do not invent checks, macros, constants, or ranges. Empty machine_guard_facts "
        "means that no local machine fact was extracted, never that the code is safe. Keep the obligation "
        "consistent with the defect (do not use input_sanitized for size/integer defects). If the source is "
        "unseen, leave the point unresolved rather than calling it safe.\n"
        f"{more}"
    )


# ========== Module 9: project_dataflow (cross-file taint chain, project mode only) ==========
# This node proposes cross-file candidate facts. obligation_checker validates
# their anchors and remains the only writer of obligation status.
PROJECT_DATAFLOW_SYSTEM = (
    "You propose cross-file execution certificates, never verdicts. The checker replays trusted C "
    "source and independently computes calls, argument-to-parameter mappings, returns, assignments "
    "and taken guards. Quotes, claimed paths and provenance cannot authorize a finding.\n"
    "Supported certificates use version=1, entry={file: exact supplied relative file, function: "
    "exported C function}, inputs=[concrete signed 32-bit integers in parameter order]. A null int* "
    "parameter may receive 0. Entry inputs are conditional API inputs, not proof of whole-program "
    "or attacker reachability. Use only arithmetic_bound for an actual division/modulo by zero, "
    "or null_guard for an actual null int* dereference. direction must be VIOLATED.\n"
    "Supported replay: direct functions, int locals, assignments, arithmetic, if/else and returns. "
    "Macros/includes, loops, indirect/ambiguous calls, globals, casts, unknown effects, object aliases "
    "and earlier overflow cause rejection. A guard that intercepts the concrete input rejects the "
    "certificate. Return [] for unsupported or insufficient evidence; a blocked example is not a "
    "universal safety proof. The actual unsafe operand must cross a file boundary through a "
    "parameter or return value; an unrelated call does not count.\n"
    "Use only supplied sink IDs. Output a strict JSON array such as: "
    '[{"sink_id":"consume:4","obligation_id":"arithmetic_bound","direction":"VIOLATED",'
    '"cross_file":true,"cwe_id":"CWE-369","anchors":[],'
    '"cross_file_certificate":{"version":1,"entry":{"file":"entry.c","function":"entry"},'
    '"inputs":[0]}}]'
)



def project_dataflow_user(
    current_file: str, candidates: list[dict], cross_evidence: str,
    producers: str, loaded_cwes: list[str],
    family_context: list[dict] | None = None,
) -> str:
    cand_lines = []
    for c in candidates:
        cand_lines.append(
            f"- sink_id={c.get('sink_id')!r} line={c.get('line')} "
            f"tainted={c.get('tainted_tokens')} single-file status={c.get('prior')}\n"
            f"    source: {c.get('snippet','')}"
        )
    # The cross-file final review only needs family names for attribution
    # reference; there is no need to lay out the full cards again (the evidence
    # is in cross_evidence).
    fams = [str(c.get("family")) for c in (family_context or [])
            if isinstance(c, dict) and c.get("family")]
    ref = fams or loaded_cwes
    reference = (f"Loaded family reference: {ref} (cwe_id is not limited to these).\n\n"
                 if ref else "")
    return (
        f"Current file: {current_file}\n\n"
        f"Cross-file risk points to confirm (taint comes from outside):\n" + "\n".join(cand_lines) + "\n\n"
        f"Cross-file evidence (file:line):\n----\n{cross_evidence or '(no cross-file matches)'}\n----\n\n"
        f"Producer/propagator snippets (file:line):\n----\n{producers or '(none)'}\n----\n\n"
        f"{reference}"
        "Submit a concrete cross_file_certificate only for a supported unsafe execution. "
        "The checker reconstructs the chain from source. Blocked or unsupported examples stay unresolved."

    )
