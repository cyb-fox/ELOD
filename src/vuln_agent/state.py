"""The single State definition.

Convention: State stores only structured summaries and risk fragments; function
bodies live in units (in-memory is enough at single-file scale, nothing is
persisted to disk). All substructures use TypedDict so LangGraph can merge and
serialize them.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict


class Func(TypedDict):
    func_name: str
    line_start: int  # 1-based, inclusive
    line_end: int    # 1-based, inclusive
    lang: Literal["c", "cpp"]


class Unit(TypedDict):
    func_name: str
    kind: Literal["full", "compressed"]
    content: str  # line-numbered original function text or structured compressed body


class ContractWitness(TypedDict, total=False):
    """Reviewable evidence for a contract-type discrepancy; describes the
    behavioural conflict and does not restrict the fix shape to a new guard."""
    precondition: str
    expected_behavior: str
    actual_behavior: str
    security_consequence: str
    counterexample: str
    producer_evidence: str
    consumer_evidence: str
    evidence_tokens: list[str]


# Security-proof obligation status (S1): UNDERDETERMINED not proved (read
# conservatively per I3); SATISFIED holds with evidence; VIOLATED violated with
# evidence (a vulnerability); UNRESOLVED the trace left behind when a positive call
# was downgraded (lacking independent evidence).
ObligationStatus = Literal["UNDERDETERMINED", "SATISFIED", "VIOLATED", "UNRESOLVED"]


class Obligation(TypedDict, total=False):
    """A security proof obligation that source evidence can verify (the atomic
    unit carrying S1's proved-status)."""
    id: str                 # stable slug, e.g. "attacker_control"
    predicate: str          # human-readable, e.g. "readable_bytes(ptr) >= length"
    status: ObligationStatus
    evidence: list[str]     # source-level evidence refs, e.g. ["network.c:84"]
    provenance: str         # "local_guard"|"cross_file_dataflow"|"direct_caller"|...
    critical: bool          # whether this is a key obligation for a positive call
    unresolved_reason: str  # why unresolved, when status is UNRESOLVED
    # Structured review trace that does not participate in the verdict: CFG
    # availability / degradation reasons plus candidate accept, reject or revoke
    # events.
    verification_audit: dict
    # Proposal-layer fields: writable by secure_diff or other proposers, but not
    # consulted by the arbiter's ruling.
    candidate_direction: str
    candidate_evidence: list[str]
    candidate_provenance: str


class Witness(TypedDict, total=False):
    """Machine-checkable witness: the only vulnerability claim shape the LLM may
    submit (tools/witness.py runs a four-step deterministic check).

    Division of labour with CandidateFact: CandidateFact submits an "anchor"
    (reviewable at string level), Witness submits "path + constraint" (decidable
    at semantic level). The latter closes off shortcuts via surface statistics --
    neither length nor added-code direction can fabricate an out-of-bounds path
    that is genuinely traversable on the CFG and consistent with the guards the
    machine computes itself.
    """
    obligation_id: str
    sink_line: int              # sink line the witness points at (must be in
                                 # the CFG)
    sink_expr: str              # human-readable: the expression that goes out of bounds
    path: list[dict]            # [{node|line, take}] branch decisions, rechecked
                                 # against the CFG by the checker
    precondition: str           # entry-parameter constraint, e.g. "len > 64"
    unsafe: str                 # unsafe condition, e.g. "n > 64" / "p == NULL"
    cross_file: bool            # does the precondition depend on cross-file facts
                                 # (needs separate contract evidence)


class _SinkRequired(TypedDict):
    sink_id: str
    func_name: str
    line: int
    snippet: str
    sink_type: str            # file_op / exec / sql / mem / ...
    tainted_tokens: list[str]
    source: Literal["rule", "llm", "both", "secure_diff", "triage"]


class Sink(_SinkRequired, total=False):
    # Difference axis (one-to-one with the change_kind vocabulary in
    # prompts.SECURE_DIFF_SYSTEM): guard=missing validation /
    # contract=contract mismatch / type_fix=truncation or type mismatch /
    # api_swap=swapped for a safe API.
    change_kind: Literal["guard", "contract", "type_fix", "api_swap"]
    contract_witness: ContractWitness
    obligations: list[Obligation]   # S1: security obligations pending for this sink
                                     # (instantiated by secure_diff)
    cwe_id: str                     # the model's best-fitting root cause at the
                                     # candidate stage, reused by arbiter aggregation
    # The candidate mechanism takes priority when picking the obligation
    # template; CWE is used only for attribution and as a compatible fallback.
    mechanism_kind: str
    # Distinguishable protected object/token, for de-duplicating multiple
    # mechanisms on the same line.
    subject: str
    # Forward-stepping forms hit by the unbounded-pointer-walk floor
    # (sink_locator.rule_locate_buffer_walks):
    # "incr" (*p++) | "addassign" (p+=k) | "helper" (p=Helper(...,p,...)).
    # Reporting and ablation statistics only; not consulted by the verdict.
    walk_kind: str


class _BaselineRequired(TypedDict):
    func_name: str          # one safe version per function, indexed by function
    secure_lines: str       # the secure diff for all of that function's risky lines
    rationale: str
    already_safe: bool       # whether the code already implements this CWE's safe
                              # pattern (no hardening needed -> a discriminator)


class BaselineItem(_BaselineRequired, total=False):
    contract_witnesses: list[ContractWitness]


class _ForensicRequired(TypedDict):
    sink_id: str
    hit_signals: list[str]
    evidence: str
    screen: str          # "exclude" (no risk signal, early exit) | "keep" (review)


class ForensicItem(_ForensicRequired, total=False):
    contract_verified: bool
    contract_evidence: list[str]
    # S1 evidence boundary: candidate facts proposed by the LLM (this node no
    # longer writes obligation status itself; obligation_checker rules on them).
    candidate_facts: list[dict]     # [{obligation_id, direction, anchors[], ...}]
    obligations: list[Obligation]   # proved-obligation snapshot after
                                    # obligation_checker advances them


class Verdict(TypedDict, total=False):
    sink_id: str
    is_vuln: bool
    # Three-valued verdict: "vuln" | "safe_proved" | "underdetermined".
    # is_vuln is equivalent only to resolution=="vuln"; both safe_proved and
    # underdetermined have is_vuln=False but differ in meaning -- the former has
    # safety positively proved, the latter merely failed to prove a vulnerability
    # (context missing / undetermined). Reports and evaluation use this to
    # separate "proved safe" from "unproved vulnerability" instead of repainting
    # the latter as the former.
    resolution: str
    cwe_id: str
    location: dict          # {file, line_start, line_end}
    confidence: float
    basis: list[str]
    # View A's secure side: the **expected** safe form supplied by the model, not
    # code measured to exist in the source. Human-readable output must
    # distinguish expected from actual; do not write it as "already hardened".
    fix_suggestion: str
    missing_guard: str              # the missing guard the arbiter names when
                                   # ruling vuln; empty when safe
    violated_contract: str          # the behavioural conflict named by a
                                   # contract-type positive; empty otherwise
    # Exclusive to resolution=="underdetermined": surface the risk without calling
    # it positive.
    # risk = human-readable "what kind of risk this is + why one file cannot
    # confirm it"; unresolved_reason = why the obligation is undetermined.
    # cwe_id is **still filled** when underdetermined (risk classification is
    # independent of whether a positive call is made), but is_vuln=False and
    # conf=0, so it stays out of bench's pred_cwes and does not affect the binary
    # metric view.
    risk: str
    unresolved_reason: str
    contract_witness: ContractWitness
    # The original threat-model self-assessment trio
    # (reachability/trigger_difficulty/impact) is not part of the pipeline:
    # both sides of a paired sample share code and self-assessment, so the
    # down-weighting is symmetric and yields no discriminative power. confidence
    # is tiered only by evidence-source strength (machine_witness > micro_check >
    # llm_anchor).
    # The following are optional: filled by de-duplication / lifecycle nodes
    root_cause: str                 # one-line description of the root cause
                                    # (used as the key finding when de-duplicating)
    propagation_path: list[dict]    # folded intermediate-point evidence chain:
                                    # [{sink_id, line, snippet}]
    origin: str                     # "arbiter" | "project", marking the source
    obligations: list[Obligation]   # S1: proved-conclusion snapshot, obligation
                                    # state after convergence


class Budget(TypedDict):
    tokens_used: int
    tokens_max: int
    llm_calls: int
    calls_max: int
    failed_calls: int  # calls still failing after retries (network/timeout);
                       # bench uses this to detect a blank run and skip persisting
    json_failures: int  # calls that succeeded but whose output was not parseable
                        # as JSON (the node degrades in place).
    # Counted separately from failed_calls: the network was fine and the chain
    # "succeeded" end to end, yet some node received None -> zero candidates /
    # knowledge fallback, so two runs give completely different reports. This is
    # the main source of run-to-run variation and must stay auditable.


class VulnState(TypedDict, total=False):
    mode: Literal["file", "project"]
    profile: str                        # "bench" (evaluation, full chain) |
                                        # "scan" (deployment, cheap prescreen
                                        # short-circuit enabled)
    target_path: str
    project_files: Optional[list[str]]
    project_index: Optional[dict]       # project mode: whole-project source index
                                        # + symbol table, for cross-file grep
                                        # and taint chains
    source_text: str                    # single-file original text, for nodes
                                        # to re-fetch context
    functions: list[Func]
    units: list[Unit]
    family_candidates: list[str]
    knowledge: dict                     # key=cwe_id -> CWE JSON (empty by design
                                        # under the clf backend)
    # Family knowledge cards (principle/trigger/examples/patch/safe_pattern),
    # filled by both backends, and the only source of the family block in
    # downstream prompts. Fed to the LLM only: the deterministic side
    # (checker/cfg/witness/arbiter) does not read it, so it cannot change a
    # verdict, only the "reference material".
    family_context: list[dict]
    sinks: list[Sink]
    baseline: list[BaselineItem]
    forensics: list[ForensicItem]
    verdict: list[Verdict]
    budget: Budget
    partial: bool
    errors: list[str]
    degraded: list[str]                 # degradation trace for this run (LLM
                                        # output unusable -> fallback / zero candidates)
    # When two runs disagree, this entry in the report is the explanation: which
    # node got no usable output, and which fallback it took. It **does not affect
    # the verdict**; it only makes "this report is a degraded product" visible.
    report: dict                        # the final report from the output node
    _budget_blown: bool                 # short-circuit flag for budget exhaustion
    _triage_reason: str                 # suspicions written by select_knowledge
                                        # while reading the code, passed to
                                        # secure_diff to dig further
    # Mechanism hypotheses after verbatim anchoring: clues for secure_diff only,
    # not evidence and not sinks.
    _triage_hypotheses: list[dict]
    _prescreen_clean: bool              # scan-profile prescreen verdict: no danger
                                        # indication at all -> short-circuit, skip the
                                        # LLM chain
    _secure_diff_outcome: str           # secure_diff comparison outcome:
                                        # no_diff/diff/llm_unavailable/disabled.
                                        # On empty output (sinks=0), output uses this
                                        # to separate "comparison completed, no
                                        # difference -> safe_proved" from
                                        # "comparison did not complete ->
                                        # underdetermined"; persisted in the report
                                        # summary for auditability.
    _prescreen_reasons: list[str]       # risky indications hit by the prescreen
                                        # (empty when clean)
