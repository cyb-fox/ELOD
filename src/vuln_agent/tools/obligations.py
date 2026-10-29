"""obligations: safety-proof obligation templates (the carrier for the S1
evidence boundary).

Restates "does this code have a vulnerability" as "is a set of safety-proof
obligations satisfied". Obligations are instantiated per sink_type; each one has a
verification status machine UNDERDETERMINED/SATISFIED/VIOLATED/UNRESOLVED and
source-level evidence references.

This module provides only the **templates and bucket mapping**; it adjudicates no
status -- the sole discharger is tools/obligation_checker:
- secure_diff hits a real diff -> the critical obligation targeted by that diff
  receives the candidate direction (it must still pass the checker audit).
- A candidate fact from risk_forensics advances an obligation only after checker
  review; the arbiter converges strictly on the checker snapshot.
"""
from __future__ import annotations

from ..state import Obligation

# sink_type -> the critical proof-obligation templates for that family
# (id, predicate, critical).
# The family counts as safe only when every critical=True obligation is SATISFIED;
# any VIOLATED obligation with evidence yields a positive.
_TEMPLATES: dict[str, list[tuple[str, str, bool]]] = {
    "mem": [
        ("attacker_control", "is ptr/length attacker-controlled", True),
        ("non_negative_length", "is length non-negative", True),
        ("readable_extent", "readable extent of the object ptr points to >= length", True),
        ("dominating_guard", "is there an effective bounds check dominating the sink", False),
    ],
    "alloc": [
        ("attacker_control", "is the allocation size attacker-controlled", True),
        ("size_overflow_checked", "is the size arithmetic checked for wrap/overflow", True),
    ],
    "exec": [
        ("attacker_control", "are the command/arguments attacker-controlled", True),
        ("input_sanitized", "is external input escaped or allowlist-validated", True),
    ],
    "file_op": [
        ("attacker_control", "is the path attacker-controlled", True),
        ("path_validated", "is the path normalized/traversal-validated", True),
    ],
    "format": [
        ("attacker_control", "is the format string attacker-controlled", True),
        ("format_string_fixed", "is the format string a fixed literal", True),
    ],
    "sql": [
        ("attacker_control", "is the query input attacker-controlled", True),
        ("query_parameterized", "is the query parameterized/prepared", True),
    ],
    # CWE-476 null pointer dereference: the pointer may be NULL and no dominating
    # non-null check precedes the dereference.
    "deref": [
        ("attacker_control", "can the pointer source be NULL (allocation failure / "
                             "unchecked upstream return)", True),
        ("null_guard", "is there an effective non-null check dominating the dereference", True),
    ],
    # CWE-416/415 use-after-free / double free (lifetime): is the object still alive
    # at the point of use.
    "use": [
        ("object_live_at_use", "is the object still alive at the point of use (not freed, "
                               "or reallocated/nulled after the free)", True),
        ("dominating_guard", "is there a guard blocking use after free", False),
    ],
    # CWE-369/190/191 numeric: are division by zero and integer wrap checked by a
    # bound or a zero test.
    "arith": [
        ("attacker_control", "are the operands entering the arithmetic attacker-controlled", True),
        ("arithmetic_bound", "is a non-zero divisor / no overflow wrap checked", True),
    ],
    # Non-memory mechanisms (Phase 1): dedicated obligations; reusing
    # attacker_control / readable_extent is forbidden.
    "state_transition": [
        ("state_origin_verified", "has the state source been validated or authenticated", True),
        ("state_transition_valid", "does the transition follow the agreed order and legal values", True),
        ("state_checked_before_use", "was the current value checked before it was consumed", True),
    ],
    "boundary_validation": [
        ("external_input_identified", "is external / trust-boundary-crossing input identified", True),
        ("boundary_constraint_enforced", "is the boundary constraint enforced before consumption", True),
    ],
    "resource_lifecycle": [
        ("resource_acquired_validly", "was the resource acquired validly with clear ownership", True),
        ("release_or_transfer_complete", "is release or ownership transfer completed on every exit path", True),
        ("failure_path_cleanup", "is cleanup completed on failure/exception paths", True),
    ],
    "concurrency_state": [
        ("shared_state_synchronized", "is access to shared state synchronized", True),
        ("operation_order_enforced", "is the operation order of callbacks/queues/locks enforced", True),
    ],
}

# Generic fallback template (unknown sink_type): controllability plus one guard
# obligation only.
_FALLBACK = [
    ("attacker_control", "is the input of the dangerous operation attacker-controlled", True),
    ("dominating_guard", "is there an effective guard dominating that operation", True),
]

# change_kind / sink_type -> the id of the critical obligation that the diff mainly
# targets (used by secure_diff).
_GUARD_TARGET: dict[str, str] = {
    "mem": "readable_extent",
    "alloc": "size_overflow_checked",
    "exec": "input_sanitized",
    "file_op": "path_validated",
    "format": "format_string_fixed",
    "sql": "query_parameterized",
    "deref": "null_guard",
    "use": "object_live_at_use",
    "arith": "arithmetic_bound",
    "state_transition": "state_checked_before_use",
    "boundary_validation": "boundary_constraint_enforced",
    "resource_lifecycle": "release_or_transfer_complete",
    "concurrency_state": "shared_state_synchronized",
}


def template_for(sink_type: str) -> list[tuple[str, str, bool]]:
    return _TEMPLATES.get(sink_type, _FALLBACK)


def instantiate(sink_type: str, *, targeted: str = "", initial_status: str = "UNDERDETERMINED",
                evidence: list[str] | None = None, provenance: str = "",
                candidate_direction: str = "", candidate_evidence: list[str] | None = None,
                candidate_provenance: str = "") -> list[Obligation]:
    """Instantiate obligations for a sink_type; the candidate fields are strictly
    separated from the verification status."""
    out: list[Obligation] = []
    for oid, pred, crit in template_for(sink_type):
        ob = Obligation(
            id=oid, predicate=pred, critical=crit,
            status=initial_status if oid == targeted else "UNDERDETERMINED",
        )
        if oid == targeted:
            if evidence:
                ob["evidence"] = list(evidence)
            if provenance:
                ob["provenance"] = provenance
            if candidate_direction:
                ob["candidate_direction"] = candidate_direction
            if candidate_evidence:
                ob["candidate_evidence"] = list(candidate_evidence)
            if candidate_provenance:
                ob["candidate_provenance"] = candidate_provenance
        out.append(ob)
    return out


def guard_target(sink_type: str, change_kind: str) -> str:
    """Which critical obligation a secure_diff diff mainly restores.

    contract diff -> the controllability obligation (a contract is most often a
    producer/consumer semantic mismatch, so the controllability criterion changes);
    guard / type_fix / api_swap -> the boundary/guard obligation of that sink_type:
    all three restore the same obligation and differ only in repair form (add a
    branch / widen a signed type / swap to a bounded API), so they share one target.
    An obligation states "what must be guaranteed", not "how it is written", so no
    separate obligation name is introduced per diff form.
    """
    if change_kind == "contract":
        return "attacker_control"
    return _GUARD_TARGET.get(sink_type, "dominating_guard")


def critical_ids(sink_type: str) -> list[str]:
    return [oid for oid, _p, crit in template_for(sink_type) if crit]


# sink_type -> the most typical CWE number for that family (a deterministic
# fallback).
# Because the arbiter aggregates deterministically, cwe_id does not come from the
# LLM: the cwe_id carried by the secure_diff candidate is preferred (the model
# supplies the most fitting root cause at **proposal** time, still filtered by the
# candidate gate), and this table is the fallback.
# Used only for family classification in reports and evaluation; it adjudicates
# nothing.
_SINK_CWE: dict[str, str] = {
    "mem": "CWE-119",
    "alloc": "CWE-190",
    "exec": "CWE-78",
    "file_op": "CWE-22",
    "format": "CWE-134",
    "sql": "CWE-89",
    "deref": "CWE-476",
    "use": "CWE-416",
    "arith": "CWE-369",
    "state_transition": "CWE-354",
    "boundary_validation": "CWE-20",
    "resource_lifecycle": "CWE-404",
    "concurrency_state": "CWE-362",
}


def cwe_hint(sink_type: str) -> str:
    """The deterministic CWE classification of a sink_type (the fallback when no
    candidate carries a cwe_id)."""
    return _SINK_CWE.get(sink_type, "")


# ---- Registry of APIs with nullable returns (used by gate 1, the precondition
#      gate) ----
# A positive for CWE-476 rests on the precondition "the pointer may be NULL". If
# that precondition points at **the return value of an external API**, it is not
# observable within a single file -- an LLM could assert "X may return NULL"
# verbally and advance the obligation, which is a direct source of pair-level
# symmetric false positives (the same code and the same assertion on both the
# patched and the vulnerable side). This precondition is therefore established
# only three ways:
#   a) the API is in this registry (a libc/POSIX return widely understood to be
#      nullable, machine-checkable);
#   b) the candidate fact carries a cross-file anchor (the nullable return of that
#      API was really located in the project, in code or in a doc comment);
#   c) within the consumption window, the pointer is seen to come from an assignment
#      of an API in this registry.
# Project-private APIs are deliberately excluded; they must go through b), and
# otherwise the obligation stays UNDERDETERMINED (the conservative reading of I3).
NULLABLE_APIS: frozenset[str] = frozenset({
    # allocation family: returns NULL on failure
    "malloc", "calloc", "realloc", "reallocarray", "aligned_alloc", "valloc",
    "strdup", "strndup", "asprintf",
    "kmalloc", "kzalloc", "kcalloc", "krealloc", "vmalloc", "kmemdup",
    "g_malloc0", "av_malloc", "av_mallocz", "xmlMalloc", "PyMem_Malloc",
    # open/lookup family: returns NULL on failure or miss
    "fopen", "fdopen", "freopen", "tmpfile", "opendir", "fgets",
    "getenv", "secure_getenv", "dlopen", "dlsym",
    "strchr", "strrchr", "strstr", "strcasestr", "strpbrk", "strtok",
    "memchr", "memrchr", "bsearch", "gets",
    "localtime", "gmtime", "getpwnam", "getgrnam", "gethostbyname",
    "mmap",  # returns MAP_FAILED on failure, but is commonly NULL-checked, so listed
})


def mentions_nullable_api(text: str) -> bool:
    """Whether the text (an anchor/snippet) names a nullable-return API in the
    registry. Matched as whole identifiers."""
    import re
    for tok in re.findall(r"[A-Za-z_]\w*", text or ""):
        if tok in NULLABLE_APIS:
            return True
    return False


# ---- Registry of recognisable attacker-controlled source APIs (used by gate I6,
#      the external-source decidability gate) ----
# Same design intent as NULLABLE_APIS, different direction: the positive
# precondition "this value is attacker-controlled", when it points at **the return
# value of an external function**, is likewise not observable within a single file.
# Copying the assignment line back as an anchor (e.g.
# `packet_size = ExternalPacketSize(image);`) advances the obligation, yet that line
# only says "this value is invisible to me" -- a self-report of missing information,
# not evidence of controllability.
# The precondition is accepted only three ways: a) the callee is in this registry
# (a recognised external input / byte-order conversion, machine-checkable);
# b) the candidate fact carries a cross-file anchor (the producer was really located
# in the project); c) it goes through the witness channel (taint-reachable on the
# CFG and the constraints are not refutable).
# Project-private APIs are deliberately excluded; they must go through b) or c),
# otherwise the obligation stays UNDERDETERMINED (the conservative reading of I3).
SOURCE_APIS: frozenset[str] = frozenset({
    # reading external bytes directly
    "read", "recv", "recvfrom", "recvmsg", "pread", "readv",
    "fread", "fgets", "fgetc", "getc", "getchar", "gets", "getline", "getdelim",
    "scanf", "fscanf", "sscanf", "vscanf", "vfscanf",
    "copy_from_user", "get_user", "simple_write_to_buffer",
    # byte-order/width conversion: its presence means the value came off the wire
    "ntohs", "ntohl", "ntohll", "be16toh", "be32toh", "be64toh",
    "le16toh", "le32toh", "le64toh", "bswap_16", "bswap_32", "bswap_64",
    # out-of-process configuration/parameters
    "getenv", "secure_getenv", "getopt", "getopt_long",
    # common container/stream reading wrappers
    "ReadBlob", "ReadBlobByte", "ReadBlobLSBShort", "ReadBlobMSBShort",
    "ReadBlobLSBLong", "ReadBlobMSBLong", "ReadBlobString",
    "avio_r8", "avio_rb16", "avio_rb32", "avio_rl16", "avio_rl32",
    "get_bits", "get_bits_long", "bytestream_get_le32", "bytestream_get_be32",
})


def mentions_source_api(text: str) -> bool:
    """Whether the text names an attacker-controlled source API in the registry.
    Matched as whole identifiers."""
    import re
    for tok in re.findall(r"[A-Za-z_]\w*", text or ""):
        if tok in SOURCE_APIS:
            return True
    return False


# CWE root -> family bucket (trajectories/evaluation aggregate by
# (family|sink_type); a coarse granularity suffices).
_CWE_FAMILY: dict[str, str] = {
    "119": "buffer", "120": "buffer", "125": "buffer", "787": "buffer", "121": "buffer",
    "190": "intoverflow", "191": "intoverflow",
    "77": "injection", "78": "injection", "134": "injection", "89": "injection",
    "22": "pathtraversal", "23": "pathtraversal", "36": "pathtraversal",
    "416": "lifetime", "415": "lifetime", "476": "nullptr",
    "369": "numeric", "193": "numeric",
}


def bucket_key(cwe_id: str, sink_type: str) -> str:
    """Trajectory bucket key: (CWE family | sink_type). Used for the side-channel
    trajectory audit and for offline evaluation aggregation; it produces no
    executable artifact and adjudicates nothing."""
    num = ""
    for part in str(cwe_id).upper().replace("CWE-", "").split("-"):
        part = part.strip()
        if part.isdigit():
            num = part
            break
    fam = _CWE_FAMILY.get(num, "other")
    return f"{fam}|{sink_type or 'other'}"
