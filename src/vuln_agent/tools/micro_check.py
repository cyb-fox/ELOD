"""micro_check (step 5, semantic micro-check: a deterministic proposer, zero LLM,
no build).

Applies syntactic-level abstract interpretation to the neighbourhood window of a risky
line to mechanically discover candidate facts of the "size arithmetic overflow" kind:
a tainted operand appears under multiplication, addition or shift inside an
allocation or copy length argument, and the window contains no overflow guard for that
operand (SIZE_MAX comparison, division bound check, __builtin_*_overflow,
reallocarray, and so on).

Role within the evidence boundary: this module is a *proposer* with the same standing
as an LLM node. It emits CandidateFacts (anchors = verbatim source lines from the
window) which are handed to obligation_checker as usual and advance an obligation only
after re-check; it never writes an obligation status directly. The sole difference is
that this proposal is deterministic, recomputable and costs zero tokens.

The reported scope is deliberately narrow, to avoid symmetric false positives:
  - only "size arithmetic without a guard" is reported (the classic fixable form of
    CWE-190, where a patch usually *adds* a check, so the two versions are asymmetric
    and the discriminative signal is valid);
  - truncating casts such as (unsigned char)x are not reported: such fixes are often
    just a widening conversion in the middle, and the cast itself has the same form
    in the vulnerable and in the patched version, so reporting it necessarily reports
    both sides.
"""
from __future__ import annotations

import re

from .. import config

# sink_type -> the critical obligation for a size arithmetic overflow (see
# obligations._TEMPLATES; CWE-190 maps to mem in _sink_type_of, and its
# "length out of range after wraparound" lands on readable_extent).
_TARGET = {
    "alloc": "size_overflow_checked",
    "arith": "arithmetic_bound",
    "mem": "readable_extent",
}

# allocation / copy family calls (the length argument is where the overflow is
# consumed).
_SIZE_CALL = re.compile(
    r"\b(malloc|calloc|realloc|reallocarray|alloca|xmalloc|xcalloc|"
    r"kmalloc|kzalloc|krealloc|kcalloc|vmalloc|av_malloc\w*|g_malloc\w*|"
    r"\w*_alloc\w*|Acquire\w*Memory|Resize\w*Memory|"
    r"memcpy|memmove|memset|bcopy|strncpy|strncat|snprintf|CopyMagickMemory)\s*\(",
)

# Explicitly overflow-safe APIs: their presence means the window already has a
# mechanically checkable guard.
_SAFE_OVERFLOW_API = re.compile(
    r"__builtin_\w*_overflow|check_\w*_overflow|reallocarray|ckd_(add|sub|mul)",
)

_GUARD_HEAD = re.compile(r"\b(if|while|assert|BUG_ON|WARN_ON|CHECK|OP_REQUIRES)\b")
_NUM_LINE = re.compile(r"^\s*(\d+)\s*[|:]\s?(.*)$")


def _lines(consumer_source: str) -> list[str]:
    """Expand the window into a list of source lines, stripping the "N|" prefix added
    by number_lines; lines without a prefix are kept as they are."""
    out = []
    for ln in (consumer_source or "").split("\n"):
        m = _NUM_LINE.match(ln)
        out.append(m.group(2) if m else ln)
    return out


def _has_overflow_guard(lines: list[str], tok: str) -> bool:
    """Whether the window already contains an overflow guard for `tok`: either an
    overflow-safe API, or a check-shaped line containing both `tok` and a *_MAX
    macro / the word overflow / a division bound comparison."""
    text = "\n".join(lines)
    if _SAFE_OVERFLOW_API.search(text):
        return True
    tok_re = re.compile(rf"\b{re.escape(tok)}\b")
    for ln in lines:
        if not _GUARD_HEAD.search(ln) or not tok_re.search(ln):
            continue
        if re.search(r"_MAX\b", ln) or re.search(r"overflow", ln, re.I) or "/" in ln:
            return True
    return False


def propose_facts(sink: dict, consumer_source: str) -> list[dict]:
    """Run the micro-check on the consumption window of a single sink and return the
    list of candidate facts (possibly empty).

    The anchor is the matched verbatim source line, which always lies inside the
    window and is therefore re-checkable by verify_local.
    """
    if not config.MICRO_CHECK_ENABLED:
        return []
    target = _TARGET.get(str(sink.get("sink_type", "")))
    if not target:
        return []
    tainted = [t for t in (sink.get("tainted_tokens") or [])
               if isinstance(t, str) and len(t) > 1]
    if not tainted:
        return []
    lines = _lines(consumer_source)
    facts: list[dict] = []
    for raw in lines:
        m = _SIZE_CALL.search(raw)
        if not m:
            continue
        args = raw[m.end():]
        for tok in tainted:
            t = re.escape(tok)
            # tainted arithmetic in a length argument: tok*x / x*tok / tok+x /
            # tok<<x.
            if not re.search(rf"(\b{t}\s*[*+]|[*+]\s*{t}\b|\b{t}\s*<<)", args):
                continue
            if _has_overflow_guard(lines, tok):
                break
            anchor = raw.strip()
            if len(anchor) >= 6:
                facts.append({
                    "obligation_id": target, "direction": "VIOLATED",
                    "anchors": [anchor], "cross_file": False,
                    "provenance": "micro_check",
                })
            break
        if len(facts) >= 2:  # at most two per sink, to avoid noise accumulation
            break
    return facts
