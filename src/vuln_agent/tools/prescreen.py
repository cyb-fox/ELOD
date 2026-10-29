"""prescreen: cheap file-level pre-filter (pure rules, zero token).

Most files in a real project are benign (declarations only, helper functions,
getters/setters), so running the full discipline chain on every file is the largest
avoidable cost. This module uses zero-cost tree-sitter plus regex to decide whether a
file contains "not even one suspicious operation"; if so it short-circuits to output
and skips every LLM call.

The bias is deliberately conservative (towards risky): a file is judged clean only
when it carries no danger indicator at all, which keeps the false-negative risk at a
minimum, since a real vulnerability almost always carries at least one of the
indicators below. Active in the scan profile only; disabled in the bench profile so
that the evaluation recall floor is protected.

Danger indicators (any hit means the file is not clean):
  - a rule sink (sensitive API call, families_from_ops non-empty)
  - array-index write / pointer dereference write / pointer arithmetic
  - memory allocation and release (malloc/free/new/delete...)
  - external input sources (recv/read/fread/fgets/scanf/getenv/argv...)
"""
from __future__ import annotations

import re

from ..state import Func
from . import sink_locator as sl

# array-index write: a[i] = ... (== comparisons excluded)
_ARR_WRITE = re.compile(r"\w+\s*\[[^\]]*\]\s*=(?!=)")
# pointer dereference write: *p = ...
_PTR_WRITE = re.compile(r"\*\s*\w+\s*=(?!=)")
# pointer arithmetic: p + n / p += / p++ (coarse match, prefers a false risky)
_PTR_ARITH = re.compile(r"\w+\s*(\+\+|--|\+=|-=)|\*\s*\(\s*\w+\s*[+\-]")
# memory allocation / release
_MEM_MGMT = re.compile(r"\b(malloc|calloc|realloc|free|alloca)\b|\b(new|delete)\b")
# external input sources (taint origins)
_EXT_INPUT = re.compile(
    r"\b(recv|recvfrom|read|fread|fgets|gets|scanf|sscanf|fscanf|getenv|"
    r"argv|getchar|fgetc|readlink|mmap)\b")

_RISK_PATTERNS = (
    ("array-write", _ARR_WRITE),
    ("ptr-write", _PTR_WRITE),
    ("ptr-arith", _PTR_ARITH),
    ("mem-mgmt", _MEM_MGMT),
    ("ext-input", _EXT_INPUT),
)


def prescreen_file(source: str, functions: list[Func]) -> tuple[bool, list[str]]:
    """Return (is_clean, reasons). is_clean=True means there is no danger indicator
    and the LLM chain can be short-circuited away.

    reasons: the list of danger indicators that were hit (empty when clean; a
    non-empty list explains why the full analysis is still required).
    """
    reasons: list[str] = []

    # 1) rule sinks / sensitive-operation families (reusing the zero-cost
    # tree-sitter path)
    if sl.families_from_ops(source, functions):
        reasons.append("sensitive-api")

    # 2) line-by-line regex scan for dangerous syntax (a coarse filter that skips
    # comment lines: lines starting with // or * are not counted)
    for raw in source.split("\n"):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("//", "*", "/*")):
            continue
        for label, pat in _RISK_PATTERNS:
            if label in reasons:
                continue  # one hit per category is enough
            if pat.search(stripped):
                reasons.append(label)
        if len(reasons) >= len(_RISK_PATTERNS) + 1:
            break  # enough indicators, no need to scan further

    return (len(reasons) == 0, reasons)
