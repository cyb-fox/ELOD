"""Root-cause deduplication (deterministic clustering, zero LLM).

Problem: one root cause (e.g. an overflow of some tag_len) yields one verdict
per line along its propagation path, which inflates the vulnerability count and
buries independent problems.

Approach: union-find clustering over the verdicts with is_vuln, collapsing each
cluster into one primary finding:
- Union rules (either one suffices):
  1) the tainted tokens of two findings intersect (propagation along the data
     flow, same taint source);
  2) same function + same CWE family (the same point reported repeatedly over
     several lines).
- Primary finding: the highest confidence in the cluster (ties broken by the
  smallest line number); the remaining waypoints are folded into
  propagation_path as an evidence chain and no longer listed separately.
- safe verdicts (is_vuln=False) are kept as is and take no part in
  deduplication.

Pure Python, no LLM call.  Cross-function findings produced by the
project-level dataflow node enter the same clustering.
"""
from __future__ import annotations

from ..state import Sink, Verdict


def _cwe_root(cwe_id: str) -> str:
    """CWE family root: drop sub-class differences so that different sub-classes
    of one family can be merged.  Taking the raw value already works, but
    normalizing the known parent/child pairs reduces missed merges caused by
    119/120/125 differing as sub-classes."""
    fam = {
        "CWE-120": "CWE-119", "CWE-125": "CWE-119", "CWE-787": "CWE-119",
        "CWE-78": "CWE-77",
        "CWE-23": "CWE-22", "CWE-36": "CWE-22",
    }
    return fam.get(cwe_id, cwe_id)


def _sink_index(sinks: list[Sink]) -> dict:
    idx = {}
    for s in sinks or []:
        idx[s["sink_id"]] = s
    return idx


def _func_of(sink_id: str) -> str:
    # sink_id looks like "func:line"
    return sink_id.rsplit(":", 1)[0] if ":" in sink_id else sink_id


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def dedup_findings(verdicts: list[Verdict], sinks: list[Sink]) -> list[Verdict]:
    """Cluster and collapse the verdicts with is_vuln by root cause; safe ones
    are kept as is.  Returns a new list."""
    verdicts = verdicts or []
    vulns = [v for v in verdicts if v.get("is_vuln")]
    safes = [v for v in verdicts if not v.get("is_vuln")]
    if len(vulns) <= 1:
        return verdicts

    sink_idx = _sink_index(sinks)

    def tokens_of(v: Verdict) -> set[str]:
        s = sink_idx.get(v.get("sink_id", ""))
        toks = set(s["tainted_tokens"]) if s else set()
        return {t for t in toks if len(t) > 1}  # filter single-char noise

    n = len(vulns)
    uf = _UF(n)
    meta = [
        (tokens_of(v), _func_of(v.get("sink_id", "")), _cwe_root(v.get("cwe_id", "")))
        for v in vulns
    ]
    for i in range(n):
        ti, fi, ci = meta[i]
        for j in range(i + 1, n):
            tj, fj, cj = meta[j]
            # An identical CWE family root is a necessary condition for merging:
            # otherwise format-string / command-injection / integer-overflow
            # findings would collapse together because they share broad
            # variables such as s/pkt/buf, which buries independent problems.
            if ci == "" or ci != cj:
                continue
            share_tok = bool(ti & tj)   # same taint source along the data flow
            same_func = (fi == fj)      # same point reported over several lines
            if share_tok or same_func:
                uf.union(i, j)

    # Group into clusters
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    out: list[Verdict] = []
    for members in clusters.values():
        # Primary finding: highest conf, ties broken by the smallest line
        def sort_key(idx: int):
            v = vulns[idx]
            return (-float(v.get("confidence", 0.0) or 0.0),
                    int((v.get("location") or {}).get("line_start", 1 << 30)))
        members.sort(key=sort_key)
        main = dict(vulns[members[0]])
        path = list(main.get("propagation_path") or [])
        for idx in members[1:]:
            v = vulns[idx]
            loc = v.get("location") or {}
            path.append({
                "sink_id": v.get("sink_id", ""),
                "line": loc.get("line_start"),
                "cwe_id": v.get("cwe_id", ""),
                "snippet": (sink_idx.get(v.get("sink_id", "")) or {}).get("snippet", ""),
            })
        if path:
            main["propagation_path"] = path
            if not main.get("root_cause"):
                main["root_cause"] = (main.get("basis") or [""])[0]
        out.append(main)  # type: ignore[arg-type]

    # Primary findings sorted by line number, safe ones appended after
    out.sort(key=lambda v: int((v.get("location") or {}).get("line_start", 0)))
    return out + safes
