"""Three-metric independent evaluation (score-only, zero token): detection / CWE
attribution / tolerance curve, plus a context-sufficiency stratification.

Key difference from `bench.py`: the label condition is **not** ANDed into the
confusion matrix. The cwe-family view in bench is `pred_vuln AND family_hit`, i.e.
detection plus one more necessary condition, so its TP can only shrink and its
value is necessarily below detection -- that is not a "relaxed view", the two are
not comparable. This module separates the three entangled questions:

  M1 detection (true/false)  : asks only whether a vulnerability exists, symmetric
                               on both sides, no CWE involved. The only view that
                               can be aligned with external baselines; pair metrics
                               (P-C/P-V/P-B/P-R) are reported under this view alone.
  M2 CWE attribution accuracy: denominator = vulnerable files M1 got right (a
                               conditional metric, not folded into the confusion
                               matrix); reports top-1 and any-of-k exact hit rates.
  M3 tolerance curve         : grades the relation between predicted CWE and GT and
                               reports cumulative hit rate as tolerance widens:
                               exact ⊂ hierarchy ⊂ chain ⊂ family. Levels are
                               derived deterministically from KB structure
                               (family.child_of/parent_of closure +
                               consequence_chains), with no hand-written mapping
                               table. This curve is the answer to "why is family-level
                               scoring lower": family buckets cut apart
                               same-mechanism chains such as CWE-190→CWE-787.

Context sufficiency stratification: a substantial share of PrimeVul samples
cannot be judged from the file alone (missing caller constraints, ownership
contracts, framework semantics). `data/primevul_det/manifest.jsonl` is the current
roster of label conditions: a vulnerable function must support the target CWE, and
the patched function must additionally show local evidence of a guard against the
same mechanism. The three metrics are reported inside and outside the roster. This
stratification estimates file-level performance under the given roster conditions;
it is not an upper bound on model capability, and the difference between the two
layers must not be read as a causal effect of context.

Label-space coverage is reported separately: samples whose GT CWE falls outside
the 96 KB entries (Category/deprecated entries such as CWE-189/399) cannot be hit
by any defect-level predictor; M2/M3 drop them by default and state the
denominator.

────────────────────────── Usage ──────────────────────────
python3 -m vuln_agent.metrics --pred runs/pred_primevul_s10.jsonl
python3 -m vuln_agent.metrics --pred runs/pred_primevul.jsonl --json runs/metrics.json
python3 -m vuln_agent.metrics --pred runs/pred_primevul.jsonl --per-file runs/detail.csv
python3 -m vuln_agent.metrics --pred runs/pred.jsonl --keep-unmapped   # keep GT off-KB
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from . import config

# Tolerance levels, strict to loose; MISS is the fallback. Cumulative view =
# everything at this level or stricter counts as a hit.
LEVELS = ("exact", "hierarchy", "chain", "family")
MISS = "unrelated"
_LEVEL_RANK = {lv: i for i, lv in enumerate(LEVELS)}
_LEVEL_RANK[MISS] = len(LEVELS)

CHAIN_MAX_HOPS = 2  # max hops on consequence_chains (190→787 is 1, 20→190→835 is 2)
METRICS_SCHEMA_VERSION = 2

# ================ CWE taxonomy (all read from KB, no hand-written map) ================


class Taxonomy:
    """The three relations between CWEs, read from data/cwe_agent_kb/, used for
    tolerance grading.

    hierarchy: ancestor-descendant reachability (transitive closure) over the
      directed graph formed by family.child_of / parent_of. A prediction more
      specific than GT (a descendant) counts as a hit -- when GT is at Class/Pillar
      level (e.g. CWE-20/703), predicting CWE-119/CWE-190 is a **more precise
      answer**, and scoring it a miss would be the evaluation's error, not the
      model's.
    chain   : adjacent CWEs co-occurring in consequence_chains (undirected,
      ≤CHAIN_MAX_HOPS hops). This is the deterministic source for "different CWE
      number, same triggering mechanism": CWE-190→CWE-787 (integer overflow
      leading to under-allocation and then an out-of-bounds write) is explicitly
      registered in the KB, yet the two sit in the numeric-errors and
      memory-safety buckets respectively.
    family  : the top-level bucket in families.json, the loosest level.
    """

    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir
        self.known: set[str] = set()          # CWEs with a KB entry
        self.abstraction: dict[str, str] = {}  # cwe -> Pillar/Class/Base/Variant
        self.down: dict[str, set[str]] = defaultdict(set)  # parent -> child
        self.up: dict[str, set[str]] = defaultdict(set)    # child -> parent
        self.chain_adj: dict[str, set[str]] = defaultdict(set)  # undirected chain adj
        self.family: dict[str, str] = {}       # cwe -> top-level family name
        self._load()

    def _load(self) -> None:
        for p in sorted(self.kb_dir.glob("CWE-*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            cwe = norm_cwe(d.get("cwe_id"))
            if not cwe:
                continue
            self.known.add(cwe)
            self.abstraction[cwe] = str(d.get("abstraction") or "")
            fam = d.get("family") or {}
            for parent in fam.get("child_of") or []:
                if (pn := norm_cwe(parent)):
                    self.down[pn].add(cwe)
                    self.up[cwe].add(pn)
            for kid in fam.get("parent_of") or []:
                if (kn := norm_cwe(kid)):
                    self.down[cwe].add(kn)
                    self.up[kn].add(cwe)
            for ch in d.get("consequence_chains") or []:
                seq = [norm_cwe(x) for x in (ch.get("chain") or [])]
                seq = [x for x in seq if x]
                for a, b in zip(seq, seq[1:]):
                    self.chain_adj[a].add(b)
                    self.chain_adj[b].add(a)
        try:
            fams = json.loads(
                (self.kb_dir / "families.json").read_text(encoding="utf-8")
            ).get("families", {}) or {}
        except Exception:
            fams = {}
        for fam, info in fams.items():
            for cwe in (info or {}).get("members", []) if isinstance(info, dict) else []:
                if (cn := norm_cwe(cwe)):
                    self.family[cn] = fam

    @lru_cache(maxsize=4096)
    def _reach(self, start: str, direction: str) -> frozenset[str]:
        """Transitive closure along down/up (the KB's parent_of only reaches direct
        children, so the closure is needed to cover skip-generation relations such
        as CWE-20 → CWE-119 → CWE-125)."""
        graph = self.down if direction == "down" else self.up
        seen: set[str] = set()
        frontier = {start}
        while frontier:
            nxt: set[str] = set()
            for node in frontier:
                nxt |= graph.get(node, set())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return frozenset(seen)

    @lru_cache(maxsize=4096)
    def _chain_reach(self, start: str) -> frozenset[str]:
        seen: set[str] = set()
        frontier = {start}
        for _ in range(CHAIN_MAX_HOPS):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= self.chain_adj.get(node, set())
            nxt -= seen | {start}
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return frozenset(seen)

    def relation(self, gt: str, pred: str) -> str:
        """Relation level between GT and one predicted CWE (LEVELS or MISS)."""
        gt, pred = norm_cwe(gt), norm_cwe(pred)
        if not gt or not pred:
            return MISS
        if gt == pred:
            return "exact"
        if pred in self._reach(gt, "down") or pred in self._reach(gt, "up"):
            return "hierarchy"
        if pred in self._chain_reach(gt):
            return "chain"
        if gt in self.family and self.family.get(pred) == self.family[gt]:
            return "family"
        return MISS

    def same_family(self, gt: str, pred: str) -> bool:
        """Whether both fall in the same top-level CWE family in the KB; a
        cross-family consequence chain does not count."""
        gt, pred = norm_cwe(gt), norm_cwe(pred)
        return bool(gt and pred and gt in self.family
                    and self.family.get(pred) == self.family[gt])

    def best_relation(self, gt: str, preds: list[str]) -> str:
        """The tightest relation to GT among a set of predictions (any-of-k view)."""
        return min((self.relation(gt, p) for p in preds),
                   key=lambda lv: _LEVEL_RANK[lv], default=MISS)

    def in_label_space(self, gt: str) -> bool:
        """Whether the GT CWE is inside the KB label space. If not, no defect-level
        prediction can hit it (mostly Category or deprecated entries, e.g.
        CWE-189/399/17), so M2/M3 should drop it and state the denominator."""
        return norm_cwe(gt) in self.known


def norm_cwe(raw: Any) -> str:
    """Normalize to 'CWE-123'; returns '' when unrecognized."""
    if not raw:
        return ""
    s = str(raw).strip().upper()
    if s.startswith("CWE-"):
        s = s[4:]
    elif s.startswith("CWE"):
        s = s[3:].lstrip("-")
    return f"CWE-{s}" if s.isdigit() else ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =============== Data loading: predictions + context-sufficiency labels ===============


def _load_pred(path: Path) -> list[dict]:
    """Read predictions.jsonl, keeping the last record per logical pair side, and
    tolerate journals without pair_id.

    The same materialized source may be reused as a side of different PrimeVul
    pairs, so de-duplicating by file path alone would silently drop complete
    pairs. `error` records are still removed uniformly by the caller, so their
    count stays auditable.
    """
    by_side: dict[tuple[str, str, Optional[bool]], dict] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        pair_id = str(rec.get("pair_id") or "")
        file = str(rec.get("file") or "")
        key = (pair_id, file, bool(rec.get("is_vulnerable"))) if pair_id else ("", file, None)
        by_side[key] = rec
    return list(by_side.values())


def _load_determinable(path: Optional[Path]) -> Optional[set[str]]:
    """Read the manifest of the "this file's context suffices" subset and return
    its pair_id set.

    Building or expert-reviewing that roster is not this scorer's job; it only
    stratifies by the manifest's pair_id. Samples not listed are recorded as
    undeterminable, meaning insufficient in-function context under that roster.
    Returns None when the file is absent (stratification degrades to a single
    stratum, and the other two metrics are still reported)."""
    if path is None or not path.exists():
        return None
    pids: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            pids.add(json.loads(ln)["pair_id"])
        except Exception:
            continue
    return pids or None


def _build_pairs(recs: list[dict], determinable: Optional[set[str]]) -> list[dict]:
    """Aggregate into pairs by pair_id (1 vuln + 1 patched), attaching the
    context-sufficiency label. Incomplete pairs (the other half not run / errored)
    are counted separately and excluded from pair metrics."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[r.get("pair_id") or r.get("file", "")].append(r)
    pairs = []
    for pid, items in groups.items():
        v = [x for x in items if x.get("is_vulnerable")]
        p = [x for x in items if not x.get("is_vulnerable")]
        pairs.append({
            "pair_id": pid,
            "vuln": v[0] if len(v) == 1 else None,
            "patched": p[0] if len(p) == 1 else None,
            "complete": len(v) == 1 and len(p) == 1,
            # a None determinable means everything is labelled "all", so downstream
            # stratification emits only one stratum
            "ctx": ("all" if determinable is None
                    else ("determinable" if pid in determinable else "undeterminable")),
        })
    return pairs


# ============================= M1: detection (true/false) =============================


def _confusion(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "N": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
    }


def m1_detection(pairs: list[dict]) -> dict:
    """M1: asks only "is there a vulnerability", symmetric on both sides, with no
    CWE involved.

    This is the only view comparable to external baselines (LineVul / CodeBERT /
    direct GPT-4) and the only view in which pair metrics are meaningful: pair
    accuracy is a discrimination metric, and mixing in label agreement would make
    it uninterpretable."""
    tp = fp = tn = fn = 0
    for pr in pairs:
        for rec, gt in ((pr["vuln"], True), (pr["patched"], False)):
            if rec is None:
                continue
            pos = bool(rec.get("pred_vuln"))
            if gt:
                tp += pos
                fn += not pos
            else:
                fp += pos
                tn += not pos
    out = {"file_level": _confusion(tp, fp, tn, fn)}
    out["pair_level"] = _pair_level(pairs)
    out["evidence_backed_underdetermined_pair"] = m1u_evidence_backed_underdetermined(pairs)
    out["abstain"] = _abstain_profile(pairs)
    return out


def _abstain_profile(pairs: list[dict]) -> dict:
    """Side profile of the three-valued verdict (does not change the M1 view; it
    only explains where M1's negatives come from).

    In the binary view every non-positive counts as negative, but `underdetermined`
    (a key obligation not proved, often because the caller is invisible from a
    single file) and `safe_proved` (safety positively proved) are two different
    conclusions. Counting them separately:
    - how many FN on the vulnerable side are actually `underdetermined` -> the
      share that is "not judged wrong, but not dared to judge";
    - how many TN on the patched side are `safe_proved` -> the proportion
      genuinely proved safe (the higher it is, the less TN was bought by silence).
    Records with no file_resolution field are counted under `legacy` and do not
    contaminate the first two."""
    prof = {
        "fn_underdetermined": 0, "fn_safe_proved": 0,
        "tn_underdetermined": 0, "tn_safe_proved": 0,
        "legacy": 0,
    }
    for pr in pairs:
        for rec, gt in ((pr["vuln"], True), (pr["patched"], False)):
            if rec is None or bool(rec.get("pred_vuln")):
                continue
            res = str(rec.get("file_resolution") or "")
            if res not in ("underdetermined", "safe_proved"):
                prof["legacy"] += 1
                continue
            side = "fn" if gt else "tn"
            prof[f"{side}_{res}"] += 1
    neg = sum(prof[k] for k in ("fn_underdetermined", "fn_safe_proved",
                                "tn_underdetermined", "tn_safe_proved"))
    unk = prof["fn_underdetermined"] + prof["tn_underdetermined"]
    prof["underdetermined_share"] = round(unk / neg, 4) if neg else 0.0
    return prof


def _pair_level(pairs: list[dict]) -> dict:
    """PrimeVul pair view: P-C both sides correct, P-V vulnerable side only,
    P-B patched side only, P-R both wrong."""
    pc = pv = pb = pr_ = 0
    incomplete = 0
    for p in pairs:
        if not p["complete"]:
            incomplete += 1
            continue
        v_ok = bool(p["vuln"].get("pred_vuln"))
        p_ok = not bool(p["patched"].get("pred_vuln"))
        if v_ok and p_ok:
            pc += 1
        elif v_ok:
            pv += 1
        elif p_ok:
            pb += 1
        else:
            pr_ += 1
    total = pc + pv + pb + pr_
    return {
        "pairs_scored": total, "pairs_incomplete": incomplete,
        "P-C": pc, "P-V": pv, "P-B": pb, "P-R": pr_,
        "pair_accuracy": round(pc / total, 4) if total else 0.0,
    }


def _evidence_backed_underdetermined(rec: Optional[dict]) -> bool:
    """Accept only an unknown verdict that was reviewed and left a risk CWE; an
    empty output or an error must never count as a hit."""
    if not rec or rec.get("error") is not None or rec.get("pred_vuln"):
        return False
    if rec.get("file_resolution") != "underdetermined":
        return False
    if int(rec.get("sinks_examined") or 0) <= 0 or int(rec.get("n_underdetermined") or 0) <= 0:
        return False
    return any(norm_cwe(c) for c in (rec.get("underdetermined_cwes") or []))


def m1u_evidence_backed_underdetermined(pairs: list[dict]) -> dict:
    """M1-U / P-C1: allows a source-anchored undetermined risk on the vulnerable
    side, without altering strict M1.

    This side channel answers "did the model spot the risk and abstain honestly",
    not a substitute for the file-level binary result. The patched side must
    still carry no confirmed vulnerability; a confirmed finding there is not
    exempted by the vulnerable side being underdetermined.
    """
    strict_pc = tolerated_underdetermined = blocked_by_patch = incomplete = 0
    vulnerable_underdetermined = unsupported_underdetermined = 0
    for pair in pairs:
        if not pair["complete"]:
            incomplete += 1
            continue
        vuln, patched = pair["vuln"], pair["patched"]
        strict_hit = bool(vuln.get("pred_vuln"))
        underdetermined_hit = _evidence_backed_underdetermined(vuln)
        if vuln.get("file_resolution") == "underdetermined" and not strict_hit:
            vulnerable_underdetermined += 1
            unsupported_underdetermined += not underdetermined_hit
        patched_negative = not bool(patched.get("pred_vuln"))
        if strict_hit and patched_negative:
            strict_pc += 1
        elif underdetermined_hit and patched_negative:
            tolerated_underdetermined += 1
        elif underdetermined_hit:
            blocked_by_patch += 1
    total = len(pairs) - incomplete
    pc1 = strict_pc + tolerated_underdetermined
    return {
        "pairs_scored": total, "pairs_incomplete": incomplete,
        "P-C1": pc1, "P-C_strict": strict_pc, "P-U": tolerated_underdetermined,
        "eligible_underdetermined_blocked_by_patch": blocked_by_patch,
        "vulnerable_underdetermined": vulnerable_underdetermined,
        "vulnerable_underdetermined_without_evidence": unsupported_underdetermined,
        "pair_accuracy": round(pc1 / total, 4) if total else 0.0,
    }


# ============================ M2: CWE attribution accuracy ============================


def _top1_cwe(rec: dict) -> str:
    """CWE of the highest-confidence confirmed finding (top-1 attribution).

    pred_cwes cannot be used directly -- it is a cross-finding deduplicated union,
    so reporting 3 CWEs yields 3 chances to hit, which trades accuracy for
    over-reporting. top-1 is the comparable attribution metric; any-of-k is
    reported separately. vuln_findings contains suspected entries (below
    threshold), so it is filtered back to the confirmed set via pred_cwes."""
    confirmed = {norm_cwe(c) for c in (rec.get("pred_cwes") or [])}
    cands = [d for d in (rec.get("vuln_findings") or [])
             if norm_cwe(d.get("cwe")) in confirmed]
    if not cands:
        return next(iter(sorted(confirmed)), "")
    return norm_cwe(max(cands, key=lambda d: float(d.get("conf") or 0.0)).get("cwe"))


def _attribution_rows(pairs: list[dict], tax: Taxonomy,
                      drop_unmapped: bool) -> list[dict]:
    """Sample set for M2/M3: **vulnerable files M1 got right** (a conditional
    metric; the denominator excludes misses).

    Conditioning avoids the trap in bench.py: ANDing the CWE condition into the
    confusion matrix can only shrink TP, so "adding a label requirement"
    necessarily makes detection look worse arithmetically and the two views are
    not comparable. Here attribution is scored only on samples where detection
    succeeded, keeping the metrics mutually independent."""
    rows = []
    for p in pairs:
        v = p["vuln"]
        if v is None or not v.get("pred_vuln"):
            continue
        gt = norm_cwe(v.get("gt_cwe"))
        in_space = tax.in_label_space(gt)
        if drop_unmapped and not in_space:
            continue
        preds = [norm_cwe(c) for c in (v.get("pred_cwes") or []) if norm_cwe(c)]
        top1 = _top1_cwe(v)
        rows.append({
            "pair_id": p["pair_id"], "file": v.get("file"), "ctx": p["ctx"],
            "gt_cwe": gt, "in_label_space": in_space,
            "top1": top1, "n_pred": len(preds),
            "rel_top1": tax.relation(gt, top1) if top1 else MISS,
            "rel_anyk": tax.best_relation(gt, preds),
            "same_family_top1": tax.same_family(gt, top1) if top1 else False,
            "same_family_anyk": any(tax.same_family(gt, pred) for pred in preds),
        })
    return rows


def m2_attribution(rows: list[dict]) -> dict:
    """M2: exact-CWE and same-family attribution. Denominator = the number of
    vulnerable files M1 got right."""
    n = len(rows)
    top1 = sum(r["rel_top1"] == "exact" for r in rows)
    anyk = sum(r["rel_anyk"] == "exact" for r in rows)
    family_top1 = sum(r["same_family_top1"] for r in rows)
    family_anyk = sum(r["same_family_anyk"] for r in rows)
    return {
        "denominator": n,
        "denominator_note": "vulnerable files M1 got right (misses excluded)",
        "top1_exact": top1, "anyk_exact": anyk,
        "top1_accuracy": round(top1 / n, 4) if n else 0.0,
        "anyk_accuracy": round(anyk / n, 4) if n else 0.0,
        "top1_family": family_top1, "anyk_family": family_anyk,
        "top1_family_accuracy": round(family_top1 / n, 4) if n else 0.0,
        "anyk_family_accuracy": round(family_anyk / n, 4) if n else 0.0,
        "mean_preds_per_file": round(sum(r["n_pred"] for r in rows) / n, 2) if n else 0.0,
    }


# ================================ M3: tolerance curve =================================


def m3_tolerance(rows: list[dict]) -> dict:
    """M3: treats tolerance as the independent variable and reports cumulative hit
    rate. Levels nest strict to loose: exact ⊂ hierarchy ⊂ chain ⊂ family.

    The **increment** at each level is where the information lies: hierarchy's
    increment is the volume that would be penalized as "prediction more specific
    than GT"; chain's increment is same-mechanism cross-bucket pairs split apart
    by family buckets (e.g. CWE-190↔CWE-787); family's increment is pairs in the
    same bucket with no explicit relation. bench.py's cwe-family view reports only
    the single loosest number, which hides all of this."""
    n = len(rows)
    dist_top1 = Counter(r["rel_top1"] for r in rows)
    dist_anyk = Counter(r["rel_anyk"] for r in rows)
    curve = []
    cum1 = cumk = 0
    for lv in LEVELS:
        cum1 += dist_top1.get(lv, 0)
        cumk += dist_anyk.get(lv, 0)
        curve.append({
            "level": lv,
            "top1_delta": dist_top1.get(lv, 0), "anyk_delta": dist_anyk.get(lv, 0),
            "top1_cumulative": round(cum1 / n, 4) if n else 0.0,
            "anyk_cumulative": round(cumk / n, 4) if n else 0.0,
        })
    return {
        "denominator": n, "curve": curve,
        "unrelated_top1": dist_top1.get(MISS, 0),
        "unrelated_anyk": dist_anyk.get(MISS, 0),
    }


# ======================== Single-file determinable pairs: dump ========================


def dump_detectable(det_manifest: Path, pred_by_file: dict[str, dict],
                    tax: Taxonomy, path: Path,
                    only_detected: bool = False) -> dict:
    """Dump the "single-file context suffices" vulnerable pairs to JSONL, each
    with the detection outcome measured in this run.

    The roster is read **directly from the det manifest**, never back-inferred
    from predictions -- so even if predictions cover only part of the run (or are
    still running), the dumped roster stays exactly consistent with the current
    manifest, and pairs that were not reached record `detected` as null instead
    of silently disappearing.

    "Detectable" has three layers, all kept in one file and separated by field:
      · the roster itself          -- **determinable** at the dataset level
                                     (single-file context suffices)
      · detected == true          -- this run **actually detected** the vuln side
      · m1_pair_class=="P-C"      -- vuln side detected and patched side not a
                                     false positive (strictest)
    --only-detected dumps only pairs whose detected flag is true.

    The roster source is recorded as data/primevul_det/manifest.jsonl; its
    construction and review process belong in the dataset artifact. That path is
    recorded in the returned stat.ctx_source for traceability."""
    groups: dict[str, dict] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for ln in det_manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        pid = d.get("pair_id")
        if not pid:
            continue
        groups[pid]["vuln" if d.get("is_vulnerable") else "patched"] = d.get("file")
        meta.setdefault(pid, {"gt_cwe": d.get("cwe"), "project": d.get("project"),
                              "cve": d.get("cve"), "commit_id": d.get("commit_id")})

    rows = []
    for pid in sorted(groups):
        vf, pf = groups[pid].get("vuln"), groups[pid].get("patched")
        vp, pp = pred_by_file.get(vf or ""), pred_by_file.get(pf or "")
        detected = bool(vp.get("pred_vuln")) if vp else None
        if only_detected and not detected:
            continue
        gt = norm_cwe(meta[pid]["gt_cwe"])
        preds = [c for c in (norm_cwe(x) for x in ((vp or {}).get("pred_cwes") or [])) if c]
        top1 = _top1_cwe(vp) if vp else ""
        if vp is None or pp is None:
            cls = "not_run"  # predictions do not cover this pair (unreached / error)
        else:
            p_ok = not bool(pp.get("pred_vuln"))
            cls = "P-C" if (detected and p_ok) else ("P-V" if detected else
                                                    ("P-B" if p_ok else "P-R"))
        rows.append({
            "pair_id": pid,
            "ctx": "determinable",
            "gt_cwe": gt,
            "gt_in_label_space": tax.in_label_space(gt),
            "project": meta[pid]["project"], "cve": meta[pid]["cve"],
            "vuln_file": vf, "patched_file": pf,
            "detected": detected,
            "patched_false_positive": (bool(pp.get("pred_vuln")) if pp else None),
            "m1_pair_class": cls,
            "pred_cwes": preds,
            "top1_cwe": top1,
            "cwe_level_top1": tax.relation(gt, top1) if top1 else MISS,
            "cwe_level_anyk": tax.best_relation(gt, preds) if preds else MISS,
            "n_findings": (vp or {}).get("n_findings"),
            "vuln_resolution": (vp or {}).get("file_resolution"),
            "patched_resolution": (pp or {}).get("file_resolution"),
            "elapsed_s": (vp or {}).get("elapsed_s"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return {
        "path": str(path), "ctx_source": str(det_manifest),
        "pairs": len(rows),
        "with_prediction": sum(r["detected"] is not None for r in rows),
        "detected": sum(r["detected"] is True for r in rows),
        "pair_correct": sum(r["m1_pair_class"] == "P-C" for r in rows),
        "exact_cwe": sum(r["cwe_level_top1"] == "exact" for r in rows),
        "only_detected": only_detected,
    }


# ================================ Label-space coverage ================================


def label_space(pairs: list[dict], tax: Taxonomy) -> dict:
    """Statistics over samples whose GT CWE falls outside the KB's 96-entry label
    space.

    These are mostly Category or deprecated entries (CWE-189 Numeric Errors,
    CWE-399 Resource Management, CWE-17 DEPRECATED, etc.), which a
    defect-level predictor **cannot** emit; folding them into the M2/M3
    denominator would deduct an unreachable score. They are dropped by default
    (--keep-unmapped retains them), and the dropped count is reported here."""
    out_of_space: Counter = Counter()
    total = 0
    for p in pairs:
        v = p["vuln"]
        if v is None:
            continue
        total += 1
        gt = norm_cwe(v.get("gt_cwe"))
        if not tax.in_label_space(gt):
            out_of_space[gt or "(empty)"] += 1
    n_out = sum(out_of_space.values())
    return {
        "vuln_files": total, "kb_entries": len(tax.known),
        "out_of_space": n_out,
        "coverage": round((total - n_out) / total, 4) if total else 0.0,
        "out_of_space_cwes": out_of_space.most_common(),
    }


# =============================== Stratified aggregation ===============================


def evaluate(pairs: list[dict], tax: Taxonomy, drop_unmapped: bool) -> dict:
    """All samples plus a context-sufficiency stratification, computing the three
    metrics in each.

    Stratification only describes conditional performance inside and outside a
    given context-sufficiency roster. It establishes no capability ceiling and
    does not isolate confounders such as model, vulnerability class, or
    difficulty, so the difference between the two layers must not be read as a
    causal effect of context."""
    def _one(subset: list[dict]) -> dict:
        rows = _attribution_rows(subset, tax, drop_unmapped)
        return {
            "pairs": len(subset),
            "m1_detection": m1_detection(subset),
            "m2_attribution": m2_attribution(rows),
            "m3_tolerance": m3_tolerance(rows),
        }

    strata: dict[str, dict] = {"overall": _one(pairs)}
    layers = sorted({p["ctx"] for p in pairs})
    if layers != ["all"]:
        for layer in layers:
            strata[layer] = _one([p for p in pairs if p["ctx"] == layer])
    return {"strata": strata, "label_space": label_space(pairs, tax)}


# ====================================== Printing ======================================

_LEVEL_DESC = {
    "exact": "identical CWE number",
    "hierarchy": "ancestor/descendant (incl. 'prediction more specific than GT')",
    "chain": f"same-mechanism chain ≤{CHAIN_MAX_HOPS} hops (e.g. CWE-190→CWE-787)",
    "family": "same top-level family bucket only",
}


def _print_stratum(name: str, s: dict) -> None:
    print(f"\n{'═' * 68}\n【{name}】pairs={s['pairs']}\n{'═' * 68}")

    d = s["m1_detection"]
    f, pl = d["file_level"], d["pair_level"]
    print(f"\n─ M1 detection (true/false, CWE not consulted) ─")
    print(f"  file level N={f['N']}  TP={f['TP']} FP={f['FP']} TN={f['TN']} FN={f['FN']}")
    print(f"    accuracy={f['accuracy']}  precision={f['precision']}  "
          f"recall={f['recall']}  f1={f['f1']}  fpr={f['fpr']}")
    print(f"  Pair count={pl['pairs_scored']}"
          + (f" (incomplete {pl['pairs_incomplete']} skipped)" if pl["pairs_incomplete"] else ""))
    print(f"    P-C={pl['P-C']} P-V={pl['P-V']} P-B={pl['P-B']} P-R={pl['P-R']}  "
          f"pair_accuracy={pl['pair_accuracy']}")
    p1 = d["evidence_backed_underdetermined_pair"]
    print(f"  P-C1 (side channel: an evidence-backed underdetermined counts as a"
          f" vuln-side risk hit; M1 unchanged)={p1['P-C1']} "
          f"= strict P-C {p1['P-C_strict']} + P-U {p1['P-U']}; "
          f"pair_accuracy={p1['pair_accuracy']}")
    if p1["eligible_underdetermined_blocked_by_patch"] or p1["vulnerable_underdetermined_without_evidence"]:
        print(f"    evidence-backed underdetermined but patched side positive="
              f"{p1['eligible_underdetermined_blocked_by_patch']};"
              f" vuln-side underdetermined without evidence="
              f"{p1['vulnerable_underdetermined_without_evidence']}")
    ab = d.get("abstain") or {}
    if ab and (ab.get("underdetermined_share") or ab.get("legacy")):
        print(f"  Negative composition (three-valued side channel; views unchanged):")
        print(f"    among FN: underdetermined={ab.get('fn_underdetermined', 0)} / "
              f"safe_proved={ab.get('fn_safe_proved', 0)};"
              f" among TN: underdetermined={ab.get('tn_underdetermined', 0)} / "
              f"safe_proved={ab.get('tn_safe_proved', 0)}")
        print(f"    undetermined as a share of all negatives="
              f"{ab.get('underdetermined_share', 0.0)}"
              + (f" (records with no verdict field: {ab['legacy']}, excluded)"
                 if ab.get("legacy") else ""))

    a = s["m2_attribution"]
    print(f"\n─ M2 CWE attribution (exact hits, denominator={a['denominator']}:"
          f" {a['denominator_note']}) ─")
    print(f"    top-1  ={a['top1_accuracy']}  ({a['top1_exact']}/{a['denominator']})")
    print(f"    any-of-k={a['anyk_accuracy']}  ({a['anyk_exact']}/{a['denominator']})"
          f"   mean CWEs reported per file={a['mean_preds_per_file']}")
    print(f"    same family top-1={a['top1_family_accuracy']}"
          f"  ({a['top1_family']}/{a['denominator']});"
          f" any-of-k={a['anyk_family_accuracy']}"
          f"  ({a['anyk_family']}/{a['denominator']})")

    t = s["m3_tolerance"]
    print(f"\n─ M3 tolerance curve (cumulative, denominator={t['denominator']}) ─")
    print(f"    {'level':<10} {'delta(top1/anyk)':<16} {'cum top1':<10} "
          f"{'cum anyk':<10} meaning")
    for row in t["curve"]:
        delta = "+{}/+{}".format(row["top1_delta"], row["anyk_delta"])
        print(f"    {row['level']:<10} {delta:<16} "
              f"{row['top1_cumulative']:<10} {row['anyk_cumulative']:<10} "
              f"{_LEVEL_DESC[row['level']]}")
    delta = "+{}/+{}".format(t["unrelated_top1"], t["unrelated_anyk"])
    print(f"    {MISS:<10} {delta:<16} {'—':<10} {'—':<10} no determinable relation")


def print_report(result: dict, pred_path: Path, det_path: Optional[Path],
                 drop_unmapped: bool) -> None:
    print(f"\n╔{'═' * 66}╗")
    print(f"║ Three-metric evaluation: {str(pred_path):<44}║")
    print(f"╚{'═' * 66}╝")

    ls = result["label_space"]
    print(f"\n─ Label-space coverage (KB has {ls['kb_entries']} CWE entries) ─")
    print(f"    vulnerable files {ls['vuln_files']}, GT off-KB {ls['out_of_space']} "
          f"(coverage {ls['coverage']})"
          + (", dropped from M2/M3" if drop_unmapped
             else ", retained in M2/M3 (--keep-unmapped)"))
    if ls["out_of_space_cwes"]:
        preview = ", ".join(f"{c}x{n}" for c, n in ls["out_of_space_cwes"][:8])
        print(f"    GT outside the KB (mostly Category/deprecated entries that a"
              f" defect-level predictor cannot hit): {preview}"
              + (" ..." if len(ls["out_of_space_cwes"]) > 8 else ""))

    strata = result["strata"]
    if det_path is None or "determinable" not in strata:
        print(f"\n⚠ Context-sufficiency subset not loaded, skipping stratification"
              f" (needs {det_path or 'data/primevul_det/manifest.jsonl'})")
    for name in ("overall", "determinable", "undeterminable"):
        if name in strata:
            _print_stratum({"overall": "all",
                            "determinable": "context suffices (one file decides)",
                            "undeterminable": "insufficient context (needs"
                                              " cross-file/contract info)"}[name],
                           strata[name])

    det, und = strata.get("determinable"), strata.get("undeterminable")
    if det and und:
        rd = det["m1_detection"]["file_level"]["recall"]
        ru = und["m1_detection"]["file_level"]["recall"]
        print(f"\n─ Conditional difference under the context-sufficiency roster"
              f" (descriptive) ─")
        print(f"    recall: determinable stratum {rd} vs undeterminable stratum {ru}"
              f"  difference {round(rd - ru, 4)}")
        print(f"    This difference only describes conditional performance under the"
              f" current roster; it is not a capability ceiling nor a causal effect"
              f" of context.")


# =============================== Per-pair detail export ===============================


def write_per_pair(pairs: list[dict], tax: Taxonomy, path: Path) -> None:
    """Per-pair detail CSV: the full decision process for all three metrics is
    dumped, so results can be inspected manually and bad cases extracted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "pair_id", "ctx_sufficiency", "gt_cwe", "gt_in_label_space",
            "vuln_file", "vuln_pred", "vuln_pred_cwes", "vuln_top1_cwe",
            "patched_file", "patched_pred", "patched_pred_cwes",
            "m1_pair_class", "m1u_pair_class", "vuln_evidence_backed_underdetermined",
            "m2_top1_exact", "m2_top1_same_family", "m2_anyk_same_family",
            "m3_level_top1", "m3_level_anyk",
        ])
        for p in sorted(pairs, key=lambda x: x["pair_id"]):
            v, pt = p["vuln"], p["patched"]
            gt = norm_cwe((v or {}).get("gt_cwe") or (pt or {}).get("gt_cwe"))
            preds = [norm_cwe(c) for c in ((v or {}).get("pred_cwes") or [])]
            preds = [c for c in preds if c]
            top1 = _top1_cwe(v) if v else ""
            underdetermined_hit = _evidence_backed_underdetermined(v)
            if not p["complete"]:
                cls = "incomplete"
                cls_u = "incomplete"
            else:
                v_ok = bool(v.get("pred_vuln"))
                p_ok = not bool(pt.get("pred_vuln"))
                cls = "P-C" if (v_ok and p_ok) else ("P-V" if v_ok else
                                                     ("P-B" if p_ok else "P-R"))
                cls_u = "P-C1/P-C" if (v_ok and p_ok) else (
                    "P-C1/P-U" if (underdetermined_hit and p_ok) else "not-P-C1")
            w.writerow([
                p["pair_id"], p["ctx"], gt, tax.in_label_space(gt),
                (v or {}).get("file", ""), (v or {}).get("pred_vuln", ""),
                "|".join(preds), top1,
                (pt or {}).get("file", ""), (pt or {}).get("pred_vuln", ""),
                "|".join(norm_cwe(c) for c in ((pt or {}).get("pred_cwes") or [])),
                cls, cls_u, underdetermined_hit,
                (tax.relation(gt, top1) == "exact") if top1 else False,
                tax.same_family(gt, top1) if top1 else False,
                any(tax.same_family(gt, pred) for pred in preds),
                tax.relation(gt, top1) if top1 else MISS,
                tax.best_relation(gt, preds) if preds else MISS,
            ])


# ======================================== CLI =========================================


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Three-metric independent evaluation: M1 detection / M2 CWE"
                    " attribution / M3 tolerance curve, stratified by context"
                    " sufficiency (score-only, zero token)")
    ap.add_argument("--pred", required=True,
                    help="predictions JSONL (produced by vuln_agent.bench)")
    ap.add_argument("--determinable",
                    default="data/primevul_det/manifest.jsonl",
                    help="manifest of the 'context suffices' subset; if absent,"
                         " stratification is skipped")
    ap.add_argument("--no-strata", action="store_true",
                    help="disable context-sufficiency stratification")
    ap.add_argument("--keep-unmapped", action="store_true",
                    help="M2/M3 retain samples whose GT CWE falls outside the KB"
                         " label space (dropped by default)")
    ap.add_argument("--kb-dir", default=None,
                    help="CWE knowledge base directory (default config.KB_DIR)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write all metrics as JSON")
    ap.add_argument("--per-pair", default=None, help="per-pair detail CSV")
    ap.add_argument("--dump-detectable", default=None,
                    help="dump the 'single-file context suffices' pairs to JSONL"
                         " (with the measured detected / m1_pair_class / CWE level)")
    ap.add_argument("--only-detected", action="store_true",
                    help="with --dump-detectable: dump only pairs whose vulnerable"
                         " side was actually detected in this run")
    args = ap.parse_args(argv)

    pred_path = Path(args.pred)
    if not pred_path.exists():
        print(f"✖ prediction file not found: {pred_path}", file=sys.stderr)
        return 1

    recs = _load_pred(pred_path)
    ok = [r for r in recs if r.get("error") is None]
    if len(ok) < len(recs):
        print(f"note: {len(recs) - len(ok)} error records dropped")

    tax = Taxonomy(Path(args.kb_dir) if args.kb_dir else config.KB_DIR)
    det_path = None if args.no_strata else Path(args.determinable)

    # --dump-detectable needs only the det roster, not prediction volume: when
    # predictions are empty or half-run it still dumps the full roster, with
    # `detected` as null for pairs not reached. So it is handled before the
    # "no valid predictions" early return.
    if args.dump_detectable:
        if det_path is None or not det_path.exists():
            print(f"✖ --dump-detectable requires the context-sufficiency roster"
                  f" ({args.determinable} does not exist)", file=sys.stderr)
            return 1
        st = dump_detectable(det_path, {r.get("file", ""): r for r in ok},
                            tax, Path(args.dump_detectable), args.only_detected)
        print(f"\nsingle-file determinable pairs -> {st['path']}"
              f" (roster source {st['ctx_source']})")
        print(f"    dumped {st['pairs']} pairs"
              + (" (filtered by --only-detected)" if st["only_detected"] else "")
              + f", of which {st['with_prediction']} have prediction coverage:"
                f" detected {st['detected']},"
                f" pair fully correct (P-C) {st['pair_correct']},"
                f" CWE top-1 exact {st['exact_cwe']}")

    if not ok:
        print(f"\n✖ {pred_path} has no valid predictions ({len(recs)} records),"
              f" skipping the three-metric computation",
              file=sys.stderr)
        return 0 if args.dump_detectable else 1

    determinable = _load_determinable(det_path)
    pairs = _build_pairs(ok, determinable)
    drop_unmapped = not args.keep_unmapped

    result = evaluate(pairs, tax, drop_unmapped)
    result["meta"] = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "pred": str(pred_path), "pred_sha256": _sha256_file(pred_path),
        "records": len(recs), "valid": len(ok),
        "kb_dir": str(tax.kb_dir), "chain_max_hops": CHAIN_MAX_HOPS,
        "drop_unmapped": drop_unmapped,
        "determinable_source": str(det_path) if determinable else None,
    }
    print_report(result, pred_path, det_path if determinable else None, drop_unmapped)

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nmetrics JSON -> {p}")
    if args.per_pair:
        write_per_pair(pairs, tax, Path(args.per_pair))
        print(f"per-pair detail CSV -> {args.per_pair}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
