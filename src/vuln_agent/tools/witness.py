"""witness: machine-checked witness verifier (replaces "comparative judgement"
with "a machine-decidable out-of-bounds witness").

**Why it is needed**: on a PrimeVul pair the vulnerable and the patched version are
nearly identical text, so an LLM can take a shortcut by latching onto surface
statistics (which version is shorter, which version added code) instead of
understanding the code. The witness mechanism closes that path: the LLM does not
answer "which one is vulnerable" at all, it must submit a **witness**

    {sink_line, sink_expr, path: [{node|line, take}], precondition, overflow: {condition}}

which this module puts through four deterministic checks (no LLM, zero tokens):

1. **Path existence**: the decision sequence is really walkable on the tree-sitter
   CFG, and the sink line is reachable from the path end;
2. **Constraint consistency**: the path conditions (**extracted by the checker
   itself from the CFG branch text**, never trusting the LLM's paraphrase),
   conjoined with the LLM-submitted precondition, are not contradictory;
3. **Overflow holds**: adding the "unsafe condition" to the constraint store still
   yields no contradiction. The patched version is rejected precisely because that
   `if` left an interval in the path condition that contradicts the unsafe condition
   -- **the asymmetry comes from the decision procedure, not from the LLM's taste**;
4. **Taint reachability**: the variables in the unsafe condition trace back through
   the def-use chain to entry parameters (an over-approximated necessary condition).

**Honest incompleteness** (must be stated whenever this is written up or reviewed):
- The solver is a **refuter**, not a complete SAT decision procedure: the constraint
  domain is integer intervals plus inter-variable difference propagation. A proof of
  contradiction refutes; failure to prove a contradiction is recorded as
  `sat="undecided"` and does not mean the path is really feasible. A passing witness
  therefore supports only "an out-of-bounds witness exists that was not refuted"; it
  does not claim a reachability proof.
- Atoms the solver cannot parse (nonlinear / bitwise / pointer arithmetic / struct
  fields) are recorded in `unparsed` and **ignored** (conservative for the refutation
  direction: dropping constraints can only weaken refutation, never add a false
  positive).
- Macros and conditional compilation are not expanded, so dominating guards may be
  under-counted; `refute_unsafe` is therefore used only for **SATISFIED bonus
  evidence**, and its "cannot prove" is never turned back into vulnerability evidence.
"""
from __future__ import annotations

import re
from typing import Optional

from .cfg import CFG, idents_of

_ZERO = "#0"  # virtual node standing for the constant 0 in the difference graph

# Atomic comparison operators (longer ones first, so "<=" is not split into "<").
_OPS = ("<=", ">=", "==", "!=", "<", ">")

_NUM = re.compile(r"^[+-]?(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*$")
_IDENT_ONLY = re.compile(r"^[A-Za-z_]\w*$")
# Pointer cast prefixes: `(char *)` / `(const StringInfo *)` / `(unsigned char **)`.
# Only **pointer** casts are stripped: they do not change the value of comparisons
# such as "is it NULL". Integral casts (`(int)` / `(ssize_t)`) truncate or change
# sign, and stripping them could fabricate a contradiction that does not hold
# (-> a false SATISFIED), so they are never stripped.
_PTR_CAST = re.compile(r"\(\s*[A-Za-z_][\w\s]*\*[\s*]*\)\s*(?=[A-Za-z_0-9(&])")
# Lvalue chains: `image->columns` / `x.size` / `data[0]` / `a->b[2]`. The whole chain is
# encoded as **one opaque integer symbol**, valid because the identifiers on the chain
# are never assigned in this function (see `_symbolize`).
_LVALUE_CHAIN = re.compile(
    r"[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*|\s*\[\s*\w+\s*\])+")

_NEG = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


def _sym_name(chain: str) -> str:
    """Map an lvalue chain to a stable symbol name (the same chain must get the
    same name in a guard and in the unsafe condition)."""
    norm = re.sub(r"\s+", "", chain)
    return "lv_" + re.sub(r"\W", "_", norm)


def _symbolize(text: str, assigned: Optional[set] = None) -> str:
    """Deterministic rewriting before constraint encoding: strip pointer casts and
    replace **read-only lvalue chains** with opaque integer symbols.

    Motivation: real guards frequently take forms such as
    `if (profile == (const StringInfo *) NULL)`, `if (x < (ssize_t) image->columns)`,
    `if (data[0] > 3)`. `parse_term` returns None whenever it meets a parenthesised
    cast or a `->`/`.`/`[]` access, so without this rewriting the whole guard would
    be ignored -- and an out-of-bounds witness that this guard should refute would
    then pass the refutation stage, leaving the refuter idle.

    Both rewrites must be reliable, hence:
      · Only **pointer** casts (`(char *)`) are stripped. An integral cast truncates
        or changes sign, and stripping it could fabricate a contradiction, so it is
        kept (keeping it => the atom fails to parse => it is ignored => the
        refutation is merely weaker).
      · An lvalue chain is symbolised only when none of its identifiers has been
        assigned in this function (`assigned` comes from `cfg.defs()`). Once
        written, it is not replaced: the same chain may hold different values in two
        places, and treating it as one symbol would fabricate a contradiction.
        When `assigned` is None (no CFG) no symbolisation happens -- weaker
        refutation is preferred.
    """
    s = _PTR_CAST.sub("", text or "")
    if assigned is None:
        return s

    def repl(m: re.Match) -> str:
        chain = m.group(0)
        names = set(re.findall(r"[A-Za-z_]\w*", chain))
        if names & assigned:
            return chain
        return _sym_name(chain)

    return _LVALUE_CHAIN.sub(repl, s)


def _as_int(tok: str) -> Optional[int]:
    tok = tok.strip()
    if not _NUM.match(tok):
        return None
    tok = tok.rstrip("uUlL")
    try:
        return int(tok, 0)                    # auto-detects 0x / decimal, with sign
    except ValueError:
        pass
    try:
        return int(tok, 8)                    # C octal literal 0755
    except ValueError:
        return None


def _split_top(text: str, seps: tuple[str, ...]) -> Optional[list[tuple[str, str]]]:
    """Split on top-level (outside parentheses) separators, returning
    [(preceding separator, fragment)]; unbalanced parentheses return None."""
    out: list[tuple[str, str]] = []
    depth = 0
    cur = ""
    sep = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                return None
        if depth == 0:
            hit = next((s for s in seps if text.startswith(s, i)), "")
            if hit:
                out.append((sep, cur))
                sep, cur = hit, ""
                i += len(hit)
                continue
        cur += ch
        i += 1
    if depth != 0:
        return None
    out.append((sep, cur))
    return out


def _strip_parens(text: str) -> str:
    s = (text or "").strip()
    while s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        if _split_top(inner, ("",)) is None:  # unbalanced parens: cannot strip
            break
        depth = 0
        ok = True
        for ch in inner:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
                if depth < 0:
                    ok = False
                    break
        if not ok or depth != 0:
            break
        s = inner.strip()
    return s


def parse_term(text: str) -> Optional[tuple[Optional[str], int]]:
    """Parse `len + 1` / `64` / `size` into (variable name | None, constant offset).
    Multiplication/division, bitwise ops, multiple variables, field access -> None
    (the caller records unparsed and ignores it)."""
    s = _strip_parens(text)
    if not s:
        return None
    parts = _split_top(s, ("+", "-"))
    if parts is None:
        return None
    var: Optional[str] = None
    const = 0
    for sep, raw in parts:
        piece = _strip_parens(raw)
        if not piece:
            return None
        if piece in ("NULL", "nullptr"):     # null pointer constant encoded as 0
            piece = "0"                       # (deref witnesses need this)
        sign = -1 if sep == "-" else 1
        num = _as_int(piece)
        if num is not None:
            const += sign * num
            continue
        if not _IDENT_ONLY.match(piece) or sign < 0 or var is not None:
            return None                      # multi-var / negated var / not a bare ident
        var = piece
    return var, const


def parse_atom(text: str) -> Optional[list[tuple[str, str, int]]]:
    """Parse one comparison atom into a difference-constraint list `u - v <= w`
    (u/v are variable names or _ZERO).

    Supports `a < b+1` / `n >= 0` / `p == NULL` / `!p`. `!=` and bare-identifier truth
    tests are left to parse_diseq (difference logic cannot express disequality, so it
    forms a separate class). Multiplication/division, bitwise ops, or multi-variable
    linear combinations -> return None (the caller records unparsed and ignores that
    atom).
    """
    s = _strip_parens(text)
    if not s:
        return None
    if s.startswith("!") and _IDENT_ONLY.match(_strip_parens(s[1:])):
        s = _strip_parens(s[1:]) + " == 0"    # `!p` => p == 0
    parts = None
    op = ""
    for cand in _OPS:
        parts = _split_top(s, (cand,))
        if parts and len(parts) == 2:
            op = cand
            break
        parts = None
    if not parts or op == "!=":               # no comparison / disequality
        return None                           # (handled by parse_diseq)
    lhs, rhs = parts[0][1], parts[1][1]
    a, b = parse_term(lhs), parse_term(rhs)
    if a is None or b is None:
        return None
    (va, ca), (vb, cb) = a, b
    u = va or _ZERO
    v = vb or _ZERO
    if u == v:                                # same var both sides: constant truth test
        u = v = _ZERO
    delta = cb - ca                           # (va + ca) op (vb + cb) => va - vb op delta
    if op == "<":
        return [(u, v, delta - 1)]
    if op == "<=":
        return [(u, v, delta)]
    if op == ">":
        return [(v, u, -delta - 1)]
    if op == ">=":
        return [(v, u, -delta)]
    if op == "==":
        return [(u, v, delta), (v, u, -delta)]
    return None


def parse_diseq(text: str) -> Optional[tuple[str, int]]:
    """Parse a "disequality" atom, returning (variable, constant) for `var != const`.

    Difference logic cannot express disequality, yet this is a **key shape** for the
    task: a non-null guard `if (!p) return;` establishes exactly `p != 0` on reaching
    the sink, while the unsafe condition of a deref witness is `p == 0` -- the two are
    contradictory once conjoined, so the patched version is deterministically refuted.
    Supports `x != c`, `c != x`, and the bare-identifier truth test `p`.
    """
    s = _strip_parens(text)
    if not s:
        return None
    if _IDENT_ONLY.match(s):                  # `if (p)` => p != 0
        return s, 0
    parts = _split_top(s, ("!=",))
    if not parts or len(parts) != 2:
        return None
    a, b = parse_term(parts[0][1]), parse_term(parts[1][1])
    if a is None or b is None:
        return None
    (va, ca), (vb, cb) = a, b
    if va and vb is None:
        return va, cb - ca
    if vb and va is None:
        return vb, ca - cb
    return None


def negate(text: str) -> str:
    """Negate one condition text (used for the else side of a path). `!x` => `x`;
    bare identifier `p` => `p == 0`; a single comparison atom flips its sign; a
    compound condition degenerates to `!(...)`, which yields no parsable atom and
    is therefore ignored (conservative for the refutation direction)."""
    s = _strip_parens(text)
    if s.startswith("!"):
        inner = _strip_parens(s[1:])
        if inner:
            return inner
    if _IDENT_ONLY.match(s):
        return f"{s} == 0"
    for cand in _OPS:
        parts = _split_top(s, (cand,))
        if parts and len(parts) == 2 and cand in _NEG:
            return f"{parts[0][1].strip()} {_NEG[cand]} {parts[1][1].strip()}"
    return f"!({s})"


def split_conjuncts(text: str) -> list[str]:
    """Split a condition on top-level `&&`; a whole condition containing a top-level
    `||` is dropped (a disjunction cannot be encoded conjunctively, and ignoring it
    is more conservative)."""
    s = _strip_parens(text)
    if not s:
        return []
    ors = _split_top(s, ("||",))
    if ors is None or len(ors) > 1:
        return []
    ands = _split_top(s, ("&&",))
    if ands is None:
        return []
    return [p.strip() for _sep, p in ands if p.strip()]


class ConstraintStore:
    """Difference-logic constraint store (`u - v <= w`) + disequality atoms
    (`v != c`) + a negative-cycle refuter.

    **Capability boundary**: this is a **refuter**, not a complete decision procedure.
    A detected negative cycle => the constraint set is unsatisfiable (a deterministic
    conclusion, usable for refutation). No negative cycle only means no contradiction
    was found inside the difference fragment, and is recorded as undecided. Unparsed
    atoms are recorded in `unparsed` and ignored -- ignoring them can only weaken the
    refutation, it can never manufacture a positive.

    Disequality atoms form a separate class: difference logic cannot express `!=`,
    but it is the core shape of a non-null guard. Contradiction is decided by
    "if the constraints pin v to exactly c (v<=c and v>=c close v in from both
    sides), then v != c is unsatisfiable" -- still only the negative-cycle mechanism,
    and the conclusion is replayable step by step.
    """

    def __init__(self, assigned: Optional[set] = None) -> None:
        self.edges: list[tuple[str, str, int, str]] = []   # (u, v, w, source text)
        self.diseqs: list[tuple[str, int, str]] = []       # (var, const, source)
        self.unparsed: list[str] = []
        self.vars: set[str] = set()
        # Identifiers assigned in this function (from cfg.defs()): only read-only
        # lvalue chains may be symbolised, see _symbolize.
        self.assigned: Optional[set] = assigned

    def add(self, cond: str, *, origin: str = "") -> bool:
        """Add one condition (automatically split on &&). Returns whether at least
        one atom was encoded successfully."""
        got = False
        atoms = split_conjuncts(_symbolize(cond, self.assigned))
        if not atoms:
            self.unparsed.append(origin or cond)
            return False
        for atom in atoms:
            triples = parse_atom(atom)
            if triples:
                for u, v, w in triples:
                    self.edges.append((u, v, w, origin or atom))
                    self.vars.update({u, v})
                got = True
                continue
            neq = parse_diseq(atom)
            if neq:
                self.diseqs.append((neq[0], neq[1], origin or atom))
                self.vars.add(neq[0])
                got = True
                continue
            self.unparsed.append(f"{origin or cond} :: {atom}")
        return got

    def clone(self) -> "ConstraintStore":
        new = ConstraintStore(self.assigned)
        new.edges = list(self.edges)
        new.diseqs = list(self.diseqs)
        new.unparsed = list(self.unparsed)
        new.vars = set(self.vars)
        return new

    def _neg_cycle(self, edges: list[tuple[str, str, int, str]]) -> tuple[bool, list[str]]:
        if not edges:
            return False, []
        nodes = sorted({u for u, _v, _w, _o in edges} | {v for _u, v, _w, _o in edges}
                       | {_ZERO})
        dist = {n: 0 for n in nodes}          # all-zero init: a virtual source node
        pred: dict[str, tuple[str, str]] = {}
        for _ in range(len(nodes)):
            changed = False
            for u, v, w, origin in edges:
                # u - v <= w  =>  edge v --w--> u
                if dist[v] + w < dist[u]:
                    dist[u] = dist[v] + w
                    pred[u] = (v, origin)
                    changed = True
            if not changed:
                return False, []
        for u, v, w, origin in edges:          # still relaxing at |V| => negative cycle
            if dist[v] + w < dist[u]:
                trace, cur, seen = [origin], v, {u}
                while cur in pred and cur not in seen:
                    seen.add(cur)
                    prev, org = pred[cur]
                    trace.append(org)
                    cur = prev
                uniq: list[str] = []
                for t in trace:
                    if t not in uniq:
                        uniq.append(t)
                return True, uniq
        return False, []

    def refute(self) -> tuple[bool, list[str]]:
        """Whether refutation succeeds. Returns (refuted, constraint sources that
        participate in the contradiction)."""
        hit, trace = self._neg_cycle(self.edges)
        if hit:
            return True, trace
        for var, const, origin in self.diseqs:
            # v != const contradicts "the constraints pin v to exactly c":
            # v>=const+1 and v<=const-1 are both unsat.
            hi, t_hi = self._neg_cycle(
                self.edges + [(_ZERO, var, -(const + 1), f"probe {var}>={const + 1}")])
            if not hi:
                continue
            lo, t_lo = self._neg_cycle(
                self.edges + [(var, _ZERO, const - 1, f"probe {var}<={const - 1}")])
            if lo:
                srcs = [origin] + [t for t in t_hi + t_lo if not t.startswith("probe ")]
                uniq: list[str] = []
                for s in srcs:
                    if s not in uniq:
                        uniq.append(s)
                return True, uniq
        return False, []

    def satisfiable(self) -> str:
        """"unsat" (a contradiction was found, deterministic) | "undecided" (no
        contradiction inside the difference fragment)."""
        return "unsat" if self.refute()[0] else "undecided"


def assigned_idents(g: Optional[CFG]) -> Optional[set]:
    """The set of identifiers assigned in this function (the keys of `cfg.defs()`);
    returns None when there is no CFG.

    `_symbolize` uses it to decide whether an lvalue chain is read-only -- only then
    may the chain be encoded as an opaque constant symbol. Note that `x->y = 1` and
    `a[i] = 1` are recorded in `cfg._assignments` as `y` and `a` respectively, so
    chains containing those names are excluded; the direction is conservative.
    """
    if g is None or not g.nodes:
        return None
    return set(g.defs().keys())


# ---- witness verification entry point (four steps) ----

def path_conditions(g: CFG, decisions: list[dict]) -> list[tuple[str, str]]:
    """Translate the LLM-submitted decision sequence into (condition text, source
    label). **The condition text is taken from the CFG node's own text**, never
    from the LLM's paraphrase; the take=False side is negated automatically."""
    out: list[tuple[str, str]] = []
    for d in decisions or []:
        nid = str(d.get("node") or "")
        n = g.node(nid)
        if n is None:
            cands = [x for x in g.nodes_at_line(int(d.get("line") or 0))
                     if x.kind == "cond"]
            n = cands[0] if cands else None
        if n is None or n.kind != "cond":
            continue
        take = bool(d.get("take", d.get("branch", True)))
        cond = n.text if take else negate(n.text)
        out.append((cond, f"path {n.id}@{n.line}{'' if take else ' (else)'}"))
    return out


def guard_conditions(g: CFG, sink_line: int) -> list[tuple[str, str]]:
    """Guard conditions that necessarily hold on reaching the sink, **computed by
    the machine** (true dominators + the must-take side).

    This is the **only source** of the vulnerable/patched asymmetry: the patched
    `if (n > 64) return -1;` makes the else side mandatory, so `n <= 64` enters the
    constraint store and forms a negative cycle with the unsafe condition `n > 64`,
    and the witness is deterministically refuted; no such constraint can be computed
    for the vulnerable version, so its witness passes. The LLM cannot forge this
    because it does not participate.
    """
    out: list[tuple[str, str]] = []
    for gd in g.dominating_guards(sink_line):
        if gd.get("must_take") is None:
            continue                          # both sides reach the sink
                                                  # => no constraining force on it
        text = gd["text"] if gd["must_take"] else negate(gd["text"])
        out.append((text, f"guard {gd['id']}@{gd['line']}"))
    return out


def check_witness(g: Optional[CFG], w: dict) -> dict:
    """The four deterministic checks. Returns the report dict:

        {verified, steps:{path,consistent,overflow,taint}, reason, anchors[],
         sat, unparsed[], degraded[]}

    verified=True means "**an out-of-bounds witness exists that was not refuted**"
    (not a reachability proof); when verified=False, `reason` names the step that
    refuted it -- this is the reviewable machine verdict.
    """
    rep: dict = {
        "verified": False,
        "steps": {"path": False, "consistent": False, "overflow": False, "taint": False},
        "reason": "", "anchors": [], "sat": "undecided",
        "unparsed": [], "degraded": list(g.degraded) if g else ["CFG unavailable"],
    }
    if g is None or not g.entry:
        rep["reason"] = ("CFG unavailable (parse failure / function not located): "
                         "the witness cannot be machine-checked, the obligation "
                         "stays UNDERDETERMINED")
        return rep
    sink_line = int(w.get("sink_line") or 0)
    if not g.nodes_at_line(sink_line):
        rep["reason"] = (f"sink line {sink_line} is not in the CFG: the witness points "
                         "at a code location that does not exist")
        return rep

    # Step 1: the path is really walkable and the sink is reachable from its end.
    decisions = list(w.get("path") or [])
    ok, why = g.path_exists(decisions)
    if not ok:
        rep["reason"] = f"step1 path does not exist: {why}"
        return rep
    cur = g.entry
    for d in decisions:
        n = g.node(str(d.get("node") or "")) or next(
            (x for x in g.nodes_at_line(int(d.get("line") or 0))
             if x.kind == "cond"), None)
        if n is None:
            continue
        take = bool(d.get("take", d.get("branch", True)))
        cur = (n.cond_true if take else n.cond_false) or cur
    sink_ids = {x.id for x in g.nodes_at_line(sink_line)}
    if not (sink_ids & g.reachable(cur)):
        rep["reason"] = (f"step1 sink line {sink_line} is not reachable from the "
                         "claimed end of the path")
        return rep
    rep["steps"]["path"] = True
    rep["anchors"] = [f"{n.id}@{n.line}" for n in g.nodes_at_line(sink_line)]

    # Step 2: path conditions AND machine-computed guards AND the LLM precondition
    # must not be mutually contradictory once conjoined.
    store = ConstraintStore(assigned_idents(g))
    for cond, origin in path_conditions(g, decisions) + guard_conditions(g, sink_line):
        store.add(cond, origin=origin)
    pre = str(w.get("precondition") or "").strip()
    if pre:
        store.add(pre, origin="precondition")
    refuted, trace = store.refute()
    rep["unparsed"] = list(store.unparsed)
    if refuted:
        rep["reason"] = "step2 precondition contradicts the path conditions: " \
            + " ∧ ".join(trace)
        return rep
    rep["steps"]["consistent"] = True

    # Step 3: still not refutable after adding the unsafe condition => the
    # out-of-bounds witness holds inside the difference fragment.
    unsafe = str(w.get("unsafe") or w.get("overflow") or "").strip()
    if not unsafe:
        rep["reason"] = ("step3 the witness gives no unsafe condition (unsafe): "
                         "there is no checkable out-of-bounds assertion")
        return rep
    probe = store.clone()
    if not probe.add(unsafe, origin="unsafe"):
        rep["unparsed"] = list(probe.unparsed)
        rep["reason"] = (f"step3 the unsafe condition cannot be machine-checked "
                         f"(not encodable as interval/difference constraints): {unsafe}")
        return rep
    refuted, trace = probe.refute()
    rep["sat"] = "unsat" if refuted else "undecided"
    rep["unparsed"] = list(probe.unparsed)
    if refuted:
        rep["reason"] = ("step3 the out-of-bounds condition is refuted by the guards "
                         "(constraint contradiction): " + " ∧ ".join(trace))
        return rep
    rep["steps"]["overflow"] = True

    # Step 4: the variables in the unsafe condition must trace back to entry
    # parameters through the def-use chain (an over-approximated necessary condition).
    cand = sorted(v for v in idents_of(unsafe) if _IDENT_ONLY.match(v))
    if cand and not any(g.taint_reachable(v) for v in cand):
        rep["reason"] = (f"step4 no variable of the unsafe condition is entry-reachable "
                         f"({', '.join(cand)}): suspected local constants, "
                         "not attacker-controllable")
        return rep
    rep["steps"]["taint"] = True
    rep["verified"] = True
    rep["reason"] = ("all four checks passed: an out-of-bounds witness exists that "
                     "was not refuted")
    return rep


def guard_proves_safe(g: Optional[CFG], sink_line: int, unsafe: str) -> dict:
    """The opposite direction (SATISFIED): do the machine-computed dominating guards
    **refute** the unsafe condition?

    Only this step can prove "an out-of-bounds access at this sink is impossible" --
    its evidence is a negative cycle, replayable step by step. Since macros and
    conditional compilation are not expanded, guards may be **under-counted**;
    therefore a "cannot prove" from this function is not vulnerability evidence
    (the conservative reading of I3).
    """
    out = {"proved": False, "reason": "", "anchors": [], "unparsed": []}
    if g is None or not g.entry or not str(unsafe or "").strip():
        out["reason"] = "CFG unavailable or no unsafe condition given"
        return out
    store = ConstraintStore(assigned_idents(g))
    guards = guard_conditions(g, sink_line)
    for cond, origin in guards:
        store.add(cond, origin=origin)
    # If the guard set **itself** is already refutable, then adding any unsafe
    # condition is bound to "contradict" it -- that is a vacuous proof (the
    # contradiction comes from the guard encoding itself, not from it blocking the
    # out-of-bounds access). This is the only deterministic channel for writing
    # SATISFIED, so giving up is preferable to certifying safety vacuously.
    if store.refute()[0]:
        out["unparsed"] = list(store.unparsed)
        out["reason"] = ("the dominating guard constraints are themselves "
                         "unsatisfiable (encoding degenerated): refusing a "
                         "vacuous proof")
        return out
    if not store.add(unsafe, origin="unsafe"):
        out["unparsed"] = list(store.unparsed)
        out["reason"] = f"the unsafe condition is not encodable: {unsafe}"
        return out
    refuted, trace = store.refute()
    out["unparsed"] = list(store.unparsed)
    if refuted:
        out["proved"] = True
        out["anchors"] = [o for _c, o in guards]
        out["reason"] = ("the dominating guards contradict the unsafe condition: "
                         + " ∧ ".join(trace))
    else:
        out["reason"] = ("no dominating guard contradicting the unsafe condition was "
                         "found (this does not mean safe; read it conservatively as "
                         "UNDERDETERMINED)")
    return out
