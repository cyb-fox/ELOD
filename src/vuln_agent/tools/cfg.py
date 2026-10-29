"""cfg: intra-function control-flow graph, true dominance, and def-use taint
reachability (deterministic, no LLM, zero token).

Why this module. In the PrimeVul paired setting, the only discriminative signal is
whether a guard *dominates* the sink, so `obligation_checker.verify_mitigation`
accepts only the true dominance computed here. A line-order comparison is not
control-flow evidence; when the CFG is unavailable the corresponding mitigation
claims stay UNDERDETERMINED.

The statement-level CFG is built directly from the tree-sitter syntax tree, which
turns three relations into computed facts:
  * Dominance -> an iterative fixed-point dominator algorithm; `dominating_guards`
                 returns the guards that truly dominate the sink together with their
                 must-take side (true/false), i.e. whether the check introduced by
                 the patch actually constrains the sink;
  * Path existence -> `path_exists` decides whether a branch-decision sequence
                 submitted by the LLM is really walkable on the CFG (witness check
                 step 1; hallucinated paths are rejected here);
  * Taint reachability -> `taint_reachable` walks assignment dependencies back to
                 the entry parameters (witness check step 4).

Honest boundaries. Without a compilation setting this is a *syntax-level*
approximation; the three known incompletions all degrade conservatively:
  1. Macros and conditional compilation are not expanded, so guards inside macro
     bodies are invisible. This may undercount dominating guards, i.e. bias towards
     reporting, so `dominating_guards` is only used to *veto* a positive verdict and
     never to prove safety.
  2. `goto` edges are built by label; `setjmp`, function pointers and exceptions are
     not modelled.
  3. def-use is a *flow-insensitive* assignment-dependency closure (no alias
     analysis, no path sensitivity), so `taint_reachable` is an over-approximation:
     it serves only as a necessary-condition filter for a positive verdict and is not
     a taint proof.
When no CFG can be built (parse failure / missing function body) an empty graph is
returned and callers read it conservatively as UNDERDETERMINED (I3).
"""
from __future__ import annotations

import re
from typing import Optional

from tree_sitter_languages import get_parser

_IDENT = re.compile(r"[A-Za-z_]\w*")
_KW = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "return",
    "goto", "break", "continue", "sizeof", "int", "char", "void", "const",
    "unsigned", "signed", "long", "short", "float", "double", "struct", "union",
    "enum", "static", "extern", "register", "volatile", "typedef", "NULL",
}

# Statement types the builder cares about; everything else is sequential.
_COND_STMT = {"if_statement", "while_statement", "for_statement", "do_statement"}
_JUMP_STMT = {"return_statement", "break_statement", "continue_statement",
              "goto_statement"}
# Conditional compilation: the syntax tree gives #if/#ifdef a node that wraps the
# guarded statements, with mutually exclusive branches in its `alternative` chain.
_PREPROC_COND = {"preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else",
                 "preproc_elifdef"}


class Node:
    """A CFG node = one statement (or one branch condition).

    id      stable identifier such as "n12"; the LLM cites it when submitting a
            path, which removes the need to copy source text verbatim
    line    1-based absolute line number, in the same coordinate system as
            number_lines, so it points back into the original source
    kind    "cond" (branch condition with true/false out-edges) | "stmt" | "entry"
            | "exit"
    text    the statement text, whitespace-collapsed onto one line; for human
            reading and for the guard-shape predicate only
    """

    __slots__ = ("id", "line", "kind", "text", "succ", "cond_true", "cond_false")

    def __init__(self, nid: str, line: int, kind: str, text: str):
        self.id = nid
        self.line = line
        self.kind = kind
        self.text = text
        self.succ: list[str] = []        # sequential successor (stmt)
        self.cond_true: Optional[str] = None   # then branch when kind=="cond"
        self.cond_false: Optional[str] = None  # else/fallthrough when kind=="cond"

    def successors(self) -> list[str]:
        if self.kind == "cond":
            return [x for x in (self.cond_true, self.cond_false) if x]
        return list(self.succ)

    def as_dict(self) -> dict:
        return {"id": self.id, "line": self.line, "kind": self.kind,
                "text": self.text}


class CFG:
    """Statement-level control-flow graph of one function.

    Node ids are stable (n0, n1, ... in build order) so the LLM can cite them.
    """

    def __init__(self, func_name: str, line_start: int, line_end: int):
        self.func_name = func_name
        self.line_start = line_start
        self.line_end = line_end
        self.nodes: dict[str, Node] = {}
        self.entry = ""
        self.exit = ""
        self.params: list[str] = []       # entry parameter names (taint sources)
        self.degraded: list[str] = []     # incompleteness notes from the build
                                            # (macros / goto / unknown constructs)
        # The graph is immutable once built, so the three O(n^2) derived quantities
        # are computed once on demand and cached: protection_exists calls dominates
        # pairwise over candidate lines, which would otherwise recompute the whole
        # dominator set on every call.
        self._dom_cache: Optional[dict[str, set[str]]] = None
        self._defs_cache: dict[bool, dict[str, set[str]]] = {}
        self._ctrl_cache: Optional[dict[str, set[str]]] = None

    # ---- basic queries ----

    def node(self, nid: str) -> Optional[Node]:
        return self.nodes.get(nid)

    def nodes_at_line(self, line: int) -> list[Node]:
        return [n for n in self.nodes.values() if n.line == line]

    def reachable(self, start: str, *, avoid: str = "") -> set[str]:
        """Nodes reachable from `start`.

        The node named by `avoid` is treated as deleted, which is how dominance is
        read off a reachability query.
        """
        seen: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen or cur == avoid or cur not in self.nodes:
                continue
            seen.add(cur)
            stack.extend(self.nodes[cur].successors())
        return seen

    def preds(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, n in self.nodes.items():
            for s in n.successors():
                if s in out:
                    out[s].append(nid)
        return out

    # ---- true dominance (iterative fixed point) ----

    def dominators(self) -> dict[str, set[str]]:
        """Standard iterative algorithm:

        dom(entry) = {entry};  dom(n) = {n} | intersection over p in preds(n).

        Unreachable nodes keep the full node set as their dominator set, which is
        the conservative choice: no dominance conclusion is claimed for them.
        """
        if not self.entry:
            return {}
        if self._dom_cache is not None:
            return self._dom_cache
        alln = set(self.nodes)
        preds = self.preds()
        dom: dict[str, set[str]] = {n: set(alln) for n in alln}
        dom[self.entry] = {self.entry}
        changed = True
        while changed:
            changed = False
            for n in alln:
                if n == self.entry:
                    continue
                ps = [p for p in preds[n] if p in dom]
                if not ps:
                    continue
                new = set(alln)
                for p in ps:
                    new &= dom[p]
                new.add(n)
                if new != dom[n]:
                    dom[n] = new
                    changed = True
        self._dom_cache = dom
        return dom

    def dominates(self, a: str, b: str) -> bool:
        """Whether a dominates b, i.e. every path from entry to b passes a."""
        return a in self.dominators().get(b, set())

    def branch_taken(self, cond_id: str, target: str) -> Optional[bool]:
        """Which side must be taken to reach `target` from branch node `cond_id`?

        Returns True (then is mandatory), False (else is mandatory), or None (both
        sides reach it, so the branch constrains nothing). This is the deterministic
        criterion for "does the patched if actually constrain the sink": if the
        target is reachable only from the false side, then the sink is unreachable
        when the condition holds, so the guard holds.
        """
        n = self.nodes.get(cond_id)
        if n is None or n.kind != "cond":
            return None
        via_t = target in self.reachable(n.cond_true) if n.cond_true else False
        via_f = target in self.reachable(n.cond_false) if n.cond_false else False
        if via_t and not via_f:
            return True
        if via_f and not via_t:
            return False
        return None

    def dominating_guards(self, sink_line: int) -> list[dict]:
        """Branch guards that truly dominate the sink line (machine facts, not
        LLM assertions).

        Each entry is {id, line, text, must_take}, where must_take True/False marks
        the side that must be taken to reach the sink, and None marks a branch that
        does not constrain the sink (it dominates, but both sides get there, e.g. a
        branch before the join that follows an if/else). Callers should only accept
        entries whose must_take is not None as guards.
        """
        targets = [n.id for n in self.nodes_at_line(sink_line)]
        if not targets:
            return []
        dom = self.dominators()
        out: list[dict] = []
        seen: set[str] = set()
        for t in targets:
            for d in sorted(dom.get(t, set()), key=lambda x: self.nodes[x].line
                            if x in self.nodes else 0):
                nd = self.nodes.get(d)
                if nd is None or nd.kind != "cond" or d in seen:
                    continue
                seen.add(d)
                out.append({
                    "id": nd.id, "line": nd.line, "text": nd.text,
                    "must_take": self.branch_taken(d, t),
                })
        return out

    def dep_closure(self, seeds: set[str], *, control: bool = True) -> set[str]:
        """Undirected connected closure of `seeds` in the dependency graph
        (traversing lhs <-> rhs both ways).

        The purpose is *relatedness*, not derivability: deciding whether a statement
        talks about the same data. The direction is undirected because a guard may
        act on the source variable (`if (n > cap)` while the sink uses
        `len = n - k`) or on the derived quantity (`if (len > cap)` while the sink
        uses `n`). It is an over-approximation, used only to *widen the view*.
        """
        deps = self.defs(control=control)
        rev: dict[str, set[str]] = {}
        for lhs, rhs in deps.items():
            for r in rhs:
                rev.setdefault(r, set()).add(lhs)
        seen: set[str] = set()
        stack = [s for s in seeds if s]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(x for x in deps.get(cur, set()) | rev.get(cur, set())
                         if x not in seen)
        return seen

    def dominating_statements(self, sink_line: int, max_nodes: int = 8) -> list[Node]:
        """Non-branch statements that truly dominate the sink line and are
        data-related to it (the nearest few by line number).

        Motivation: an existing guard is not always a branch. A validation function
        that returns a status code checked by the caller, a wrapper that validates at
        construction time, and a length-capping API are all `stmt` nodes in the CFG
        and are therefore invisible to `dominating_guards`. The risk window spans
        only +/- config.RISK_WINDOW lines around the sink, so a validation outside
        that window cannot be presented to the model at all. This method brings that
        deterministic structure into view at zero marginal token cost (a few extra
        lines in the same prompt). The criterion is "truly dominates + connected to
        the sink variables in the dependency graph", and it involves no table of
        validation function names, so it works equally on unfamiliar projects and
        private APIs.

        Relatedness uses `dep_closure` (including control dependencies) rather than
        literal shared identifiers: a macro-shaped validation
        (`OP_REQUIRES(ctx, !overflow, ...)`) and the sink often share no identifier
        at all, so a purely lexical intersection cannot see it, while it is exactly
        the guard in question.
        """
        targets = [n.id for n in self.nodes_at_line(sink_line)]
        if not targets or not self.entry:
            return []
        want = set()
        for t in targets:
            want |= idents_of(self.nodes[t].text)
        if not want:
            return []
        want = self.dep_closure(want)
        dom = self.dominators()
        keep: dict[str, Node] = {}
        for t in targets:
            for d in dom.get(t, set()):
                nd = self.nodes.get(d)
                if (nd is None or nd.kind != "stmt" or nd.line == sink_line
                        or d in keep or not (idents_of(nd.text) & want)):
                    continue
                keep[d] = nd
        return sorted(keep.values(), key=lambda x: (x.line, x.id))[-max_nodes:]

    def path_exists(self, decisions: list[dict]) -> tuple[bool, str]:
        """Witness check step 1: is a branch-decision sequence really walkable on
        the CFG?

        decisions = [{"node": "n7", "take": true}, ...]; {"line": 84, ...} is also
        accepted. Each step asks whether the next decision node is reachable from the
        current position and then advances along `take`; any unwalkable step returns
        (False, reason). (True, "") means the control-flow path actually exists.
        """
        if not self.entry:
            return False, "CFG is empty (parse failure or missing function body)"
        cur = self.entry
        for i, d in enumerate(decisions or []):
            nid = str(d.get("node") or "")
            if nid not in self.nodes:
                line = int(d.get("line") or 0)
                cands = [n.id for n in self.nodes_at_line(line)
                         if self.nodes[n.id].kind == "cond"]
                if not cands:
                    return False, (f"decision #{i} points to a nonexistent node "
                                   f"(node={nid} line={line})")
                nid = cands[0]
            if nid not in self.reachable(cur):
                return False, (f"decision #{i} ({nid}) is unreachable from "
                               f"{cur}: no such path")
            n = self.nodes[nid]
            if n.kind != "cond":
                return False, f"decision #{i} ({nid}) is not a branch node"
            take = bool(d.get("take", d.get("branch", True)))
            nxt = n.cond_true if take else n.cond_false
            if not nxt:
                return False, (f"decision #{i} ({nid}) has no "
                               f"{'then' if take else 'else'} branch")
            cur = nxt
        return True, ""

    # ---- def-use (flow-insensitive assignment-dependency closure; an
    # over-approximation used only as a necessary-condition filter) ----

    def _controlling_idents(self) -> dict[str, set[str]]:
        """{node id: identifiers of the branch conditions that dominate it and
        from which it is reachable through exactly one side}.

        This is syntactic *control dependence*: in
        `if (size > limit) overflow = true;` the value of `overflow` depends on
        size/limit, yet the right-hand side contains only the literal `true`, so a
        pure assignment closure cannot see the dependency. The criterion is "the
        branch dominates the node AND the node is reachable through exactly one
        side", with no name table involved.
        """
        if self._ctrl_cache is not None:
            return self._ctrl_cache
        out: dict[str, set[str]] = {}
        dom = self.dominators()
        for c in list(self.nodes.values()):
            if c.kind != "cond":
                continue
            ids = idents_of(c.text)
            if not ids:
                continue
            rt = self.reachable(c.cond_true) if c.cond_true else set()
            rf = self.reachable(c.cond_false) if c.cond_false else set()
            for nid in rt ^ rf:      # reachable via one side only => controlled
                if c.id in dom.get(nid, set()):
                    out.setdefault(nid, set()).update(ids)
        self._ctrl_cache = out
        return out

    def defs(self, control: bool = False) -> dict[str, set[str]]:
        """{assigned variable: set of identifiers appearing on the right-hand side
        of that assignment}, unioned over the whole function (flow-insensitive).

        No alias analysis: `*p = x` records p only and does not follow what p points
        to.

        With `control=True`, control dependencies are folded into the right-hand-side
        sets (see `_controlling_idents`). A flag-style validation
        (`if (n > cap) bad = 1;` followed by `if (bad) return;`) is disconnected from
        the quantity it validates in a pure assignment closure, so connectivity would
        miss it. The extension only makes the dependency closure *larger* (easier to
        accept a derivation), so it serves a *one-sided* purpose: voiding a preset
        assumption (-> UNDERDETERMINED) and placing a possibly existing guard in the
        model's view. The strict guard re-check that writes SATISFIED keeps using the
        narrow closure with control=False.
        """
        if control in self._defs_cache:
            return self._defs_cache[control]
        ctrl = self._controlling_idents() if control else {}
        out: dict[str, set[str]] = {}
        for n in self.nodes.values():
            extra = ctrl.get(n.id, set()) if control else set()
            for lhs, rhs in _assignments(n.text):
                cur = out.setdefault(lhs, set())
                cur.update(rhs)
                cur.update(x for x in extra if x != lhs)
        self._defs_cache[control] = out
        return out

    def taint_reachable(self, var: str) -> bool:
        """Whether `var` is reachable from an entry parameter along an assignment
        dependency chain (a necessary condition of witness check step 4).

        *Over-approximation*: it is flow-insensitive, performs no alias analysis and
        is path-insensitive, so it can only be used to *refute* a witness of the form
        "the variable in the premise is actually a local constant"; it is not a taint
        proof. When the parameter list is empty (degraded build) it returns True
        unconditionally, which keeps the analysis from suppressing recall.
        """
        if not self.params:
            return True
        if var in self.params:
            return True
        deps = self.defs()
        seen: set[str] = set()
        stack = [var]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in self.params:
                return True
            for nxt in deps.get(cur, set()):
                if nxt not in seen:
                    stack.append(nxt)
        return False

    def brief(self, sink_line: int = 0, max_nodes: int = 24) -> str:
        """Compact branch-structure string fed to the LLM:
        `n3@5 if(n > 64) ->T:n4 F:n6`.

        The point is to let the model cite *node ids* instead of copying source text
        verbatim. Only branch nodes are listed (a sink verdict only cares about
        branches), sorted by line, with the ones that truly dominate the sink line
        marked. A trailing section lists the non-branch statements that truly
        dominate the sink and share identifiers with it (`dominating_statements`):
        existing guards often take the shape of a validation call that returns a
        status code, or of a wrapper that validates at construction time, and often
        fall outside the risk window; without them in view the model can only report
        a missing guard on the ground that the code does not look like its ideal
        form. That section is deterministic structure, not an LLM paraphrase.
        """
        if not self.entry:
            return "(control-flow graph unavailable)"
        doms = {g["id"] for g in self.dominating_guards(sink_line)
                if g.get("must_take") is not None} if sink_line else set()
        conds = sorted((n for n in self.nodes.values() if n.kind == "cond"),
                       key=lambda x: (x.line, x.id))[:max_nodes]
        parts = []
        for n in conds:
            mark = " *dominates-sink" if n.id in doms else ""
            parts.append(f"{n.id}@{n.line} {n.text} →T:{n.cond_true or '-'} "
                         f"F:{n.cond_false or '-'}{mark}")
        head = "; ".join(parts) if parts else "(no branches in this function)"
        if len(conds) >= max_nodes:
            head += " ...(truncated)"
        pre = self.dominating_statements(sink_line) if sink_line else []
        if pre:
            head += ("\n    statements that must precede the sink and involve the "
                     "same variables (an existing guard may be here): \""
                     + "\" \"".join(f"{n.line}: {n.text}" for n in pre) + "\"")
        return head

    def summary(self) -> dict:
        """Compact graph summary fed to the LLM (node ids + line numbers + branch
        structure), so the model need not copy source text verbatim."""
        return {
            "func": self.func_name,
            "entry": self.entry,
            "params": list(self.params),
            "nodes": [
                {**n.as_dict(),
                 **({"true": n.cond_true, "false": n.cond_false}
                    if n.kind == "cond" else {"next": n.succ})}
                for n in sorted(self.nodes.values(), key=lambda x: (x.line, x.id))
            ],
            "degraded": list(self.degraded),
        }


# ---- assignment extraction (for def-use; syntactic regex, sufficient by
# construction) ----

_ASSIGN = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:\+|-|\*|/|\||&|\^|<<|>>)?=(?!=)([^;]*)")


def _assignments(text: str) -> list[tuple[str, set[str]]]:
    """Extract (assigned variable, set of right-hand-side identifiers) from one line
    or clause. Declarations such as `int n = f(x)` match as well."""
    out: list[tuple[str, set[str]]] = []
    for m in _ASSIGN.finditer(text or ""):
        lhs = m.group(1)
        if lhs in _KW:
            continue
        rhs = {t for t in _IDENT.findall(m.group(2) or "") if t not in _KW}
        out.append((lhs, rhs))
    return out


def idents_of(text: str) -> set[str]:
    """Extract identifiers (C keywords removed) for the premise variables of a
    witness check."""
    return {t for t in _IDENT.findall(text or "") if t not in _KW}


# ---- construction ----

class _Builder:
    """Linearise a tree-sitter syntax tree into a statement-level CFG.

    Strategy: a single recursion that maintains the set of current tail nodes
    (possibly several, e.g. at an if/else join) together with the pending edges of
    break/continue/goto. Compiler-grade precision is not the goal: as soon as the
    branch structure and must-take relations are correct, dominance and path
    existence are well defined.
    """

    def __init__(self, src: bytes, cfg: CFG):
        self.src = src
        self.cfg = cfg
        self.i = 0
        self.pending_break: list[str] = []
        self.pending_continue: list[str] = []
        self.goto_edges: list[tuple[str, str]] = []   # (from_node, label)
        self.labels: dict[str, str] = {}              # label -> node id

    def new(self, node, kind: str) -> Node:
        nid = f"n{self.i}"
        self.i += 1
        text = self.src[node.start_byte:node.end_byte].decode("utf-8", "replace")
        text = " ".join(text.split())
        if len(text) > 160:
            text = text[:157] + "…"
        n = Node(nid, node.start_point[0] + 1, kind, text)
        self.cfg.nodes[nid] = n
        return n

    def synth(self, line: int, kind: str, text: str) -> Node:
        nid = f"n{self.i}"
        self.i += 1
        n = Node(nid, line, kind, text)
        self.cfg.nodes[nid] = n
        return n

    @staticmethod
    def link(tails: list[str], target: str, cfg: CFG) -> None:
        for t in tails:
            n = cfg.nodes.get(t)
            if n is None or n.kind == "cond":
                continue
            if target not in n.succ:
                n.succ.append(target)

    # Each visit_* returns (first node id or "", list of tail nodes).
    # "" means the construct produced no node at all (e.g. an empty block), and
    # the caller simply passes its tails through.

    def visit_stmts(self, nodes) -> tuple[str, list[str]]:
        first = ""
        tails: list[str] = []
        for st in nodes:
            if not st.is_named or st.type == "comment":
                continue
            f, t = self.visit(st)
            if not f:
                continue
            if not first:
                first = f
            else:
                self.link(tails, f, self.cfg)
            tails = t
        return first, tails

    def visit(self, node) -> tuple[str, list[str]]:
        t = node.type
        if t == "compound_statement":
            return self.visit_stmts(node.children)
        if t == "if_statement":
            return self.visit_if(node)
        if t in ("while_statement", "for_statement"):
            return self.visit_loop(node)
        if t == "do_statement":
            return self.visit_do(node)
        if t == "switch_statement":
            return self.visit_switch(node)
        if t in _PREPROC_COND:
            return self.visit_preproc(node)
        if t == "labeled_statement":
            return self.visit_labeled(node)
        if t in _JUMP_STMT:
            n = self.new(node, "stmt")
            if t == "return_statement":
                self.link([n.id], self.cfg.exit, self.cfg)
                return n.id, []          # no sequential successor after return
            if t == "break_statement":
                self.pending_break.append(n.id)
                return n.id, []
            if t == "continue_statement":
                self.pending_continue.append(n.id)
                return n.id, []
            # goto: build an edge by label (the label may appear later, so the
            # edge is back-patched)
            lbl = ""
            for c in node.children:
                if c.type in ("statement_identifier", "identifier"):
                    lbl = self.src[c.start_byte:c.end_byte].decode("utf-8", "replace")
            if lbl:
                self.goto_edges.append((n.id, lbl))
            else:
                self.cfg.degraded.append(
                    f"line {n.line}: unresolvable goto target")
            return n.id, []
        # everything else is a sequential statement
        n = self.new(node, "stmt")
        return n.id, [n.id]

    def visit_if(self, node) -> tuple[str, list[str]]:
        cond_node = node.child_by_field_name("condition") or node
        c = self.new(cond_node, "cond")
        c.line = node.start_point[0] + 1          # the if line, for source alignment
        cons = node.child_by_field_name("consequence")
        alt = node.child_by_field_name("alternative")
        join = self.synth(node.end_point[0] + 1, "stmt", "(join)")
        if cons is not None:
            f, tails = self.visit(cons)
            c.cond_true = f or join.id
            self.link(tails, join.id, self.cfg)
        else:
            c.cond_true = join.id
        if alt is not None:
            # tree-sitter's alternative may be wrapped in an else_clause
            target = alt
            if alt.type == "else_clause":
                kids = [k for k in alt.children if k.is_named]
                target = kids[0] if kids else None
            if target is not None:
                f, tails = self.visit(target)
                c.cond_false = f or join.id
                self.link(tails, join.id, self.cfg)
            else:
                c.cond_false = join.id
        else:
            c.cond_false = join.id
        return c.id, [join.id]

    def visit_loop(self, node) -> tuple[str, list[str]]:
        """while / for: condition node -> then=body, false=exit. The back edge
        goes to the condition node.

        A for statement's initializer and increment are chained around it as ordinary
        statements (a syntactic approximation, adequate for dominance decisions).
        """
        saved_b, saved_c = self.pending_break, self.pending_continue
        self.pending_break, self.pending_continue = [], []
        head = ""
        tails: list[str] = []
        if node.type == "for_statement":
            init = node.child_by_field_name("initializer")
            if init is not None and init.is_named:
                n = self.new(init, "stmt")
                head, tails = n.id, [n.id]
        cond_src = (node.child_by_field_name("condition")
                    or node.child_by_field_name("condition_clause") or node)
        c = self.new(cond_src, "cond")
        c.line = node.start_point[0] + 1
        if head:
            self.link(tails, c.id, self.cfg)
        else:
            head = c.id
        body = node.child_by_field_name("body")
        exit_join = self.synth(node.end_point[0] + 1, "stmt", "(loop-exit)")
        if body is not None:
            f, btails = self.visit(body)
            c.cond_true = f or c.id
            self.link(btails, c.id, self.cfg)     # back edge
        else:
            c.cond_true = c.id
        c.cond_false = exit_join.id
        for b in self.pending_break:
            self.link([b], exit_join.id, self.cfg)
        for k in self.pending_continue:
            self.link([k], c.id, self.cfg)
        self.pending_break, self.pending_continue = saved_b, saved_c
        return head, [exit_join.id]

    def visit_do(self, node) -> tuple[str, list[str]]:
        saved_b, saved_c = self.pending_break, self.pending_continue
        self.pending_break, self.pending_continue = [], []
        body = node.child_by_field_name("body")
        cond_src = node.child_by_field_name("condition") or node
        exit_join = self.synth(node.end_point[0] + 1, "stmt", "(do-exit)")
        head = ""
        btails: list[str] = []
        if body is not None:
            head, btails = self.visit(body)
        c = self.new(cond_src, "cond")
        c.line = cond_src.start_point[0] + 1
        if not head:
            head = c.id
        else:
            self.link(btails, c.id, self.cfg)
        c.cond_true = head          # back edge (do-while runs the body first)
        c.cond_false = exit_join.id
        for b in self.pending_break:
            self.link([b], exit_join.id, self.cfg)
        for k in self.pending_continue:
            self.link([k], c.id, self.cfg)
        self.pending_break, self.pending_continue = saved_b, saved_c
        return head, [exit_join.id]

    def visit_switch(self, node) -> tuple[str, list[str]]:
        """A switch becomes "one multi-exit node + one body per case".

        Conservatively the switch head is built as a `stmt` (not a `cond`): it has N
        successors, so the binary branch criterion does not apply; case fallthrough
        is wired in source order.

        A `case_statement` **must** be expanded statement by statement instead of
        being folded into a single node by the generic visit: folding would collapse
        the whole case body into one `stmt` whose line is the `case` label line, so
        the sink lines inside the body would not exist in the graph at all, witness
        step 1 ("sink line absent from the CFG") would reject the whole sample class,
        and the `if` inside the body would contribute no dominating guard.
        """
        head = self.new(node.child_by_field_name("condition") or node, "stmt")
        head.line = node.start_point[0] + 1
        exit_join = self.synth(node.end_point[0] + 1, "stmt", "(switch-exit)")
        saved_b = self.pending_break
        self.pending_break = []
        body = node.child_by_field_name("body")
        prev_tails: list[str] = []
        if body is not None:
            for st in body.children:
                if not st.is_named or st.type == "comment":
                    continue
                if st.type == "case_statement":
                    # the case label itself produces no node; its body is built
                    # statement by statement (an empty case simply falls through)
                    f, tails = self.visit_stmts(
                        [k for k in st.children
                         if k.is_named and k.type != "comment"
                         and k != st.child_by_field_name("value")])
                else:
                    f, tails = self.visit(st)
                if not f:
                    continue
                self.link([head.id], f, self.cfg)
                self.link(prev_tails, f, self.cfg)   # case fallthrough
                prev_tails = tails
        self.link(prev_tails, exit_join.id, self.cfg)
        for b in self.pending_break:
            self.link([b], exit_join.id, self.cfg)
        self.pending_break = saved_b
        self.cfg.degraded.append(
            f"line {head.line}: switch approximated as multi-exit "
            f"(no binary branch criterion)")
        return head.id, [exit_join.id]

    def visit_preproc(self, node) -> tuple[str, list[str]]:
        """Conditional compilation (#if/#ifdef/#elif/#else) becomes "one multi-exit
        node + one body per branch".

        Folding the whole region into a single `stmt` node (line = the #if line)
        would leave the statements inside the branches absent from the graph, making
        a sink inside an #ifdef invisible.

        Modelling it as mutually exclusive multi-branches is a conservative
        over-approximation: only one branch is ever compiled, so statements in any
        branch body may be present. Conversely the head node is built as a `stmt`
        (not a `cond`) and an extra bypass edge is always added to the join, so a
        guard inside a branch body is never counted as a dominating guard: an
        analysis limitation can only under-prove safety, never manufacture a
        `safe_proved`. Macro bodies themselves remain unexpanded (see the degraded
        notes emitted by build).
        """
        head = self.synth(node.start_point[0] + 1, "stmt",
                          f"(preproc {node.type})")
        join = self.synth(node.end_point[0] + 1, "stmt", "(preproc-join)")
        cur = node
        while cur is not None:
            alt = cur.child_by_field_name("alternative")
            # Exclude the condition/macro name and the next branch (the
            # `alternative`). Note that nodes cannot be compared with id():
            # tree_sitter creates a fresh wrapper object on every
            # child_by_field_name call, so they are compared by byte range.
            excl = {(x.start_byte, x.end_byte)
                    for x in (alt, cur.child_by_field_name("condition"),
                              cur.child_by_field_name("name")) if x is not None}
            body = [k for k in cur.children
                    if k.is_named and k.type != "comment"
                    and (k.start_byte, k.end_byte) not in excl
                    and k.type not in _PREPROC_COND]
            f, tails = self.visit_stmts(body)
            if f:
                self.link([head.id], f, self.cfg)
                self.link(tails, join.id, self.cfg)
            cur = alt
        # A bypass edge is always kept: which branch is compiled is unknowable
        # without preprocessing, so no branch body may be counted as mandatory on
        # the way to the sink. This is conservative in one direction only: it can
        # under-prove safety.
        self.link([head.id], join.id, self.cfg)
        self.cfg.degraded.append(
            f"line {head.line}: conditional compilation approximated as mutually "
            f"exclusive multi-branches (guards inside branches do not count as "
            f"dominating guards)")
        return head.id, [join.id]

    def visit_labeled(self, node) -> tuple[str, list[str]]:
        lbl = ""
        inner = None
        for c in node.children:
            if c.type in ("statement_identifier", "identifier") and not lbl:
                lbl = self.src[c.start_byte:c.end_byte].decode("utf-8", "replace")
            elif c.is_named:
                inner = c
        if inner is None:
            n = self.new(node, "stmt")
            if lbl:
                self.labels[lbl] = n.id
            return n.id, [n.id]
        f, tails = self.visit(inner)
        if lbl and f:
            self.labels[lbl] = f
        return f, tails


def _params_of(fdef, src: bytes) -> list[str]:
    """Extract formal parameter names (taint sources): the innermost identifier of
    every parameter_declaration inside the parameter_list."""
    out: list[str] = []

    def find_plist(node):
        if node.type == "parameter_list":
            return node
        for c in node.children:
            r = find_plist(c)
            if r is not None:
                return r
        return None

    decl = fdef.child_by_field_name("declarator")
    plist = find_plist(decl) if decl is not None else None
    if plist is None:
        return out
    for p in plist.children:
        if p.type != "parameter_declaration":
            continue
        last = ""
        stack = [p]
        while stack:
            cur = stack.pop()
            if cur.type == "identifier":
                last = src[cur.start_byte:cur.end_byte].decode("utf-8", "replace")
            stack.extend(cur.children)
        if last and last not in _KW:
            out.append(last)
    return out


def _find_func(root, src: bytes, func_name: str, line_start: int):
    """Locate the target function definition in the syntax tree: by start line
    first, falling back to the function name."""
    best = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            if node.start_point[0] + 1 == line_start:
                return node
            if best is None and func_name:
                txt = src[node.start_byte:node.end_byte].decode("utf-8", "replace")
                head = txt.split("{", 1)[0]
                if re.search(rf"\b{re.escape(func_name)}\s*\(", head):
                    best = node
        stack.extend(node.children)
    return best


def build(source: str, func_name: str, line_start: int, line_end: int,
          lang: str = "c") -> CFG:
    """Build the CFG of one function. On parse failure or a missing function body an
    empty graph is returned (with the reason recorded in `degraded`); callers read it
    conservatively as UNDERDETERMINED (I3), and a verdict is never forced merely
    because the analysis ran out of power.

    `source` must be the *entire* file text without line-number prefixes; the line
    coordinate system matches number_lines.
    """
    cfg = CFG(func_name, line_start, line_end)
    try:
        parser = get_parser("cpp" if lang == "cpp" else "c")
        src = source.encode("utf-8")
        tree = parser.parse(src)
    except Exception as e:                       # parser unavailable / bad source
        cfg.degraded.append(f"parse failed: {type(e).__name__}")
        return cfg
    fdef = _find_func(tree.root_node, src, func_name, line_start)
    if fdef is None:
        cfg.degraded.append(f"function not located: {func_name}@{line_start}")
        return cfg
    body = fdef.child_by_field_name("body")
    if body is None:
        cfg.degraded.append("function has no body (a declaration?)")
        return cfg
    cfg.params = _params_of(fdef, src)
    b = _Builder(src, cfg)
    entry = b.synth(fdef.start_point[0] + 1, "entry", f"(entry {func_name})")
    exit_n = b.synth(fdef.end_point[0] + 1, "exit", "(exit)")
    cfg.entry, cfg.exit = entry.id, exit_n.id
    first, tails = b.visit_stmts(body.children)
    _Builder.link([entry.id], first or exit_n.id, cfg)
    _Builder.link(tails, exit_n.id, cfg)
    # goto back-patching
    for src_id, lbl in b.goto_edges:
        target = b.labels.get(lbl)
        if target:
            n = cfg.nodes.get(src_id)
            if n is not None and target not in n.succ:
                n.succ.append(target)
        else:
            cfg.degraded.append(f"goto {lbl}: label not found (edge missing)")
    if "#" in source:
        cfg.degraded.append(
            "source contains preprocessor directives: guards inside macro bodies "
            "are invisible (dominating guards may be undercounted)")
    return cfg


def build_for_line(source: str, functions: list, line: int,
                   lang: str = "c") -> Optional[CFG]:
    """Convenience entry: locate the enclosing function by line number and build
    its graph (`functions` is state["functions"])."""
    for f in functions or []:
        if f.get("line_start", 0) <= line <= f.get("line_end", 0):
            return build(source, f.get("func_name", ""), int(f["line_start"]),
                         int(f["line_end"]), f.get("lang", lang))
    return None
