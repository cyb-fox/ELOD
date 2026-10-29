"""sink_locator: rule-path sink localisation + an operation summary.

- SENSITIVE_APIS: the sensitive-API table (grouped by sink_type), a
  family-independent recall floor.
- rule_locate_sinks: tree-sitter locates call_expressions and matches them against
  the sensitive APIs; it then overlays rule_locate_buffer_walks (a syntax-level floor
  for **unbounded pointer walks**) and rule_locate_dereferences.
- build_op_summary: the "function name + sensitive calls" summary consumed by
  select_knowledge; the full code is never fed.
- extract_tainted_tokens: identifiers drawn from call actuals, used as forensics
  tracing seeds.

Deliberately permissive and tolerant of false positives -- precision is delegated to
perspective B and the arbiter.

Why rule_locate_buffer_walks exists: the sensitive-API table only recognises libc
names, so project-private APIs never match and the rule floor would be identically
zero; every sink in such a file could then only come from the LLM secure_diff
channel, and `sinks_examined=0` on that channel turns the report silently into "no
vulnerability". A pointer that keeps advancing inside a loop while the loop bound
has nothing to do with the readable extent of that buffer is a **purely syntactic**
suspicious shape -- it does not require knowing how large the buffer is; the very
absence of a bound is what makes it suspicious. Hence it is made a deterministic
floor: it produces sinks (i.e. raises suspicions) and no verdicts.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Optional

from tree_sitter_languages import get_parser

from ..state import Func, Sink
from . import obligations as ob

warnings.filterwarnings("ignore", category=FutureWarning)

# sink_type -> set of sensitive API names
SENSITIVE_APIS: dict[str, set[str]] = {
    "mem": {
        "memcpy", "memmove", "memset", "strcpy", "strncpy", "strcat",
        "strncat", "sprintf", "vsprintf", "gets", "alloca", "bcopy",
    },
    "alloc": {"malloc", "calloc", "realloc", "alloca"},
    "exec": {
        "system", "popen", "exec", "execl", "execlp", "execle", "execv",
        "execvp", "execve", "fork", "posix_spawn",
        "luaL_loadstring", "luaL_dostring", "lua_pcall", "luaL_loadbuffer",
    },
    "file_op": {
        "open", "fopen", "openat", "fread", "read", "unlink", "remove",
        "rename", "opendir", "access", "stat", "readlink",
    },
    "format": {"printf", "fprintf", "snprintf", "vsnprintf", "syslog", "scanf", "sscanf"},
    "sql": {"mysql_query", "sqlite3_exec", "PQexec", "execute"},
}

# Reverse lookup: api -> sink_type
_API_TYPE: dict[str, str] = {}
for _t, _apis in SENSITIVE_APIS.items():
    for _a in _apis:
        _API_TYPE.setdefault(_a, _t)


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _parser_for(lang: str):
    return get_parser("cpp" if lang == "cpp" else "c")


def _iter_calls(source: str, func: Func):
    """yield (call_node, callee_name, arg_text, line) for calls inside func."""
    parser = _parser_for(func["lang"])
    src = source.encode("utf-8")
    tree = parser.parse(src)
    s, e = func["line_start"], func["line_end"]

    results = []

    def visit(node):
        if node.type == "call_expression":
            line = node.start_point[0] + 1
            if s <= line <= e:
                fn = node.child_by_field_name("function")
                name = ""
                if fn is not None:
                    txt = src[fn.start_byte:fn.end_byte].decode("utf-8", "replace")
                    # Take the last identifier segment (drops namespace/member
                    # prefixes).
                    m = _IDENT.findall(txt)
                    name = m[-1] if m else txt
                args = node.child_by_field_name("arguments")
                arg_text = (
                    src[args.start_byte:args.end_byte].decode("utf-8", "replace")
                    if args is not None else ""
                )
                results.append((node, name, arg_text, line))
        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return results


def extract_tainted_tokens(arg_text: str) -> list[str]:
    """Extract identifiers from an actual-argument text, filtering out common
    types/keywords."""
    stop = {
        "sizeof", "int", "char", "void", "unsigned", "long", "short",
        "const", "static", "struct", "NULL", "nullptr", "true", "false",
        "uint32_t", "uint8_t", "size_t", "return",
    }
    out: list[str] = []
    for tok in _IDENT.findall(arg_text):
        if tok not in stop and tok not in out:
            out.append(tok)
    return out[:6]


def rule_locate_sinks(
    source: str, func: Func, knowledge: Optional[dict[str, Any]] = None,
    *, include_phase1: bool = True,
) -> list[Sink]:
    """The rule path: sensitive APIs, dereferences and unbounded pointer walks
    (a deterministic floor, zero LLM).

    ``include_phase1`` exists for offline safe replay: broad Phase-1 risk hints
    without candidate family context must not become obligations that block
    ``safe_proved``.
    """
    sinks: list[Sink] = []
    seen: set[int] = set()
    lines = source.split("\n")

    for _node, name, arg_text, line in _iter_calls(source, func):
        if name in _API_TYPE and line not in seen:
            seen.add(line)
            snippet = lines[line - 1].strip()
            sinks.append(Sink(
                sink_id=f"{func['func_name']}:{line}",
                func_name=func["func_name"], line=line, snippet=snippet,
                sink_type=_API_TYPE[name],
                tainted_tokens=extract_tainted_tokens(arg_text),
                source="rule",
            ))
    for s in rule_locate_buffer_walks(source, func):
        if s["line"] not in seen:
            seen.add(s["line"])
            sinks.append(s)
    for s in rule_locate_dereferences(source, func):
        if s["line"] not in seen:
            seen.add(s["line"])
            sinks.append(s)
    return merge_sinks(sinks, rule_locate_phase1_patterns(source, func) if include_phase1 else [])


def rule_locate_dereferences(source: str, func: Func) -> list[Sink]:
    """AST-level local dereference candidates for ``p->field`` and ``*p``.

    Candidates only; non-nullness must still be established by the checker's CFG
    dominating-guard rules.
    """
    node = _func_node(source, func)
    src = source.encode("utf-8")
    lines = source.split("\n")
    found: list[Sink] = []
    seen: set[tuple[int, str]] = set()

    def text(n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def visit(n) -> None:
        token = ""
        if n.type == "field_expression":
            op, arg = n.child_by_field_name("operator"), n.child_by_field_name("argument")
            if op is not None and text(op) == "->" and arg is not None:
                ids = _IDENT.findall(text(arg))
                token = ids[-1] if ids else ""
        elif n.type == "pointer_expression":
            op, arg = n.child_by_field_name("operator"), n.child_by_field_name("argument")
            if op is not None and text(op) == "*" and arg is not None:
                ids = _IDENT.findall(text(arg))
                token = ids[-1] if ids else ""
        if token:
            line = n.start_point[0] + 1
            key = (line, token)
            if func["line_start"] <= line <= func["line_end"] and key not in seen:
                seen.add(key)
                found.append(Sink(
                    sink_id=f"{func['func_name']}:{line}:deref:{token}",
                    func_name=func["func_name"], line=line,
                    snippet=lines[line - 1].strip(), sink_type="deref",
                    tainted_tokens=[token], source="rule",
                ))
        for child in n.children:
            visit(child)

    visit(node)
    return found[:12]


# ---- Unbounded pointer walk floor (syntax level) ----
# The shape: `p` traces back to a **pointer parameter of this function** (its
# readable extent is not visible in the file) and keeps advancing inside a loop
# body (`*p++` / `p = Helper(..., p, ...)` / `p += k`), while the function contains
# no relational comparison against `p` at all (no `p < end`-style bound). This does
# not require knowing how large the buffer is -- precisely because the file states
# no bound, the shape must be raised as suspicious. It produces sinks (suspicion
# raising) and writes no obligation status.
_LOOPS = {"for_statement", "while_statement", "do_statement"}
_REL_OPS = {"<", ">", "<=", ">="}


def _decl_pointers(src: bytes, func_node) -> dict[str, str]:
    """Pointer names in the function -> role ("param" formal parameter / "local")."""
    out: dict[str, str] = {}

    def name_of(node):
        for c in node.children:
            if c.type == "identifier":
                return src[c.start_byte:c.end_byte].decode("utf-8", "replace")
            if c.type in ("pointer_declarator", "array_declarator",
                          "function_declarator", "init_declarator"):
                n = name_of(c)
                if n:
                    return n
        return ""

    def visit(node, in_params: bool):
        if node.type == "parameter_list":
            in_params = True
        if node.type in ("parameter_declaration", "declaration"):
            for c in node.children:
                if c.type == "pointer_declarator":
                    n = name_of(c)
                    if n:
                        out.setdefault(n, "param" if in_params else "local")
                elif c.type == "init_declarator":
                    for cc in c.children:
                        if cc.type == "pointer_declarator":
                            n = name_of(cc)
                            if n:
                                out.setdefault(n, "param" if in_params else "local")
        for c in node.children:
            visit(c, in_params)

    visit(func_node, False)
    return out


def _func_node(source: str, func: Func):
    """The function_definition node for this Func (falls back to root if absent)."""
    parser = _parser_for(func["lang"])
    tree = parser.parse(source.encode("utf-8"))
    s, e = func["line_start"], func["line_end"]
    best = None
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            ls, le = n.start_point[0] + 1, n.end_point[0] + 1
            if ls >= s and le <= e:
                if best is None or (le - ls) > (best.end_point[0] - best.start_point[0]):
                    best = n
        stack.extend(n.children)
    return best or tree.root_node


def _bounded_pointers(src: bytes, func_node) -> set[str]:
    """Pointer names that appear in a relational comparison serving as a bounds
    check (`p < end` / `end > p`). A pointer listed here has a file-visible bound
    and is no longer reported as an unbounded walk."""
    out: set[str] = set()

    def visit(node):
        if node.type == "binary_expression":
            op = ""
            for c in node.children:
                if c.type in _REL_OPS:
                    op = c.type
            if op:
                for side in (node.child_by_field_name("left"),
                             node.child_by_field_name("right")):
                    if side is None:
                        continue
                    txt = src[side.start_byte:side.end_byte].decode("utf-8", "replace")
                    ids = _IDENT.findall(txt)
                    if len(ids) == 1:
                        out.add(ids[0])
        for c in node.children:
            visit(c)

    visit(func_node)
    return out


def _advance_sites(src: bytes, node, ptrs: set[str], acc: list) -> None:
    """Collect the advance sites of the pointers in ptrs inside a loop body:
    (line, ptr, kind).

    Three syntactic shapes (none requires knowing the buffer size):
      · `*p++` / `p++` / `++p`      => update_expression
      · `p += k`                     => assignment_expression, operator "+="
      · `p = Helper(..., p, ...)`    => p is handed to a helper function and its
                                        return value is taken back
    """
    if node.type == "update_expression":
        arg = node.child_by_field_name("argument")
        if arg is not None:
            txt = src[arg.start_byte:arg.end_byte].decode("utf-8", "replace")
            ids = _IDENT.findall(txt)
            if len(ids) == 1 and ids[0] in ptrs:
                acc.append((node.start_point[0] + 1, ids[0], "incr"))
    elif node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        lname = ""
        if left is not None:
            lids = _IDENT.findall(
                src[left.start_byte:left.end_byte].decode("utf-8", "replace"))
            lname = lids[0] if len(lids) == 1 else ""
        if lname in ptrs and right is not None:
            op = "".join(c.type for c in node.children if c.type in ("+=", "-="))
            rtxt = src[right.start_byte:right.end_byte].decode("utf-8", "replace")
            if op:
                acc.append((node.start_point[0] + 1, lname, "addassign"))
            elif right.type == "call_expression" and re.search(
                    rf"\b{re.escape(lname)}\b", rtxt):
                acc.append((node.start_point[0] + 1, lname, "helper"))
    for c in node.children:
        _advance_sites(src, c, ptrs, acc)


def rule_locate_buffer_walks(source: str, func: Func) -> list[Sink]:
    """The unbounded pointer walk floor: a **pointer parameter** that advances
    inside a loop body with no bounds comparison for that pointer in the function.

    Suspicion only (sink_type="mem", matching the readable_extent obligation), no
    adjudication whatsoever -- whether an out-of-bounds access really occurs is
    still decided by obligation_checker after it reviews the evidence; when it
    cannot be established, the obligation stays underdetermined.
    """
    fn = _func_node(source, func)
    src = source.encode("utf-8")
    ptrs = _decl_pointers(src, fn)
    params = {n for n, role in ptrs.items() if role == "param"}
    if not params:
        return []
    bounded = _bounded_pointers(src, fn)
    lines = source.split("\n")

    # A local pointer can "inherit" a param's invisible readable extent
    # (`p=pixels;`): fold such aliases into the candidate set.
    alias: dict[str, str] = {}
    for name, role in ptrs.items():
        if role != "local":
            continue
        for m in re.finditer(rf"\b{re.escape(name)}\s*=\s*([A-Za-z_]\w*)\s*;", source):
            if m.group(1) in params:
                alias[name] = m.group(1)
    watch = (params | set(alias)) - bounded

    sites: list[tuple[int, str, str]] = []
    stack = [(fn, False)]
    while stack:
        node, in_loop = stack.pop()
        loop = in_loop or node.type in _LOOPS
        if loop and node.type in ("update_expression", "assignment_expression"):
            _advance_sites(src, node, watch, sites)
        for c in node.children:
            stack.append((c, loop))

    out: list[Sink] = []
    seen: set[int] = set()
    for line, ptr, kind in sorted(sites):
        if line in seen or not (func["line_start"] <= line <= func["line_end"]):
            continue
        seen.add(line)
        origin = alias.get(ptr, ptr)
        toks = [ptr] if ptr == origin else [ptr, origin]
        out.append(Sink(
            sink_id=f"{func['func_name']}:{line}",
            func_name=func["func_name"], line=line,
            snippet=lines[line - 1].strip() if 1 <= line <= len(lines) else "",
            sink_type="mem",
            tainted_tokens=toks,
            source="rule",
            walk_kind=kind,
        ))
    return out[:4]   # one walk usually hits at several sites; a few are enough
                      # to raise suspicion, and this avoids flooding noise


def build_op_summary(source: str, funcs: list[Func]) -> str:
    """The operation summary for select_knowledge: one line per function, listing
    its sensitive calls."""
    out: list[str] = []
    for f in funcs:
        calls = []
        for _node, name, _arg, _line in _iter_calls(source, f):
            if name in _API_TYPE and name not in calls:
                calls.append(name)
        if calls:
            out.append(f"- {f['func_name']}: {', '.join(calls)}")
        else:
            out.append(f"- {f['func_name']}: (no sensitive call)")
    return "\n".join(out)


# sink_type -> deterministic family floor (CWE root); select_knowledge drills down
# or falls back and merges it into the knowledge set.
# LLM family seeds under-recall; here a family lower bound is derived in reverse
# from "sensitive operations that the rules have already confirmed", and it is
# unioned with the LLM candidates so that integer-overflow / heap-overflow chains
# are never let through for lack of a seed.
_OP_FAMILY_FLOOR: dict[str, str] = {
    "mem": "CWE-119",     # memcpy/strcpy... => buffer overflow / out-of-bounds
                          # (119=>120/125)
    "alloc": "CWE-190",   # malloc/calloc... => size arithmetic wrap (integer
                          # overflow)
    "exec": "CWE-77",     # system/popen/exec... => command injection (77=>78)
    "file_op": "CWE-22",  # open/fopen... => path traversal (22=>23/36)
    "format": "CWE-134",  # the printf family => format string
}


def families_from_ops(source: str, funcs: list[Func]) -> list[str]:
    """Deterministically map detected sensitive calls onto family floors (CWE
    roots).

    LLM family seeds miss reports; this function relies only on sensitive APIs
    that the rules have already confirmed to match, and returns a family lower
    bound unaffected by LLM fluctuation, which select_knowledge unions with the
    LLM candidates.
    """
    fams: list[str] = []
    for f in funcs:
        for _node, name, _arg, _line in _iter_calls(source, f):
            t = _API_TYPE.get(name)
            fam = _OP_FAMILY_FLOOR.get(t) if t else None
            if fam and fam not in fams:
                fams.append(fam)
    return fams


_PHASE1_CALLS: dict[str, tuple[str, str]] = {
    # callee -> (mechanism_kind, sink_type)
    "luaL_loadstring": ("trust_boundary", "boundary_validation"),
    "luaL_dostring": ("trust_boundary", "boundary_validation"),
    "lua_pcall": ("trust_boundary", "boundary_validation"),
    "luaL_loadbuffer": ("trust_boundary", "boundary_validation"),
    "system": ("trust_boundary", "boundary_validation"),
    "popen": ("trust_boundary", "boundary_validation"),
    "execl": ("trust_boundary", "boundary_validation"),
    "execlp": ("trust_boundary", "boundary_validation"),
    "execle": ("trust_boundary", "boundary_validation"),
    "execv": ("trust_boundary", "boundary_validation"),
    "execvp": ("trust_boundary", "boundary_validation"),
    "execve": ("trust_boundary", "boundary_validation"),
    "xmlXPathEvalExpression": ("trust_boundary", "boundary_validation"),
    "xmlXPathEval": ("trust_boundary", "boundary_validation"),
    "set_header": ("trust_boundary", "boundary_validation"),
    "add_header": ("trust_boundary", "boundary_validation"),
    "open": ("initialization_cleanup", "resource_lifecycle"),
    "fopen": ("initialization_cleanup", "resource_lifecycle"),
    "openat": ("initialization_cleanup", "resource_lifecycle"),
    "close": ("initialization_cleanup", "resource_lifecycle"),
    "fclose": ("initialization_cleanup", "resource_lifecycle"),
    "pthread_mutex_lock": ("concurrency_order", "concurrency_state"),
    "pthread_mutex_unlock": ("concurrency_order", "concurrency_state"),
    "pthread_cond_wait": ("concurrency_order", "concurrency_state"),
    "pthread_cond_signal": ("concurrency_order", "concurrency_state"),
}

_MEMBER_ASSIGN = re.compile(
    r"(?:^|[^A-Za-z0-9_])([A-Za-z_]\w*)\s*(?:->|\.)\s*([A-Za-z_]\w*)\s*=(?!=)"
)


def _phase1_sink(func: Func, line: int, snippet: str, sink_type: str,
                 mechanism: str, subject: str, tokens: list[str]) -> Sink:
    return Sink(
        sink_id=f"{func['func_name']}:{line}:{sink_type}:{mechanism}:{subject}",
        func_name=func["func_name"], line=line, snippet=snippet,
        sink_type=sink_type, tainted_tokens=tokens or ([subject] if subject else []),
        source="rule", mechanism_kind=mechanism, subject=subject,
        obligations=ob.instantiate(sink_type),
    )


def rule_locate_phase1_patterns(source: str, func: Func) -> list[Sink]:
    """A small auditable pattern table: it raises candidates to be checked and
    never writes an API name as VIOLATED."""
    sinks: list[Sink] = []
    lines = source.split("\n")
    start, end = func["line_start"], func["line_end"]

    for _node, name, arg_text, line in _iter_calls(source, func):
        snippet = lines[line - 1].strip() if 1 <= line <= len(lines) else ""
        tokens = extract_tainted_tokens(arg_text)
        mapped = _PHASE1_CALLS.get(name)
        if name == "curl_easy_setopt" and re.search(r"\bCURLOPT_(?:URL|HEADER)\b", arg_text):
            mapped = ("trust_boundary", "boundary_validation")
        if not mapped:
            continue
        mechanism, sink_type = mapped
        subject = tokens[0] if tokens else name
        sinks.append(_phase1_sink(func, line, snippet, sink_type, mechanism, subject, tokens))

    for line_no in range(start, end + 1):
        if not (1 <= line_no <= len(lines)):
            continue
        match = _MEMBER_ASSIGN.search(lines[line_no - 1])
        if not match:
            continue
        obj, field = match.group(1), match.group(2)
        snippet = lines[line_no - 1].strip()
        sinks.append(_phase1_sink(
            func, line_no, snippet, "state_transition", "state_contract",
            obj, [obj, field],
        ))
    return sinks


def _sink_identity(sink: Sink) -> tuple:
    subject = sink.get("subject") or ""
    tokens = tuple(sink.get("tainted_tokens") or [])
    distinguishing = subject or (tokens[0] if tokens else "")
    return (
        sink.get("func_name") or "",
        int(sink.get("line") or 0),
        sink.get("sink_type") or "",
        sink.get("mechanism_kind") or "",
        distinguishing,
    )


def merge_sinks(rule_sinks: list[Sink], llm_sinks: list[Sink]) -> list[Sink]:
    """Merge on (func, line, sink_type, mechanism, subject); deduplicate only on
    exact equivalence."""
    by_key: dict[tuple, Sink] = {}
    for s in rule_sinks:
        by_key[_sink_identity(s)] = s
    for s in llm_sinks:
        key = _sink_identity(s)
        if key in by_key:
            ex = by_key[key]
            if ex.get("source") != s.get("source"):
                ex["source"] = "both"
            for t in s.get("tainted_tokens") or []:
                if t not in ex["tainted_tokens"]:
                    ex["tainted_tokens"].append(t)
            for field in ("change_kind", "contract_witness", "obligations", "cwe_id",
                          "mechanism_kind", "subject"):
                if field in s:
                    ex[field] = s[field]
        else:
            by_key[key] = s
    return [by_key[k] for k in sorted(by_key, key=lambda k: (k[0], k[1], k[2], k[3], k[4]))]
