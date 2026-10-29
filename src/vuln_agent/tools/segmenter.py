"""segmenter: tree-sitter C/C++ function splitting + the feeding decision.

- split_functions: parses a source file and returns a list of Func (function name,
  start/end line, language).
- number_lines: adds 1-based line numbers to a code segment (both LLM localisation
  and validation rely on real line numbers).
- strip_line_numbers: removes the line numbers (the input of the family
  classification header must be number-free); the separating space is kept, so this
  is the inverse of number_lines up to one leading space per line -- it matches the
  shape of the training corpus and must not be "tidied up".
- prepare_unit: feed the whole segment when it is at or below the threshold (full);
  above the threshold, apply structured compression (compressed) -- keep the
  signature, the control-flow-critical lines, the sink hit lines and their
  neighbourhood, and elide the rest behind a placeholder; the boundaries are
  semantic and never cut a construct in half.
"""
from __future__ import annotations

import re
import warnings
from typing import Optional

from tree_sitter_languages import get_parser

from .. import config
from ..state import Func, Unit

warnings.filterwarnings("ignore", category=FutureWarning)

_EXT_LANG = {
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
}


def detect_lang(path: str) -> str:
    for ext, lang in _EXT_LANG.items():
        if path.endswith(ext):
            return lang
    return "c"  # default to C


def _func_name(node, src: bytes) -> str:
    """Extract the function name from the declarator of a function_definition."""
    decl = node.child_by_field_name("declarator")
    cur = decl
    # Drill down layer by layer to identifier / field_identifier / operator_name
    while cur is not None:
        if cur.type in ("identifier", "field_identifier", "operator_name",
                         "destructor_name", "qualified_identifier"):
            return src[cur.start_byte:cur.end_byte].decode("utf-8", "replace")
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            # The name of a function_declarator sits in its declarator; failing
            # that, try the first named child.
            named = [c for c in cur.children if c.is_named]
            nxt = named[0] if named else None
        cur = nxt
    return "<anonymous>"


def split_functions(source: str, lang: str) -> list[Func]:
    parser = get_parser("cpp" if lang == "cpp" else "c")
    src = source.encode("utf-8")
    tree = parser.parse(src)
    funcs: list[Func] = []

    def visit(node):
        if node.type == "function_definition":
            funcs.append(Func(
                func_name=_func_name(node, src),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                lang=lang,  # type: ignore[typeddict-item]
            ))
            return  # do not recurse into the body (avoids misjudging nested
                    # lambdas)
        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return funcs


def number_lines(text: str, start_line: int = 1) -> str:
    """Prefix every line of the code with 'line| '."""
    lines = text.split("\n")
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(
        f"{start_line + i:>{width}}| {ln}" for i, ln in enumerate(lines)
    )


# number_lines writes "{n}| " (a pipe followed by one space). The inverse operation
# leaves that space **to the code** rather than dropping it: that is the shape of
# the corpus the family classification header was trained on (the training-side
# extraction uses a regex of the same shape and likewise captures the separating
# space), and the encoder is a byte-level BPE -- "\n int" and "\nint" cut into
# different tokens. Losing one space at inference time amounts to shipping a
# different distribution.
_NUMBERED_RE = re.compile(r"^\s*\d+\|(.*)$")


def strip_line_numbers(numbered: str) -> str:
    """Strip the 'line|' prefix written by number_lines, **keeping the separating
    space that follows it**.

    Hence it is the inverse of number_lines up to one leading space per line, not
    a strict inverse -- that offset is deliberately preserved: the corpus has this
    shape (see the comment above) and the frozen development metric was measured on
    the space-bearing data. Lines that do not match are dropped entirely -- the
    folded placeholder lines of compress_source ('  | /* ... N lines omitted ... */')
    carry no line number and are likewise dropped on the training side.
    """
    out = []
    for line in numbered.split("\n"):
        m = _NUMBERED_RE.match(line)
        if m is not None:
            out.append(m.group(1))
    return "\n".join(out)


def func_source(source: str, func: Func) -> str:
    """Take the function text by lines (without line numbers)."""
    lines = source.split("\n")
    return "\n".join(lines[func["line_start"] - 1: func["line_end"]])


# Control-flow / structural keywords: these lines are kept during compression
# (to preserve the structural skeleton).
_STRUCT_KW = (
    "if", "else", "for", "while", "switch", "case", "do",
    "return", "goto", "break", "continue", "{", "}",
)


def _is_struct_line(stripped: str) -> bool:
    if stripped in ("{", "}") or stripped.endswith("{") or stripped.startswith("}"):
        return True
    for kw in _STRUCT_KW:
        if stripped == kw or stripped.startswith(kw + " ") or stripped.startswith(kw + "("):
            return True
    return False


def prepare_unit(
    source: str, func: Func, keep_lines: Optional[set[int]] = None
) -> Unit:
    """Decide how to feed the function.

    keep_lines: absolute line numbers that must additionally be kept (e.g. sink hit
    lines). During compression they are kept together with their neighbourhood. The
    first round (no sink yet) passes None and compresses by signature plus control
    flow only.
    """
    n_lines = func["line_end"] - func["line_start"] + 1
    if n_lines <= config.FUNC_FULL_MAX_LINES:
        body = func_source(source, func)
        return Unit(
            func_name=func["func_name"], kind="full",
            content=number_lines(body, func["line_start"]),
        )

    return Unit(
        func_name=func["func_name"], kind="compressed",
        content=_compress(source, func, keep_lines or set()),
    )


def _compress(source: str, func: Func, keep_lines: set[int]) -> str:
    """Structured compression: keep the signature / control flow / the
    keep_lines neighbourhood, and elide the rest in blocks."""
    all_lines = source.split("\n")
    s, e = func["line_start"], func["line_end"]
    nbhd = config.SINK_NEIGHBORHOOD

    keep: set[int] = set()
    # Function signature region: the start through the line holding the first '{'.
    sig_end = s
    for ln in range(s, e + 1):
        keep.add(ln)  # placeholder first, refined below; this keeps a multi-line
                      # signature too
        if "{" in all_lines[ln - 1]:
            sig_end = ln
            break
    keep = {ln for ln in range(s, sig_end + 1)}

    # Control-flow skeleton
    for ln in range(s, e + 1):
        if _is_struct_line(all_lines[ln - 1].strip()):
            keep.add(ln)
    # Sink hit lines and their neighbourhood
    for ln in keep_lines:
        if s <= ln <= e:
            for k in range(ln - nbhd, ln + nbhd + 1):
                if s <= k <= e:
                    keep.add(k)
    keep.add(e)  # the closing '}'

    return _emit_folded(all_lines, s, e, keep)


def _emit_folded(all_lines: list, s: int, e: int, keep: set[int]) -> str:
    """Emit [s, e] line by line: lines inside keep are kept verbatim with their
    absolute line number, and each run of dropped lines is folded into a
    '/* ... N lines omitted ... */' placeholder. Shared by _compress and
    compress_source."""
    width = len(str(e))
    out: list[str] = []
    ln = s
    while ln <= e:
        if ln in keep:
            out.append(f"{ln:>{width}}| {all_lines[ln - 1]}")
            ln += 1
        else:
            gap_start = ln
            while ln <= e and ln not in keep:
                ln += 1
            omitted = ln - gap_start
            pad = " " * width
            out.append(f"{pad}| /* ... {omitted} lines omitted ... */")
    return "\n".join(out)


# Leading keywords of top-level declarations/definitions: these lines are kept
# when a whole file is compressed structurally (the body of a declaration-dense
# header lives at the top level, where function-level compression does not apply).
# A hit keeps that line (together with the control-flow skeleton and the
# neighbourhoods of already recalled lines).
_DECL_PREFIXES = (
    "#", "typedef", "struct", "union", "enum", "class", "namespace",
    "template", "using", "static", "extern", "const", "inline",
    "void", "int", "char", "long", "short", "unsigned", "signed",
    "float", "double", "bool", "size_t", "uint", "int8", "int16",
    "int32", "int64", "u8", "u16", "u32", "u64",
)


def compress_source(source: str, keep_lines: Optional[set[int]] = None) -> str:
    """Structured compression of a whole file (applied to the entire source rather
    than to a single function): keeps top-level declaration/definition lines, the
    control-flow skeleton, keep_lines (recalled sink lines) and their
    SINK_NEIGHBORHOOD neighbourhoods, and folds each remaining run into a
    placeholder. Line numbers stay absolute (1-based), so basis/location line
    numbers still point back into the original text.

    Used for very large files that exceed SOURCE_MAX_LINES (e.g. declaration-dense
    headers), so that a whole file is never pushed into the prompt at once.
    """
    all_lines = source.split("\n")
    n = len(all_lines)
    if n == 0:
        return ""
    nbhd = config.SINK_NEIGHBORHOOD
    keep: set[int] = set()
    for ln in range(1, n + 1):
        stripped = all_lines[ln - 1].strip()
        if not stripped:
            continue
        if _is_struct_line(stripped) or stripped.startswith(_DECL_PREFIXES):
            keep.add(ln)
    # Recalled sink lines and their neighbourhood
    for ln in (keep_lines or set()):
        if 1 <= ln <= n:
            for k in range(ln - nbhd, ln + nbhd + 1):
                if 1 <= k <= n:
                    keep.add(k)
    return _emit_folded(all_lines, 1, n, keep)
