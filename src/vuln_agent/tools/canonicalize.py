"""canonicalize (step 3, source normalisation, deterministic and zero token): folds
statement continuations that span several physical lines.

Rationale: a patch often reflows one statement over several lines (splitting a single
assignment into three, say). Such a purely *formatting-level perturbation* changes the
token layout of the prompt, which combined with the server-side non-determinism of a
MoE model is enough to make one and the same piece of semantic code receive different
verdicts. Normalisation folds a continuation with unclosed parentheses or brackets
back onto the logical line's first physical line, so that the vulnerable and the
patched version present exactly the same text wherever there is no semantic
difference.

Hard constraint: the *total line count is preserved*, with continuation positions
turned into blank lines. Line numbers are the coordinate system for anchors along the
whole pipeline (sink localisation, diff anchoring, obligation evidence, report
back-references) and normalisation never moves them; a folded statement lands on its
first physical line.

Conservative boundary (preferring not to fold over risking a change of meaning):
  - continuations are detected by parenthesis and bracket depth only; braces are not
    counted (counting { } would fold an entire function body onto one line);
  - preprocessor directives (#...) and macros continued with a backslash are never
    touched;
  - brackets inside string and character literals, line comments and block comments do
    not contribute to the depth;
  - folding is abandoned and the original text kept when the span exceeds
    MAX_JOIN_LINES or the joined text exceeds MAX_JOIN_CHARS.
"""
from __future__ import annotations

MAX_JOIN_LINES = 12   # max number of physical lines folded into one statement (a
                      # larger span is treated as unusual layout and left alone)
MAX_JOIN_CHARS = 400  # max length of the logical line after folding (guards against a
                      # pathologically long line blowing up the window)


def _scan(line: str, in_block: bool) -> tuple[int, bool, bool]:
    """Scan one line and return (net depth increment of parentheses and brackets,
    whether the line ends inside a block comment, whether it contains a line
    comment).

    Brackets inside string and character literals and inside comments are not
    counted. The line-comment flag is used for folding safety: appending code after a
    // would comment it out, so a non-final line carrying a line comment is never
    folded."""
    depth = 0
    i, n = 0, len(line)
    in_str = in_chr = False
    has_lc = False
    while i < n:
        c = line[i]
        if in_block:
            if c == "*" and i + 1 < n and line[i + 1] == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if in_chr:
            if c == "\\":
                i += 2
                continue
            if c == "'":
                in_chr = False
            i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            has_lc = True
            break  # line comment: the rest of the line is not counted
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            in_block = True
            i += 2
            continue
        if c == '"':
            in_str = True
        elif c == "'":
            in_chr = True
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        i += 1
    return depth, in_block, has_lc


def _is_directive(line: str) -> bool:
    return line.lstrip().startswith("#")


def canonicalize_source(src: str) -> str:
    """Fold continuations (line count preserved). See the module docstring."""
    lines = src.split("\n")
    out = list(lines)
    n = len(lines)
    i = 0
    in_block = False
    while i < n:
        line = lines[i]
        if in_block or _is_directive(line) or line.rstrip().endswith("\\"):
            _, in_block, _ = _scan(line, in_block)
            i += 1
            continue
        depth, block_after, lc = _scan(line, in_block)
        if depth <= 0 or lc:
            # Brackets already closed, or the line ends with a // comment (joining
            # a continuation after the comment would comment the code out) -> no
            # fold.
            in_block = block_after
            i += 1
            continue
        # Brackets unclosed at end of line -> try folding forward until the depth
        # returns to zero.
        j = i
        total = depth
        buf = [line.rstrip()]
        bc = block_after
        closed = False
        while j + 1 < n and (j - i) < MAX_JOIN_LINES:
            j += 1
            nxt = lines[j]
            if _is_directive(nxt) or nxt.rstrip().endswith("\\"):
                break  # a directive or macro continuation mixed into the
                       # argument list -> abandon folding
            d, bc, lc = _scan(nxt, bc)
            buf.append(nxt.strip())
            total += d
            if total <= 0:
                closed = True
                break
            if lc:
                break  # a non-final line carries a line comment, so joining would
                   # comment out the following code -> abandon folding
        joined = " ".join(x for x in buf if x)
        if closed and len(joined) <= MAX_JOIN_CHARS:
            out[i] = joined
            for k in range(i + 1, j + 1):
                out[k] = ""
            in_block = bc
            i = j + 1
        else:
            in_block = block_after
            i += 1
    return "\n".join(out)
