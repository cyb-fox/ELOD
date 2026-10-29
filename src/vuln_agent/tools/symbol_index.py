"""Lightweight, build-free C/C++ repository symbol index.

The index is deliberately syntactic.  It records source spans for definitions,
declarations, calls, and local includes, but does not attempt type resolution,
macro expansion, overload resolution, or a complete call graph.
"""
from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Iterable

from tree_sitter_languages import get_parser


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _parser_for(lang: str):
    return get_parser("cpp" if lang == "cpp" else "c")


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _name_from_declarator(node, src: bytes) -> str:
    """Return the innermost useful name from a declarator node."""
    cur = node
    while cur is not None:
        if cur.type in {
            "identifier", "field_identifier", "operator_name",
            "destructor_name", "qualified_identifier",
        }:
            return _text(src, cur)
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            named = [child for child in cur.children if child.is_named]
            nxt = named[0] if named else None
        cur = nxt
    return ""


def _span(path: str, node, kind: str, name: str = "") -> dict:
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    out = {
        "path": path,
        "line_start": start_row + 1,
        "line_end": end_row + 1,
        "column_start": start_col,
        "column_end": end_col,
        "kind": kind,
    }
    if name:
        out["name"] = name
    return out


def _callee_name(node, src: bytes) -> str:
    function = node.child_by_field_name("function")
    if function is None:
        return ""
    raw = _text(src, function)
    # Keep the final component for member/qualified calls, matching the
    # existing sink locator's conservative name normalization.
    names = _IDENTIFIER.findall(raw)
    return names[-1] if names else raw


def _walk(node) -> Iterable:
    yield node
    for child in node.children:
        yield from _walk(child)


def _enclosing_definition(node, definitions: list[dict]) -> str:
    line = node.start_point[0] + 1
    containing = [item for item in definitions
                  if item["line_start"] <= line <= item["line_end"]]
    if not containing:
        return ""
    # Nested syntactic definitions are rare in C/C++, but choosing the
    # smallest containing span is deterministic if a lambda is represented as
    # a function-like node by a grammar version.
    return min(containing, key=lambda item: item["line_end"] - item["line_start"])["name"]


def _include_target(path: str, raw: str, files: list[str]) -> str:
    """Resolve a local include when possible; unresolved targets stay visible."""
    source = Path(path).resolve()
    root_candidates = [source.parent / raw, Path(raw)]
    file_set = {str(Path(item).resolve()) for item in files}
    for candidate in root_candidates:
        resolved = str(candidate.resolve())
        if resolved in file_set:
            return resolved
    wanted = posixpath.basename(raw)
    matches = sorted(item for item in file_set if Path(item).name == wanted)
    return matches[0] if matches else ""


def _index_file(path: str, source: str, files: list[str], lang: str) -> dict:
    src = source.encode("utf-8")
    tree = _parser_for(lang).parse(src)
    definitions: list[dict] = []
    declarations: list[dict] = []

    for node in _walk(tree.root_node):
        if node.type == "function_definition":
            name = _name_from_declarator(node.child_by_field_name("declarator"), src)
            if name:
                item = _span(path, node, "definition", name)
                item["lang"] = lang
                definitions.append(item)
        elif node.type == "declaration":
            declarators = [child for child in _walk(node)
                           if child.type == "function_declarator"]
            for declarator in declarators:
                name = _name_from_declarator(declarator, src)
                if name:
                    declarations.append(_span(path, declarator, "declaration", name))

    call_sites: list[dict] = []
    for node in _walk(tree.root_node):
        if node.type != "call_expression":
            continue
        callee = _callee_name(node, src)
        if not callee:
            continue
        item = _span(path, node, "call", callee)
        item["caller"] = _enclosing_definition(node, definitions)
        call_sites.append(item)

    includes: list[dict] = []
    for node in _walk(tree.root_node):
        if node.type != "preproc_include":
            continue
        raw = _text(src, node)
        match = re.search(r"#\s*include\s+\"([^\"]+)\"", raw)
        if not match:
            continue  # system headers are outside the repository index
        include_name = match.group(1)
        includes.append({
            "from": path,
            "target": include_name,
            "resolved": _include_target(path, include_name, files),
            "line": node.start_point[0] + 1,
            "local": True,
        })

    return {
        "definitions": definitions,
        "declarations": declarations,
        "call_sites": call_sites,
        "includes": includes,
    }


def build_symbol_index(sources: dict[str, str], root: str = "") -> dict:
    """Build a deterministic repository index from an in-memory source snapshot."""
    files = sorted(sources)
    definitions: dict[str, list[dict]] = {}
    declarations: dict[str, list[dict]] = {}
    call_sites: dict[str, list[dict]] = {}
    includes: list[dict] = []

    for path in files:
        suffix = Path(path).suffix.lower()
        lang = "cpp" if suffix in {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"} else "c"
        try:
            parsed = _index_file(path, sources[path], files, lang)
        except Exception:
            # A malformed translation unit must not make the whole project
            # unavailable; the existing text/symbol paths remain usable.
            continue
        for item in parsed["definitions"]:
            definitions.setdefault(item["name"], []).append(item)
        for item in parsed["declarations"]:
            declarations.setdefault(item["name"], []).append(item)
        for item in parsed["call_sites"]:
            call_sites.setdefault(item["name"], []).append(item)
        includes.extend(parsed["includes"])

    reverse_refs: dict[str, list[dict]] = {}
    for name, items in call_sites.items():
        reverse_refs.setdefault(name, []).extend(
            {**item, "ref_kind": "call"} for item in items
        )
    for name, items in declarations.items():
        reverse_refs.setdefault(name, []).extend(
            {**item, "ref_kind": "declaration"} for item in items
        )

    # Stable ordering makes reports and tests reproducible across runs.
    for bucket in (definitions, declarations, call_sites, reverse_refs):
        for name in bucket:
            bucket[name].sort(key=lambda item: (
                item.get("path", ""), item.get("line_start", item.get("line", 0)),
                item.get("column_start", 0), item.get("kind", ""),
            ))
    includes.sort(key=lambda item: (item["from"], item["line"], item["target"]))

    return {
        "definitions": definitions,
        "declarations": declarations,
        "call_sites": call_sites,
        "includes": includes,
        "reverse_refs": reverse_refs,
    }
