"""context_search: a dataflow-directed grep loop with two hard caps against
divergence.

- Only dataflow-related lines are followed (assignment, argument passing,
  concatenation); unrelated references are dropped.
- MAX_VARS / MAX_ROUNDS are hard caps, and the scope is locked per mode:
  - single-file mode: search only the current file (source_text);
  - project mode: search across all project source files (project_index.sources) with
    a file:line prefix on every hit, so forensics can follow the taint across files
    back to its producer (e.g. tracking msg->len in auth.c hits its origin in
    network.c).
- The forensics node calls search_context each round with the LLM's
  need_more.tokens; the round and variable counts are enforced by counters inside
  ContextSearcher, so the loop is truncated even if the LLM keeps asking for more.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .. import config


class ContextSearcher:
    """A forensic search session for a single sink, maintaining the round and
    variable budgets internally.

    Single file: pass `source` (the full text of this file); the scope is locked to
    this file.
    Project: additionally pass project_index to grep across all project files; the
    current file still comes first and hits carry a relative-path prefix.
    """

    def __init__(
        self, source: str, scope_label: str = "<file>",
        project_index: Optional[dict] = None, current_path: str = "",
    ):
        self._lines = source.split("\n")
        self._scope = scope_label
        self._index = project_index
        self._current = current_path
        self.rounds_used = 0
        self._searched_vars: set[str] = set()

    @property
    def rounds_left(self) -> int:
        return max(0, config.CTX_MAX_ROUNDS - self.rounds_used)

    def can_search(self) -> bool:
        return self.rounds_left > 0 and len(self._searched_vars) < config.CTX_MAX_VARS

    def search(self, tokens: list[str]) -> str:
        """Search the dataflow-related lines of a batch of variables and return a
        summary for the LLM. Consumes one round."""
        if not self.can_search():
            return "(search budget exhausted, no further results)"
        self.rounds_used += 1

        # take new variables within the variable cap
        picks: list[str] = []
        for t in tokens:
            if t in self._searched_vars:
                continue
            if len(self._searched_vars) >= config.CTX_MAX_VARS:
                break
            self._searched_vars.add(t)
            picks.append(t)
        if not picks:
            picks = [t for t in tokens][:1]  # always return at least something

        blocks: list[str] = []
        for var in picks:
            hits = self._grep_var(var)
            if hits:
                joined = "\n".join(f"  {loc}| {txt}" for loc, txt in hits)
                blocks.append(f"[{var}] scope={self._scope}\n{joined}")
            else:
                blocks.append(f"[{var}] no dataflow-related hits")
        return "\n".join(blocks)

    # ---- dataflow hints: assignment, argument position, concatenation,
    # declaration with initialiser ----
    _FLOW_HINTS = ("=", "(", ",", "+", "strcpy", "memcpy", "sprintf", "strcat")

    def _grep_var(self, var: str) -> list[tuple[str, str]]:
        """Search the dataflow-related lines of `var` within the scope. Single-file
        mode returns (line number, text); in project mode the line number carries a
        relative-path prefix (file:line) so that the LLM can tell cross-file hits
        apart."""
        if self._index is None:
            return [(str(ln), txt) for ln, txt in self._grep_lines(self._lines, var)]
        # Project mode: the current file first, then the remaining files. The total
        # number of hits keeps the per-variable cap of 12.
        out: list[tuple[str, str]] = []
        sources: dict = self._index.get("sources", {})
        root = self._index.get("root", "")
        # The current file is searched first (reusing the split self._lines) so that
        # the nearby context takes precedence.
        ordered = ([self._current] if self._current in sources else []) + [
            p for p in sources if p != self._current
        ]
        for path in ordered:
            lines = self._lines if path == self._current else sources[path].split("\n")
            label = _rel(root, path)
            for ln, txt in self._grep_lines(lines, var):
                out.append((f"{label}:{ln}", txt))
                if len(out) >= 12:
                    return out
        return out

    def _grep_lines(self, lines: list[str], var: str) -> list[tuple[int, str]]:
        """Keep only the lines of a set that are dataflow-related to `var`:
        assignment / argument passing / concatenation / declaration."""
        pat = re.compile(rf"\b{re.escape(var)}\b")
        out: list[tuple[int, str]] = []
        for i, line in enumerate(lines, start=1):
            if not pat.search(line):
                continue
            stripped = line.strip()
            if any(h in stripped for h in self._FLOW_HINTS):
                out.append((i, stripped))
            if len(out) >= 12:  # per-variable cap in one file, to avoid noise
                break
        return out


def _rel(root: str, path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except Exception:
        return Path(path).name
