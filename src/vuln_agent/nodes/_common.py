"""Shared helpers for the node subpackage."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

from .. import config
from ..llm import BudgetedLLM
from ..llm.budgeted import trace_context, use_trace_context
from ..state import VulnState
from ..tools import cfg as cfgmod
from ..tools import segmenter as sg

_T = TypeVar("_T")
_R = TypeVar("_R")

# (target_path, line) -> CFG | "MISS"; all lines of one function share one graph.
_CFG_CACHE: dict = {}


def make_llm(state: VulnState, tag: str = "") -> BudgetedLLM:
    """Build the LLM from the budget reference held in state (accumulated in
    place, thread-safe).

    `tag` is the owning node name, used for trace grouping (optional).
    """
    return BudgetedLLM(state["budget"], tag=tag)


def parallel_map(fn: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
    """Process each sink concurrently in a thread pool (IO-bound: waiting on LLM
    network calls).  Returns in input order.

    A BudgetExceeded raised by a worker propagates upward and is caught by
    budget_guard, which sets partial.

    Worker threads inherit the caller's trace records: the trace is thread-local
    (so several files running concurrently do not cross-talk), but fan-out inside
    one node is concurrency **within a single file**, where a fresh thread would
    otherwise start with an empty table and every call made by the worker would be
    silently dropped -- a file split into several batches would then be recorded
    as consuming zero token.
    """
    items = list(items)
    if not items:
        return []
    workers = max(1, min(config.SINK_MAX_WORKERS, len(items)))
    if workers == 1:
        return [fn(x) for x in items]
    records = trace_context()

    def _run(x: _T) -> _R:
        use_trace_context(records)
        return fn(x)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run, items))


def risk_window(state: VulnState, line: int) -> str:
    """Context window around a risk line (with line numbers), for the
    rewriting/forensics nodes."""
    lines = state["source_text"].split("\n")
    lo = max(1, line - config.RISK_WINDOW)
    hi = min(len(lines), line + config.RISK_WINDOW)
    body = "\n".join(lines[lo - 1: hi])
    return sg.number_lines(body, lo)


def bounded_source(state: VulnState, keep_lines: set[int] | None = None) -> str:
    """Single entry point for whole-file feeding: small files are numbered in
    full; beyond config.SOURCE_MAX_LINES a structured digest is used
    (signatures + control-flow skeleton + neighbourhoods of recalled lines),
    which avoids pushing a huge file into the prompt wholesale and blowing the
    budget.  `keep_lines` carries already-recalled sink lines so that the
    neighbourhood of dangerous lines is never squeezed out.  Line numbers stay
    absolute and keep pointing back into the original text."""
    src = state["source_text"]
    if config.SOURCE_MAX_LINES <= 0 or src.count("\n") + 1 <= config.SOURCE_MAX_LINES:
        return sg.number_lines(src, 1)
    return sg.compress_source(src, keep_lines)


def cfg_for_line(state: VulnState, line: int):
    """CFG of the function owning `line` (deterministic, zero token), cached
    per process by (file, function).

    Witness checking and true-dominance re-verification both need it; several
    sinks of one function build the graph only once.  Returns None when no graph
    can be built, and the caller then falls back to the anchor channel plus a
    line-number approximation (a weaker capability, which can only prove fewer
    obligations, never add positives).
    """
    key = (state.get("target_path", ""), int(line))
    hit = _CFG_CACHE.get(key)
    if hit is not None:
        return hit if hit != "MISS" else None
    if len(_CFG_CACHE) > 20000:           # bound growth on full evaluation runs
        _CFG_CACHE.clear()
    try:
        g = cfgmod.build_for_line(state.get("source_text", ""),
                                  state.get("functions") or [], int(line))
    except Exception:                     # graph building never breaks the main chain
        g = None
    for f in state.get("functions") or []:
        if f.get("line_start", 0) <= line <= f.get("line_end", 0):
            # All lines of one function share a single graph, avoiding a rebuild
            # per sink.
            for ln in range(int(f["line_start"]), int(f["line_end"]) + 1):
                _CFG_CACHE[(key[0], ln)] = g or "MISS"
            break
    else:
        _CFG_CACHE[key] = g or "MISS"
    return g


# Confidence is tiered deterministically by evidence-source strength (see
# nodes/arbiter._confidence): model self-reported reachability / impact /
# trigger_difficulty carry no discriminative power on paired samples, where both
# sides share the same code and the same self-assessment, so down-weighting them
# is symmetric and only adds a further layer of non-determinism.


def pack_batches(sinks, batch_max: int | None = None) -> list[list]:
    """Pack sinks into batches of <=batch_max by "function atom": all sinks of
    one function are never split across batches (preserving function-level
    context and the integrity of the secure diff); adjacent small functions are
    packed together where possible to amortize the fixed schema tax; a function
    whose sink count exceeds batch_max gets a batch to itself (slightly overfull
    is preferable to fragmenting a function).  Order-preserving: both within and
    across batches.

    Returns [[sink,...], ...], one LLM call per batch for forensics/arbiter.
    """
    cap = batch_max if batch_max is not None else config.SINK_BATCH_MAX
    cap = max(1, cap)
    # Aggregate by function first (order-preserving), giving
    # [(func_name,[sinks]), ...]
    by_func: dict[str, list] = {}
    for s in sinks:
        by_func.setdefault(s["func_name"], []).append(s)
    batches: list[list] = []
    cur: list = []
    for _fn, group in by_func.items():
        if len(group) >= cap:
            # One function already at/over the cap: flush the batch being
            # accumulated, this function takes a batch alone (keeps it whole).
            if cur:
                batches.append(cur)
                cur = []
            batches.append(list(group))
            continue
        if len(cur) + len(group) > cap:
            batches.append(cur)
            cur = []
        cur.extend(group)
    if cur:
        batches.append(cur)
    return batches
