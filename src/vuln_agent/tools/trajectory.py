"""trajectory: structured proof-trajectory recording (evidence-chain
introspection / debugging).

Distinct from the LLM call trace in llm/budgeted.py (which records only
tag/tokens/latency), this module records the **action steps that advance the
proof state**: each step carries action / inputs / observation / state_delta /
evidence_refs / cost, which makes it possible to check offline whether every
obligation behind a positive verdict has an anchor that the checker actually
verified (the S3 reasoning-correctness protocol).

Design:
- A global thread-safe accumulator (arbiter/forensics use parallel_map
  concurrently, so a lock is required).
- Off by default; enable_trajectory() turns it on (linked to --trajectory-out in
  run.py).
- Each step is bound to a sink_id plus a cwe/sink_type bucket key.
- Instrumentation is a **side-channel record** and changes no verdict; when
  disabled, record is near-zero cost.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_STEPS: list[dict] = []
_ENABLED = False
_COUNTER = 0


def enable_trajectory() -> None:
    global _ENABLED, _COUNTER
    _ENABLED = True
    with _LOCK:
        _STEPS.clear()
        _COUNTER = 0


def disable_trajectory() -> None:
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def get_trajectory() -> list[dict]:
    with _LOCK:
        return list(_STEPS)


def clear_trajectory() -> None:
    global _COUNTER
    with _LOCK:
        _STEPS.clear()
        _COUNTER = 0


def record_step(
    *,
    node: str,
    action: str,
    sink_id: str = "",
    bucket: str = "",
    inputs: Optional[dict] = None,
    observation: Optional[dict] = None,
    state_delta: Optional[dict] = None,
    evidence_refs: Optional[list[str]] = None,
    cost: Optional[dict] = None,
    decision: Any = None,
) -> None:
    """Record one proof-trajectory step. Returns immediately when disabled (zero
    cost).

    - node: the name of the node that produced the step
      (secure_diff/risk_forensics/arbiter/project_dataflow).
    - action: an action name from the action vocabulary (see the action backends in
      tools/obligations).
    - sink_id: the bound candidate point, used to attribute the trajectory to a
      compiled unit.
    - bucket: the bucket key "(cwe family|sink_type)", by which the compiler votes
      per bucket.
    - state_delta: the transition {obligation_id: {"from": status, "to": status}}.
    - evidence_refs: the source-level evidence references retrieved by this step
      (file:line).
    - decision: only the arbiter's final-review step fills in the verdict
      (True/False/None).
    """
    if not _ENABLED:
        return
    global _COUNTER
    with _LOCK:
        _COUNTER += 1
        _STEPS.append({
            "step": _COUNTER,
            "node": node,
            "action": action,
            "sink_id": sink_id,
            "bucket": bucket,
            "inputs": inputs or {},
            "observation": observation or {},
            "state_delta": state_delta or {},
            "evidence_refs": list(evidence_refs or []),
            "cost": cost or {},
            "decision": decision,
        })
