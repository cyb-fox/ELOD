"""graph.py: node registration + conditional-edge assembly.

Single-chain topology (nothing on the decision chain is trained; file/project
modes share it; the LLM appears only in **proposal** nodes):
    ingest → select_knowledge → secure_diff → risk_forensics → arbiter
          → project_dataflow → output
  - select_knowledge: the knowledge-loading route, **produces no evidence and
    takes no part in the verdict**. By default it uses the local family
    classification head (zero token, `models/clf_select`) and fills
    family_context (the family knowledge cards);
    `SELECT_KNOWLEDGE_BACKEND=llm` switches to the LLM backend, which adds a
    two-level menu drill-down plus a suspicions channel -- since that backend
    really calls the LLM, the budget_guard wrapper below must be kept (under clf
    it is a no-op).
  - secure_diff: view A's candidate gate -- the model generates a safe version
    and code applies a deterministic diff / counterfactual guard gate.
  - risk_forensics: view B's independent forensics (grep loop); the LLM only
    proposes candidate facts.
  - arbiter: **deterministic verdict aggregation, zero LLM calls** -- it
    aggregates, per I2/I3, the evidence-backed obligation snapshots written by
    obligation_checker (the single discharger). See the nodes/arbiter module
    documentation for the aggregation rules.
  - project_dataflow: cross-file taint-chain recovery (in file mode there is no
    project_index, so it passes through with zero calls).

Budget short-circuit: when any LLM node raises BudgetExceeded, budget_guard
catches it, sets partial=True and writes the _budget_blown flag into state; the
route after each LLM node checks that flag first and heads straight to output
once the budget is blown.
"""
from __future__ import annotations

from functools import wraps

from langgraph.graph import END, START, StateGraph

from .llm import BudgetExceeded
from .nodes.arbiter import arbiter
from .nodes.ingest import ingest
from .nodes.output import output
from .nodes.project_dataflow import project_dataflow
from .nodes.risk_forensics import risk_forensics
from .nodes.secure_diff import secure_diff
from .nodes.select_knowledge import select_knowledge
from .state import VulnState


def budget_guard(fn):
    """Wraps an LLM node: catches BudgetExceeded, sets partial and the
    short-circuit flag."""
    @wraps(fn)
    def wrapper(state: VulnState) -> dict:
        if state.get("_budget_blown"):
            return {}  # budget already blown; the node is a no-op and the route
                       # sends execution to output
        try:
            return fn(state)
        except BudgetExceeded as e:
            errs = list(state.get("errors", []) or [])
            errs.append(f"BudgetExceeded in {fn.__name__}: {e}")
            return {"partial": True, "_budget_blown": True, "errors": errs}
    return wrapper


def _route_after_ingest(state: VulnState) -> str:
    """Prescreen short-circuit for the scan profile: a file with no danger
    indication goes straight to output, skipping the entire LLM chain.
    The bench profile never sets _prescreen_clean, so it always goes through
    select_knowledge (behaviour unchanged)."""
    if state.get("_prescreen_clean"):
        return "output"
    return "select_knowledge"


def _linear(next_node: str):
    """The straight-through route after a serial LLM node, diverting to output
    only when the budget is blown."""
    def router(state: VulnState) -> str:
        return "output" if state.get("_budget_blown") else next_node
    return router


def build_graph():
    g = StateGraph(VulnState)

    # Deterministic nodes
    g.add_node("ingest", ingest)
    g.add_node("output", output)
    # arbiter makes no LLM calls (deterministic aggregation), so it needs no
    # budget_guard -- it consumes no budget.
    g.add_node("arbiter", arbiter)

    # LLM nodes (wrapped by budget_guard)
    g.add_node("select_knowledge", budget_guard(select_knowledge))
    g.add_node("secure_diff", budget_guard(secure_diff))
    g.add_node("risk_forensics", budget_guard(risk_forensics))
    g.add_node("project_dataflow", budget_guard(project_dataflow))

    g.add_edge(START, "ingest")
    # After ingest: a scan-profile prescreen-clean file goes straight to output;
    # otherwise into select_knowledge.
    g.add_conditional_edges("ingest", _route_after_ingest,
                            {"select_knowledge": "select_knowledge", "output": "output"})
    # select_knowledge → secure_diff (the safe-version comparison candidate gate)
    # → risk_forensics (view B independent forensics: reverse-grep of tainted
    # tokens, judging whether a guard covers it, proposing candidate facts) →
    # arbiter (deterministic aggregation, S1 evidence boundary, zero LLM) →
    # project_dataflow (cross-file recovery, a pass-through in file mode) →
    # output.
    g.add_conditional_edges("select_knowledge", _linear("secure_diff"),
                            {"secure_diff": "secure_diff", "output": "output"})
    g.add_conditional_edges("secure_diff", _linear("risk_forensics"),
                            {"risk_forensics": "risk_forensics", "output": "output"})
    g.add_conditional_edges("risk_forensics", _linear("arbiter"),
                            {"arbiter": "arbiter", "output": "output"})
    g.add_conditional_edges("arbiter", _linear("project_dataflow"),
                            {"project_dataflow": "project_dataflow", "output": "output"})
    g.add_conditional_edges("project_dataflow", _linear("output"),
                            {"output": "output"})

    g.add_edge("output", END)
    return g.compile()
