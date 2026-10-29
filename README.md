# ELOD

Replication package for **ELOD**, an evidence-gated LLM-assisted vulnerability
detection approach for C/C++.

It contains the implementation, the frozen prediction logs, the per-file
reasoning traces, the human audit records, and a self-test that recomputes the
main table from the shipped predictions. The snapshot referenced by the paper is
tag `v1.0`.

## Layout

**Entry point**

| Path | Content |
| :--- | :--- |
| `run.sh` | `selftest` · `env` · `check` · `analyze` |

**Implementation — `src/`**

| Path | Content |
| :--- | :--- |
| `vuln_agent/run.py` | CLI entry point |
| `vuln_agent/graph.py` · `state.py` · `config.py` | pipeline graph, state, budgets and paths |
| `vuln_agent/nodes/` | pipeline stages |
| `vuln_agent/tools/` | deterministic verifier and its analysers |
| `vuln_agent/knowledge/` · `llm/` | CWE family registry, budgeted LLM client |
| `vuln_agent/bench.py` · `project.py` · `metrics.py` | file-mode and project-mode drivers, scorer |
| `data/cwe_agent_kb/` | 96 CWE cards + family index |
| `models/clf_select/` | family classifier config, tokenizer and metrics (weights not shipped) |

**Results — `results/`**

| Path | Content |
| :--- | :--- |
| `predictions/` | frozen per-file predictions |
| `logs/` | per-file reasoning traces |
| `audit/` | blinded audit packets + reviewer tables |
| `statistics/` | frozen aggregates |
| `baselines/` | direct-prompting control outputs |
| `datasets/` | project snapshots |

## Quick start

```bash
./run.sh selftest      # recompute the main table from the shipped predictions
./run.sh env           # print the resolved configuration
./run.sh check         # grep-based check of the structural property
./run.sh analyze FILE  # run the detector on one C/C++ file (needs LLM_API_KEY)
```

`selftest` needs only a bare Python 3 — no packages, no endpoint, no dataset:

```
file level N=870  TP=224 FP=120 TN=315 FN=211
  accuracy=0.6195  precision=0.6512  recall=0.5149  f1=0.5751  fpr=0.2759
Pair count=435
  P-C=151 P-V=73 P-B=164 P-R=47  pair_accuracy=0.3471
```

Running the detector itself needs an OpenAI-compatible endpoint, supplied
through the environment; no credential is stored in this repository.

```bash
export LLM_API_KEY=...
export LLM_API_BASE=...   # optional if the provider default fits
export LLM_MODEL=...      # optional
```

The CFG-based stages additionally need
`pip install langgraph tree_sitter_languages openai requests`.
