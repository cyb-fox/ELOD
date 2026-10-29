# ELOD

Replication package for **ELOD**, an evidence-gated LLM-assisted vulnerability
detection approach for C/C++.

It contains the implementation, the frozen prediction logs, the per-file
reasoning traces, the human audit records, and a self-test that recomputes the
main table from the shipped predictions. The snapshot referenced by the paper is
tag `v1.0`.

## Layout

```
run.sh                      entry point (selftest / env / check / analyze)
src/vuln_agent/             implementation
  run.py                    CLI entry point
  config.py  state.py  graph.py
  nodes/                    pipeline stages
  tools/                    deterministic verifier and its analysers
  knowledge/  llm/          CWE family registry, budgeted LLM client
  bench.py  project.py      file-mode and project-mode drivers
  metrics.py                scorer behind the reported numbers
src/data/cwe_agent_kb/      96 CWE cards + family index
src/models/clf_select/      family classifier: config, tokenizer, metrics
results/predictions/        frozen per-file predictions
results/logs/               per-file reasoning traces
results/audit/              blinded audit packets + reviewer tables
results/statistics/         frozen aggregates
results/baselines/          direct-prompting control outputs
results/datasets/           project snapshots
```

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

## Notes

- `src/models/clf_select/model.pt` (≈ 500 MB) is not shipped.
- The alias → file mapping behind `results/audit/` is not shipped.
- Code is MIT (see `LICENSE`); the released data files are CC BY 4.0. Source
  files under `results/datasets/` stay under their original projects' licenses.
