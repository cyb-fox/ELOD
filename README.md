# ELOD

Evidence-gated LLM-assisted vulnerability detection for C/C++ — replication
package accompanying the manuscript submitted to *ACM Transactions on Software
Engineering and Methodology* (TOSEM).

This repository is the artifact snapshot referenced by the paper as **tag
`v1.0`**. Cite the commit or the tag, not `main`.

The package contains the implementation, the frozen prediction logs, the  
per-file reasoning traces, the human audit records, and a self-test that  
recomputes the reported RQ1 table **with no API key, no network access, and no  
datasets**.

---

## 1. Quick start

```bash
./run.sh selftest     # recompute the RQ1 table from the shipped prediction log
./run.sh env          # print the resolved configuration
./run.sh check        # verify the structural claim by grep (no API key)
./run.sh analyze FILE # run the detector on one C/C++ file (needs LLM_API_KEY)
```

`selftest` runs on a bare Python 3 install. It needs no third-party package, no  
model endpoint, and no PrimeVul copy. It should print:

```
file level N=870  TP=224 FP=120 TN=315 FN=211
  accuracy=0.6195  precision=0.6512  recall=0.5149  f1=0.5751  fpr=0.2759
Pair count=435
  P-C=151 P-V=73 P-B=164 P-R=47  pair_accuracy=0.3471
```

Those are the numbers reported in the paper. If your run differs, the shipped  
log and the shipped scorer are what changed, not the expectation.

To run the detector itself, supply an OpenAI-compatible endpoint through the  
environment; no credential is stored in this package:

```bash
export LLM_API_KEY=...
export LLM_API_BASE=...   # optional if the provider default fits
export LLM_MODEL=...      # optional
```

The CFG-based stages additionally require  
`pip install langgraph tree_sitter_languages openai requests`.

---

## 2. What is in the package

```
ELOD/                             (repository root, tag v1.0)
├── README.md
├── LICENSE
├── run.sh                        entry point (selftest / env / check / analyze)
├── src/
│   ├── vuln_agent/               the implementation (38 modules)
│   │   ├── run.py                CLI: the single entry point
│   │   ├── config.py             connection, budgets, paths, loop bounds
│   │   ├── state.py  graph.py    state machine and orchestration
│   │   ├── nodes/                ten pipeline stages
│   │   ├── tools/                the deterministic verifier and its analysers
│   │   ├── knowledge/            family registry
│   │   ├── llm/                  budgeted client and node prompts
│   │   ├── bench.py  project.py  file-mode and project-mode drivers
│   │   └── metrics.py            the scorer behind every reported number
│   ├── data/cwe_agent_kb/        96 CWE cards + 9-family index
│   └── models/clf_select/        family classifier: config, tokenizer, metrics
└── results/
    ├── predictions/              rq1_primevul.jsonl (870 records)
    ├── logs/
    │   ├── primevul_traces/      868 per-file reasoning traces
    │   └── trace_209801/         one verbatim trace
    ├── audit/log_audit_kit/      870 blinded packets + reviewer A/B tables
    ├── statistics/               frozen aggregates and the RQ4 ablation table
    ├── baselines/primevul-det/   direct-prompting control on the RQ1 inputs
    └── datasets/RQ3_postcut_proj/ 48 project snapshots (24 cases, two sides)
```

### The three study inputs

| Study | Case set                               | Size                    | Where                                    |
| ----- | -------------------------------------- | ----------------------- | ---------------------------------------- |
| RQ1   | PrimeVul test set (paired criterion)   | 435 pairs / 870 files   | `results/predictions/rq1_primevul.jsonl` |
| RQ2   | PrimeVul reasoning logs (expert audit) | 870 reasoning logs      | `results/logs/primevul_traces/`          |
| RQ3   | PostCut-Proj project queue             | 24 cases / 48 snapshots | `results/datasets/RQ3_postcut_proj/`     |

`results/statistics/` holds the frozen aggregates those three produce, together  
with `rq4_ablation_stats.json` — the five-arm component ablation over the full  
435-pair queue that the RQ4 table reports — and the threshold sweep behind the  
file-level gate. Every arm in that file is stated on the same pair grid as RQ1,  
and the `full` arm is byte-identical to the main table.

`results/baselines/primevul-det/` holds the per-sample outputs of the  
direct-prompting control on the RQ1 inputs, together with its scoring map, so  
that control's numbers can be recomputed from the records here.

---

## 3. The structural claim, and how to check it

The paper's central claim is a separation of authority: the proposer may  
propose, but only the deterministic verifier may change a safety-precondition  
state, and the final verdict is a function of the evidence summary alone.  
`./run.sh check` verifies the corresponding structural properties by grep over  
the shipped tree:

```
writes to a precondition state    -> obligation_checker.py only
state writes outside the checker -> (none expected)
per-CWE branches in the checker  -> one section marker, not per-CWE logic
```

The absence of per-CWE branches is the point: the checker discharges  
*obligations* — anchor independence, external-source decidability, equivalent  
protection — rather than pattern-matching CWE numbers, which is why it can  
generalise beyond the families the knowledge base enumerates.

---

## 4. The family classifier

`src/models/clf_select/` holds the configuration, tokenizer, and metrics of the  
9-output family classifier over a frozen UniXcoder encoder. The weight file  
(`model.pt`, roughly 500 MB) is **not** included in this repository and will be
released with the camera-ready artifact. Everything needed to reproduce it is here:
`config.json` gives the encoder architecture, `metrics.json` records the  
training arguments, the selected decision threshold, and the family list in  
output-dimension order, and `src/data/cwe_agent_kb/` contains the label space.  
The classifier only chooses which family knowledge cards to load — the  
adjudication chain never reads its output.

---

## 5. The human audit

`results/audit/log_audit_kit/` is the complete audit record for RQ2: 870 blinded
packets (`packets/L-0001.txt` … `L-0870.txt`), each holding one file's reasoning
trajectory under a random alias, plus the two reviewers' independent tables
(`reviewer_A_table.csv`, `reviewer_B_table.csv`), the per-rater labelling sheets
(`打标表_评审A.csv`, `打标表_评审B.csv`), and the coding protocol
(`RUBRIC_判定口径.md`, `README.md`). The alias→file mapping is **not** shipped:
`packets/` and `文件清单.txt` list aliases only, so a packet reveals neither the
file identity, nor which side of a pair the trace came from, nor the ground-truth
label.

The audit asks three questions per trace: whether the verdict is correct  
(`correct` / `too_conservative` / `too_aggressive`), how far the reasoning is  
endorsed (1–5), and whether the cited evidence grounds that step (`yes` / `no`).

Reported RQ2 figures are computed on the subset where both reviewers assigned  
the same verdict. Records where they disagree are reported as raw counts and  
are not resolved by any rule or by a third reviewer.

---

## 6. What is *not* shipped

| Omitted | Why | Where it is described |
| ------- | --- | --------------------- |
| `model.pt` (family classifier, ≈500 MB) | size; released with the camera-ready artifact | §4 above |
| The alias→file decryption key for `packets/` | it would undo the blinding of §5 | not released |
| PrimeVul raw corpus | third-party dataset, obtainable from its own release | `results/predictions/` ships the per-file records instead |

Nothing else is withheld: every number in the paper is recomputable from the
files here, and `./run.sh selftest` is the check that this is true for RQ1.

---

## 7. License and third-party material

The **code** in `src/` and the driver scripts are released under the MIT License
(see `LICENSE`). The **released data** — prediction logs, reasoning traces,
frozen statistics, audit records — are released under CC BY 4.0.

The source files under `results/datasets/RQ3_postcut_proj/` and any C/C++ excerpt
inside a reasoning trace or audit packet remain under the licenses of their
original projects; they are redistributed here only as the frozen inputs of the
experiments reported in the paper.

---

## 8. Version

| Tag | Date | Note |
| --- | ---- | ---- |
| `v1.0` | 2026-10-29 | artifact snapshot referenced by the TOSEM submission |
