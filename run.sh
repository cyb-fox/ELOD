#!/usr/bin/env bash
# ELOD — code package entry point.
#
#   ./run.sh selftest        recompute the RQ1 table from the shipped log
#                            (no API key, no network, no datasets)
#   ./run.sh env             show the resolved configuration (no API key needed)
#   ./run.sh analyze FILE    run the detector on one C/C++ source file  (file mode)
#   ./run.sh analyze DIR     run it over a project tree                (project mode)
#
# analyze needs an OpenAI-compatible endpoint, supplied only through the
# environment (no credential is stored in this package):
#     export LLM_API_KEY=...
#     export LLM_API_BASE=...     # optional if the provider default fits
#     export LLM_MODEL=...        # optional

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

cmd="${1:-help}"
shift || true

hr() { printf '%s\n' "------------------------------------------------------------"; }

case "$cmd" in
  selftest)
    # Recompute the reported RQ1 numbers from the shipped prediction log using
    # the shipped scorer. Requires nothing beyond a bare Python install: no API
    # key, no network, no PrimeVul, and none of the optional dependencies.
    LOG="$HERE/results/predictions/rq1_primevul.jsonl"
    if [ ! -f "$LOG" ]; then
      echo "missing $LOG" >&2
      exit 2
    fi
    hr; echo "recomputing the RQ1 table from the shipped log"; hr
    ( cd "$HERE/src" && "$PY" -m vuln_agent.metrics --pred "$LOG" --no-strata )
    hr; echo "expected: P-C=151 P-V=73 P-B=164 P-R=47 | TP=224 FP=120 TN=315 FN=211"
    hr; echo "          accuracy=0.6195 precision=0.6512 recall=0.5149 f1=0.5751"
    ;;

  analyze)
    if [ $# -lt 1 ]; then
      echo "usage: ./run.sh analyze <file-or-directory> [extra args]" >&2
      exit 2
    fi
    if [ -z "${LLM_API_KEY:-}" ]; then
      echo "LLM_API_KEY is not set." >&2
      echo "export LLM_API_KEY=...  (and LLM_API_BASE, LLM_MODEL if not the default)" >&2
      exit 2
    fi
    hr; echo "analyze: $1"; hr
    cd "$HERE/src"
    exec "$PY" -m vuln_agent.run "$@"
    ;;

  env)
    hr; echo "resolved configuration"; hr
    ( cd "$HERE/src" && "$PY" - <<'PYEOF'
import vuln_agent.config as c
for k in sorted(dir(c)):
    if k.isupper():
        print(f"  {k} = {getattr(c, k)}")
PYEOF
    ) 2>&1 | head -60
    ;;

  check)
    # The structural claim of README §3, verified by grep on the shipped tree.
    cd "$HERE/src"
    hr; echo "writes to a precondition state  (all six must be in the checker)"; hr
    grep -rnE '\["status"\] *=' vuln_agent/
    hr; echo "writes outside the checker      (expect no output)"; hr
    if grep -rnE '\["status"\] *=' vuln_agent/ | grep -v obligation_checker.py; then
      echo "UNEXPECTED: a state write lives outside the checker" >&2
    else
      echo "(none)"
    fi
    hr; echo "per-CWE branches in the checker (expect one section marker)"; hr
    grep -rniE 'CWE-[0-9]+' vuln_agent/tools/obligation_checker.py || echo "(none)"
    hr; echo "the positive-verdict predicate"; hr
    grep -rn 'def is_vuln_provable' vuln_agent/
    ;;

  help | *)
    cat <<'TXT'
ELOD — code package

  ./run.sh selftest        recompute the RQ1 table from the shipped log
                           (no API key, no network, no datasets)
  ./run.sh analyze FILE   run the detector on one source file   (needs LLM_API_KEY)
  ./run.sh analyze DIR    run it over a project tree, project mode
  ./run.sh env            print the resolved configuration
  ./run.sh check          verify the structural claim (grep, no API key)

  analyze needs an OpenAI-compatible endpoint:
      export LLM_API_KEY=...
      export LLM_API_BASE=...      # optional if the provider's default fits
      export LLM_MODEL=...         # optional

  CFG-based stages additionally require:
      pip install langgraph tree_sitter_languages openai requests
TXT
    ;;
esac
