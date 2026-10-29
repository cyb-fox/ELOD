"""Central configuration: LLM connection, budget ceilings, paths, loop bounds.

Every value is read from the environment first and falls back to the built-in
default. Credentials are never hard-coded; inject them through the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- LLM API ----
LLM_API_PROVIDER = os.getenv("LLM_API_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # never hard-code; inject via environment
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

# Per-node model overrides: by default every node uses LLM_MODEL. Set the
# VULN_MODEL_<TAG> environment variable to override a single node.
_NODE_MODEL: dict[str, str] = {}


def model_for(tag: str) -> str:
    """Resolve the model for one node.

    Precedence: the VULN_MODEL_<TAG> environment variable, then the
    _NODE_MODEL table, then LLM_MODEL. To pin a node to a different model, add
    an entry to _NODE_MODEL or set the environment variable.
    """
    return os.getenv(f"VULN_MODEL_{tag.upper()}", _NODE_MODEL.get(tag, LLM_MODEL))

# Sampling: adjudication must be stable, so the temperature is pinned to 0.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# Reproducibility: temperature 0 only guarantees greedy sampling. On a MoE
# endpoint, expert routing and in-batch reduction order still perturb the
# logits, so the same prompt can yield different JSON across runs. A fixed seed
# is the only request-side control over that residual variation (whether the
# endpoint honours it is implementation-defined; gateways that reject the field
# drop it silently, see llm/budgeted). A value < 0 omits the seed entirely.
LLM_SEED = int(os.getenv("LLM_SEED", "20260728"))
# top_p is not sent by default: it is meaningless under greedy sampling and some
# gateways reject it. Set an explicit value such as "1" to send one.
LLM_TOP_P = os.getenv("LLM_TOP_P", "")
# On-disk response cache directory. Disabled by default (empty string). When
# enabled, a request whose (model, system, user, temperature, max_tokens, seed)
# tuple matches a cached entry replays the stored response and is billed from
# the cached usage, so re-running one command reproduces tokens and calls
# byte-for-byte. This is the only mechanism that *guarantees* identical
# results; a seed only makes them likely. Disabled by default because silently
# replaying a stored response can make it look as though an edited prompt took
# effect when it did not.
LLM_CACHE_DIR = os.getenv("VULN_LLM_CACHE", "")
# Hard cap on a single completion. Must be large enough to hold the body text:
# models that spend reasoning tokens count them against this cap, and a
# truncated response yields empty content.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))
# Output ceiling for generative nodes (safe-reference generation): sized to
# hold a complete multi-risk-line diff plus rationale as one JSON document.
# A cap below what full generation requires truncates first and then makes
# complete_json re-send at LLM_MAX_TOKENS, which is both slower and bills the
# prompt twice. The default therefore matches LLM_MAX_TOKENS; lower it via
# GEN_MAX_TOKENS for models that need a smaller allowance.
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "8000"))
# Ceiling for truncation recovery. When complete_json sees finish="length" it
# raises the cap to min(LLM_TOKENS_HARD_CAP, max(LLM_MAX_TOKENS, 2*cap)) and
# sends once more. This is a separate knob because when GEN_MAX_TOKENS equals
# LLM_MAX_TOKENS the test `cap < LLM_MAX_TOKENS` is never true, so recovery
# would never engage; set it to 0 or any value <= LLM_MAX_TOKENS to disable.
LLM_TOKENS_HARD_CAP = int(os.getenv("LLM_TOKENS_HARD_CAP", "16000"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
# Bounded retries for network/timeout failures (excluding the first attempt).
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Base backoff in seconds for TPM rate limiting (HTTP 429). Rate limiting needs
# a longer wait than a timeout, so it gets its own base value.
LLM_RATELIMIT_BACKOFF = float(os.getenv("LLM_RATELIMIT_BACKOFF", "15"))
# Backoff jitter ratio in [0, 1], multiplied into the backoff duration. With no
# jitter, N threads hit by one rate-limit response retry in the same
# millisecond and reproduce the limit verbatim. Set 0 for deterministic backoff.
LLM_BACKOFF_JITTER = float(os.getenv("LLM_BACKOFF_JITTER", "0.3"))
# Process-level cap on in-flight requests (an egress gate). Concurrency is the
# product of two nested layers: bench/project --workers multiplied by
# SINK_MAX_WORKERS inside a node, so up to N x 6 requests can be in flight while
# the gateway enforces TPM. This gate fixes the real ceiling in one place: a
# node may raise its own concurrency without changing the true limit. Set 0 to
# disable and fall back to the product constraint alone.
LLM_MAX_INFLIGHT = int(os.getenv("LLM_MAX_INFLIGHT", "6"))

# ---- Budget control (unified inside LangGraph; short-circuits to partial) ----
# Each sink spends at least one call in each of the baseline/forensics/arbiter
# views, so the allowance scales with the number of sinks.
BUDGET_TOKENS_MAX = int(os.getenv("BUDGET_TOKENS_MAX", "3000000"))
BUDGET_CALLS_MAX = int(os.getenv("BUDGET_CALLS_MAX", "400"))

# ---- Upper bound on top-level families used by the all-empty fallback ----
# In select_knowledge: when LLM parsing fails and the rule floor is empty,
# take the first N families and the first K members of each. This guarantees
# that knowledge never drops to zero.
MAX_FAMILIES = int(os.getenv("MAX_FAMILIES", "8"))

# ---- Oversized-function threshold ----
FUNC_FULL_MAX_LINES = int(os.getenv("FUNC_FULL_MAX_LINES", "200"))
# Line ceiling for sending a whole file to the LLM. Beyond it, send a
# structured digest instead (signatures + control-flow skeleton + the
# neighbourhoods of already-recalled lines) rather than the full text.
# Declaration-dense headers benefit most: a 6994-line abi.h carries its body
# at the top level, where function-level compression has no effect. Set 0 to
# disable and always send the whole file.
SOURCE_MAX_LINES = int(os.getenv("VULN_SOURCE_MAX_LINES", "1200"))
# Neighbourhood window, in lines, kept above and below a sink hit in the
# compressed representation.
SINK_NEIGHBORHOOD = int(os.getenv("SINK_NEIGHBORHOOD", "3"))
# Context window of risk lines handed to the LLM by secure_diff / forensics.
RISK_WINDOW = int(os.getenv("RISK_WINDOW", "6"))

# ---- context_search (grep loop) bounds; prevent divergence ----
CTX_MAX_VARS = int(os.getenv("CTX_MAX_VARS", "5"))
CTX_MAX_ROUNDS = int(os.getenv("CTX_MAX_ROUNDS", "3"))

# ---- Concurrency: per-sink view and adjudication calls use a thread pool ----
# With batching, each node produces roughly 4 batches, so a concurrency of 6
# lets most files finish in a single round (ceil(4/6) = 1), which removes the
# serial rounds inside a node -- the largest speed lever available without
# changing the node structure. The ceiling is bounded by the gateway's TPM
# rate limit: lower it to 3-4 if HTTP 429 appears in the logs.
SINK_MAX_WORKERS = int(os.getenv("SINK_MAX_WORKERS", "6"))

# ---- Upper bound on sinks per batch ----
# forensics sends one call per batch (a batch may span several small functions),
# which amortises the fixed SYSTEM + CWE schema cost from once per function to
# once per batch. The per-batch sink count both caps the output array length
# (preventing attention dilution and output truncation) and merges
# single-sink functions. Functions stay atomic: sinks of one function are never
# split across batches, and a function above the limit takes a whole batch.
SINK_BATCH_MAX = int(os.getenv("SINK_BATCH_MAX", "8"))

# ---- Adjudication confidence tiers (used by the deterministic arbiter
# aggregation) ----
# The verdict itself is already deterministic (whether the checker discharged
# an obligation); confidence only expresses the *strength of the evidence
# source*. It feeds the file-level threshold gate in output and the offline
# threshold sweep, and does not enter the is_vuln decision.
#   WITNESS: a four-step machine-checked witness on the CFG (surface statistics
#            cannot fabricate one) -> strongest;
#   MICRO  : deterministic syntactic-level abstract interpretation
#            (micro_check) -> next;
#   ANCHOR : a verbatim LLM anchor confirmed by re-checking -> weakest; a lone
#            anchor falls below the threshold gate.
CONF_WITNESS = float(os.getenv("CONF_WITNESS", "0.9"))
CONF_MICRO = float(os.getenv("CONF_MICRO", "0.7"))
CONF_ANCHOR = float(os.getenv("CONF_ANCHOR", "0.6"))

# ---- Semantic micro-check: the deterministic proposer of obligation_checker.
# Performs syntactic-level abstract interpretation for size-arithmetic overflow
# on the risk-line slice (no build, no LLM). A hit yields a candidate fact that
# the same checker then re-verifies. It deliberately covers only
# multiply/add/shift with no overflow guard -- the usually asymmetric CWE-190
# shape -- and skips truncation casts, which are identical on both patch sides
# and would produce symmetric false positives. Set 0 to disable.
MICRO_CHECK_ENABLED = os.getenv("MICRO_CHECK_ENABLED", "1") not in ("0", "false", "")

# Zero-sink recall floor: turn anchored mechanism hypotheses into limited
# candidates only when ordinary discovery found no sink. The checker remains
# the sole authority for a vulnerability verdict. Set to 0 only for an
# ablation/control run.
MECHANISM_PROVISIONAL_SINKS = os.getenv(
    "MECHANISM_PROVISIONAL_SINKS", "1"
) not in ("0", "false", "")
MECHANISM_PROVISIONAL_SINKS_MAX = int(
    os.getenv("MECHANISM_PROVISIONAL_SINKS_MAX", "2")
)

# ---- Source canonicalisation: before ingest, fold statements that span
# physical lines (continuation lines with unclosed brackets) back onto the
# opening line. The line count is preserved and continuation lines are left
# blank, which removes purely cosmetic patch churn -- a patch splitting one
# line into three -- from influencing the verdict. Set 0 to disable and use the
# raw source.
CANONICALIZE_SOURCE = os.getenv("CANONICALIZE_SOURCE", "1") not in ("0", "false", "")

# ---- Evidence-boundary switches ----
# These expose the internal gates of the evidence boundary as switches so that
# each one can be disabled independently while everything else stays at its
# default. They are the knobs the component ablation is built from: set one to 0
# to measure that mechanism's marginal contribution to conclusions and to
# underdetermination on the frozen configuration. Each switch maps one-to-one onto
# a gate in obligation_checker (I5/I6/I7) or onto one of the two discharge
# channels:
#   ANCHOR_INDEPENDENCE_ENABLED     I5 anchor-independence gate
#                                   (off = an anchor may attest itself)
#   EXTERNAL_PREMISE_ENABLED        I6 external-source decidability gate
#                                   (off = an external return value may
#                                   decide positive)
#   EQUIVALENT_PROTECTION_ENABLED   I7 equivalent protection (assumption
#                                   invalidated: an equivalent guard in the
#                                   original code does not count)
#   ANCHOR_CHANNEL_ENABLED          anchor channel (off = anchor facts never
#                                   discharge, VIOLATED and SATISFIED are
#                                   capped alike, including anchor facts
#                                   from micro_check; witness only)
#   WITNESS_CHANNEL_ENABLED         witness channel (off = anchors + micro
#                                   only; candidate facts carrying a witness
#                                   fail outright and do not fall back to
#                                   the anchor)
ANCHOR_INDEPENDENCE_ENABLED = (
    os.getenv("ANCHOR_INDEPENDENCE_ENABLED", "1") not in ("0", "false", "")
)
EXTERNAL_PREMISE_ENABLED = (
    os.getenv("EXTERNAL_PREMISE_ENABLED", "1") not in ("0", "false", "")
)
EQUIVALENT_PROTECTION_ENABLED = (
    os.getenv("EQUIVALENT_PROTECTION_ENABLED", "1") not in ("0", "false", "")
)
ANCHOR_CHANNEL_ENABLED = (
    os.getenv("ANCHOR_CHANNEL_ENABLED", "1") not in ("0", "false", "")
)
WITNESS_CHANNEL_ENABLED = (
    os.getenv("WITNESS_CHANNEL_ENABLED", "1") not in ("0", "false", "")
)

# Counterfactual secure-reference gate: disable to turn off reference guidance.
# The rule-based sink floor remains; the ideal safe implementation no longer
# pre-seeds a VIOLATED assumption. Enabled by default.
SECURE_REFERENCE_ENABLED = (
    os.getenv("SECURE_REFERENCE_ENABLED", "1") not in ("0", "false", "")
)

# ---- File-level threshold gate ----
# At aggregation time in output, a finding counts towards "the file is
# vulnerable" only when its confidence reaches this value. A pure OR over
# findings would let a single low-confidence suspicion decide the file; the
# gate keeps such borderline points out of the final verdict. The recall cost
# is controlled, since a true positive usually reaches 0.6-0.9 after the
# three views plus critique. Set 0 to disable and fall back to a pure OR.
VULN_CONF_MIN = float(os.getenv("VULN_CONF_MIN", "0.5"))

# ---- Knowledge base path (reuses the shipped cwe_agent_kb) ----
_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("VULN_KB_DIR", str(_ROOT / "data" / "cwe_agent_kb")))

# ---- select_knowledge backend: classification head (default) / LLM ----
# This switch changes **three** things at once, so it is not a simple router
# A/B comparison:
#   (1) who selects the families -- the local classification head (zero token)
#       or one whole-file LLM call;
#   (2) whether a suspicions channel exists -- on the head path there is no
#       positional hypothesis, so `_triage_hypotheses` is always empty and the
#       `_provisional_sinks` zero-sink bridge in secure_diff is inert;
#   (3) whether CWE-level signals reach the downstream prompt -- the head path
#       loads family cards only, while the LLM path additionally receives
#       detection_signals, negative_signals, and a safe example.
# Because switching the backend changes recall and underdetermined_share, check
# the value of this variable before comparing numbers across the two paths.
SELECT_KNOWLEDGE_BACKEND = os.getenv("SELECT_KNOWLEDGE_BACKEND", "clf")   # clf | llm | none
SELECT_CLF_DIR = Path(os.getenv("SELECT_CLF_DIR", str(_ROOT / "models" / "clf_select")))
# Empty = read selected_threshold from the shipped metrics.json rather than
# copying the frozen value into code, where it could drift silently.
SELECT_CLF_THRESHOLD = os.getenv("SELECT_CLF_THRESHOLD", "")
SELECT_CLF_MAX_LEN = int(os.getenv("SELECT_CLF_MAX_LEN", "512"))          # must match training
SELECT_CLF_MIN_FAMILIES = int(os.getenv("SELECT_CLF_MIN_FAMILIES", "1"))  # none over threshold -> take top-N
SELECT_CLF_MAX_FAMILIES = int(os.getenv("SELECT_CLF_MAX_FAMILIES", "0"))  # 0 = unlimited
# torch intra-op thread count: bench --workers is the outer pool, so the
# default of 1 prevents intra-op oversubscription.
SELECT_CLF_TORCH_THREADS = int(os.getenv("SELECT_CLF_TORCH_THREADS", "1"))

# ---- Run profile: controls the cost / recall trade-off ----
# bench (default): the evaluation profile. No pre-screening anywhere in the
#   chain, protecting the recall floor (used for PrimeVul and sliced sets).
# scan: the deployment profile for real projects. Enables a cheap file-level
#   pre-screen that decides with pure rules and zero tokens whether a file
#   contains no suspicious operation at all, and short-circuits the whole LLM
#   chain if so. This trades recall for cost and only blocks files with no
#   danger indicator whatsoever; a true positive almost always carries one, so
#   the miss risk is small.
# Set VULN_PROFILE=scan or pass profile= to run(...) to override.
PROFILE = os.getenv("VULN_PROFILE", "bench")


def prescreen_enabled(profile: str | None = None) -> bool:
    """Whether the cheap file-level pre-screen runs: scan profile only.

    The bench profile protects evaluation recall and never pre-screens.
    """
    return (profile or PROFILE) == "scan"


# ---- Core-module filter (scan profile) ----
# In real projects, test/testdata/example/fuzz/generated code is rarely where
# vulnerabilities live, yet it is often the largest and slowest part. The scan
# profile therefore scans core modules and excludes these non-core files from
# the verdict set, while still indexing all of them in project_index so
# context_search can traverse across files on demand -- scan the core first,
# then follow the relevant variables. The bench profile does not filter, so the
# evaluation setting is unchanged.
SCAN_SKIP_NONCORE = os.getenv("VULN_SCAN_SKIP_NONCORE", "1") not in ("0", "false", "")
# Non-core path segments (case-insensitive, matched per / segment so that a
# substring never misfires).
NONCORE_DIR_SEGMENTS = {
    "test", "tests", "testing", "testdata", "__tests__", "unittest", "unittests",
    "example", "examples", "sample", "samples", "demo", "demos",
    "fuzz", "fuzzing", "fuzzers", "benchmark", "benchmarks",
    "mock", "mocks", "doc", "docs", "third_party", "vendor",
}
# Non-core filename patterns (generated code and test naming conventions):
# *_test/_unittest suffixes, mock_ prefixes, .pb.* generated protobuf, and
# .generated. artifacts. A search hit is enough (not anchored), case-insensitive.
import re as _re
_NONCORE_FILE_RE = _re.compile(
    r"(_test|_unittest|_tests|_mock|_mocks|_benchmark|_fuzz|_fuzzer)\.[^.]+$"
    r"|^(test_|mock_)"
    r"|\.pb\.(h|cc|cpp)$"
    r"|\.generated\.",
    _re.IGNORECASE,
)


def is_noncore_path(rel_path: str) -> bool:
    """Whether a relative path is non-core (test/testdata/example/generated code).

    Directory names are matched per / segment so a substring never misfires
    ('testbed' does not match 'test'), after which the naming-convention regex
    runs on the filename. The scan profile uses this to drop such files from
    the verdict set while keeping them available as context.
    """
    parts = rel_path.replace("\\", "/").split("/")
    for seg in parts[:-1]:  # directory segments
        if seg.lower() in NONCORE_DIR_SEGMENTS:
            return True
    name = parts[-1] if parts else rel_path
    return bool(_NONCORE_FILE_RE.search(name))
