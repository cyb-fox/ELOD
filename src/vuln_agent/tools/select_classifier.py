"""select_classifier: inference wrapper for the local family classifier
(UniXcoder + Linear(768, 9)).

The default backend of `select_knowledge`. It has exactly one responsibility:
**read C/C++ source text and return the vulnerability family names that file
may hit**, which decides which family knowledge cards get loaded. It takes part
in no verdict: the adjudication chain cannot read any output of this module
(`knowledge` / `family_context` only ever enter a prompt).

Interface invariants:

1. **The 9 dimensions must be aligned by name, never by index.** In the shipped
   `metrics.json`, `families` is in **alphabetical** order (`access-control` at
   position 0), whereas `families.json` puts `memory-safety` first. The sets are
   equal but the orders differ, so an index-based mapping would not crash and
   would not raise -- it would **silently permute every prediction**.
   `_families_from_metrics` therefore trusts only the order in metrics and uses
   the registry purely as a **set** check; any inconsistency rejects the whole
   result (ok=False) rather than performing a possibly wrong mapping.

2. **Never raise.** A missing torch, a missing directory, a corrupt
   checkpoint, a shape mismatch, or an OOM all return `ok=False` plus a
   `reason`, following the same convention as `tools/cfg` returning None when it
   cannot build a graph. An unavailable classifier should send the family list
   to the rule floor / fallback, never bring down the scanning chain.

3. **Lazy load + explicit lock + negative cache.** The `--workers` option of
   bench.py forms an outer thread pool. Relying on
   `lru_cache` alone is unsafe because its miss path holds no lock, so two
   threads missing at once would each run `torch.load` (480MB x 2, roughly 2GB
   RSS at peak). Hence a `threading.Lock` with double-checked locking; a failed
   load records `_MODEL_FAIL` as a negative cache so the failure is not
   repeated once per file across 870 files. **The forward pass takes no lock**:
   `eval()` plus `no_grad()` mutates no state, so sharing one module across
   threads is correct; the real hazard is intra-op oversubscription, handled by
   setting `torch.set_num_threads` once at model construction.

4. **Truncation must match training.** `model_max_length` in
   `tokenizer_config.json` is 1e30, so without an explicit
   `truncation=True, max_length=512` nothing is truncated and a 5000-token
   input would run into the 1026-position table. 512 tokens is roughly 51 lines
   of numbered C, and 83.4% of training samples are right-truncated -- the
   model is in effect a classifier over the first ~50 lines of a file. The
   `truncated` flag makes this otherwise invisible fact visible (the node
   records a degraded marker) without changing behaviour.

5. **Pooling must replicate the training side**: masked mean pooling.
   RoBERTa's own `pooler` is **untrained** (`enc.pooler.dense.bias` is all
   zeros), so using it would amount to feeding a random projection as features.

6. **Known train/inference text skew** (accepted and recorded, not corrected
   here): `nodes/ingest.py` runs `canonicalize_source`, which folds
   continuation lines, whereas the training corpus was produced with a plain
   `read_text()` and no folding. Folding shifts tokens left and removes a
   newline, which at the 512-token boundary also changes which lines survive.
   Changing the inference side would desynchronise the released checkpoint from
   its annotations, so the current behaviour is retained.
"""
from __future__ import annotations

import functools
import json
import threading
from typing import Any, Optional, TypedDict

from .. import config
from ..knowledge import registry


class SelectResult(TypedDict):
    ok: bool
    families: list[str]
    probs: dict[str, float]
    threshold: float
    reason: str
    truncated: bool


# The model is built once: on success the (tokenizer, encoder, head) triple is
# cached, on failure the reason is cached (negative cache). One lock guards both.
_LOCK = threading.Lock()
_MODEL: Optional[tuple[Any, Any, Any]] = None   # (tokenizer, encoder, head)
_MODEL_FAIL: str = ""


def _empty(reason: str, truncated: bool = False) -> SelectResult:
    return SelectResult(ok=False, families=[], probs={},
                        threshold=_threshold(), reason=reason, truncated=truncated)


# ---------------------------------------------------------------- Artefact metadata
@functools.lru_cache(maxsize=1)
def _metrics() -> dict[str, Any]:
    """Read the shipped metrics.json; return {} when missing or malformed."""
    try:
        data = json.loads(
            (config.SELECT_CLF_DIR / "metrics.json").read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _families_from_metrics(metrics: dict[str, Any]) -> list[str]:
    """Family order recorded in metrics.json (= the head output dimension order).

    Any inconsistency returns [] (reject, never guess).

    Validation: 9 strings whose **set** equals registry.family_names(). The order
    comes from metrics (alphabetical); the registry is used as a set only, per
    invariant 1 of the module docstring.
    """
    fams = metrics.get("families")
    if not (isinstance(fams, list) and all(isinstance(f, str) for f in fams)):
        return []
    known = set(registry.family_names())
    if len(fams) != len(known) or set(fams) != known:
        return []
    return list(fams)


@functools.lru_cache(maxsize=1)
def families() -> list[str]:
    """The head output dimension order (length 9); [] when it cannot be confirmed."""
    return _families_from_metrics(_metrics())


@functools.lru_cache(maxsize=1)
def _threshold() -> float:
    """Decision threshold: the environment variable wins, otherwise selected_threshold
    from the shipped metrics.json is used.

    The frozen value is deliberately **not** copied into code, since a copy would
    drift silently when the artefacts are replaced. When neither source yields a
    value, 0.5 (the sigmoid midpoint) serves as the neutral fallback for
    "threshold unavailable", not as a tuned value.
    """
    env = (config.SELECT_CLF_THRESHOLD or "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    t = _metrics().get("selected_threshold")
    if isinstance(t, (int, float)) and 0.0 < float(t) < 1.0:
        return float(t)
    return 0.5


def available() -> tuple[bool, str]:
    """Cheap probe: load no model, only check artefact completeness and dimension order."""
    d = config.SELECT_CLF_DIR
    if not d.is_dir():
        return False, f"model directory does not exist: {d}"
    for name in ("config.json", "metrics.json", "model.pt", "tokenizer.json"):
        if not (d / name).is_file():
            return False, f"model directory is missing file: {name}"
    if not families():
        return False, ("families in metrics.json disagree with families.json; "
                       "refusing an index-based mapping")
    return True, ""


def clear_cache() -> None:
    """Test use only: reset the model cache and the lru_cache entries."""
    global _MODEL, _MODEL_FAIL
    with _LOCK:
        _MODEL = None
        _MODEL_FAIL = ""
    _metrics.cache_clear()
    families.cache_clear()
    _threshold.cache_clear()


# ---------------------------------------------------------------- Model construction
# (one of the only two places that touch torch)
def _build() -> tuple[Any, Any, Any]:
    """Build (tokenizer, encoder, head) offline. Failures raise here and _model
    converts them into a reason string."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    d = str(config.SELECT_CLF_DIR)
    # Intra-op thread count: an outer --workers pool already exists, so the
    # default of 1 prevents oversubscription.
    torch.set_num_threads(max(1, config.SELECT_CLF_TORCH_THREADS))

    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    cfg = AutoConfig.from_pretrained(d, local_files_only=True)
    # from_config initialises the skeleton randomly (no download); every weight
    # comes from model.pt.
    enc = AutoModel.from_config(cfg)
    head = torch.nn.Linear(cfg.hidden_size, len(families()))

    # Every storage in the checkpoint is tagged cuda:0, so map_location is
    # required rather than defensive.
    sd = torch.load(config.SELECT_CLF_DIR / "model.pt", map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"model.pt is not a state_dict: {type(sd).__name__}")
    enc_sd = {k[len("enc."):]: v for k, v in sd.items() if k.startswith("enc.")}
    head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    if not enc_sd or not head_sd:
        raise ValueError("model.pt carries no weights with an enc.* or head.* prefix")

    res = enc.load_state_dict(enc_sd, strict=False)
    if list(res.unexpected_keys):
        raise ValueError(f"checkpoint has unexpected keys: {list(res.unexpected_keys)[:5]}")
    # position_ids is a persistent buffer in transformers 4.11 but not in
    # later releases; it is derived from arange, so its absence is harmless. Any
    # other missing key means the checkpoint is truncated and must be rejected.
    missing = [k for k in res.missing_keys if not k.endswith("position_ids")]
    if missing:
        raise ValueError(f"checkpoint is missing weights: {missing[:5]}")

    head.load_state_dict(head_sd, strict=True)   # exactly two tensors: weight and bias
    enc.eval()
    head.eval()
    return tok, enc, head


def _model() -> tuple[Optional[tuple[Any, Any, Any]], str]:
    """Fetch the model lazily: double-checked lock plus negative cache.

    Returns (model triple | None, reason).
    """
    global _MODEL, _MODEL_FAIL
    if _MODEL is not None:
        return _MODEL, ""
    if _MODEL_FAIL:
        return None, _MODEL_FAIL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, ""
        if _MODEL_FAIL:
            return None, _MODEL_FAIL
        try:
            _MODEL = _build()
            return _MODEL, ""
        except Exception as e:                       # noqa: BLE001 -- see invariant 2
            _MODEL_FAIL = f"{type(e).__name__}: {e}"
            return None, _MODEL_FAIL


# ---------------------------------------------------------------- Forward pass
# (the other of the only two places that touch torch)
def _score(text: str) -> tuple[list[float], bool]:
    """One forward pass, returning (9 sigmoid probabilities, whether right-truncated).

    Masked mean pooling replicates the training-side Clf.forward; RoBERTa's own
    pooler is **not** used because it is untrained.
    """
    import torch

    bundle, _ = _model()
    assert bundle is not None                       # the caller already confirmed availability
    tok, enc, head = bundle
    max_len = max(8, config.SELECT_CLF_MAX_LEN)
    # model_max_length is 1e30, so truncation arguments must be passed
    # explicitly. type_vocab_size=10 is a UniXcoder trait; only the fields the
    # tokenizer returns are used and token_type_ids is never synthesised.
    batch = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    with torch.no_grad():
        out = enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        probs = torch.sigmoid(head(pooled))[0]
    truncated = int(batch["attention_mask"].sum()) >= max_len
    return [float(x) for x in probs], truncated


# ---------------------------------------------------------------- The single public entry point
def select_families(text: str) -> SelectResult:
    """Given C/C++ source text without line numbers, return the family names
    that clear the threshold. **Never raises.**

    `probs` carries all 9 values, so an offline threshold sweep need not rerun
    inference and "all low" can be told apart from "one high". **probs never
    enters a prompt**: feeding model confidence to the downstream LLM is
    exactly the "hypothesis treated as conclusion" that gate I7 blocks.
    """
    if not text or not text.strip():
        return _empty("source is empty")
    ok, why = available()
    if not ok:
        return _empty(why)
    bundle, why = _model()
    if bundle is None:
        return _empty(why)

    try:
        probs, truncated = _score(text)
    except Exception as e:                          # noqa: BLE001
        return _empty(f"forward failed: {type(e).__name__}: {e}")

    fams = families()
    if len(probs) != len(fams):
        return _empty(f"head output dimension {len(probs)} != family count {len(fams)}", truncated)

    prob_map = {f: p for f, p in zip(fams, probs)}
    thr = _threshold()
    picked = [f for f in fams if prob_map[f] >= thr]
    # Nothing cleared the threshold -> take the MIN_FAMILIES highest-probability
    # entries; the family list must never be empty.
    if not picked and config.SELECT_CLF_MIN_FAMILIES > 0:
        picked = [f for f, _ in sorted(prob_map.items(), key=lambda kv: -kv[1])
                  ][: config.SELECT_CLF_MIN_FAMILIES]
    # Too many clear the threshold -> routing degenerates into no routing. This
    # is an escape hatch, unlimited by default; when truncating, keep the top-N
    # by probability.
    cap = config.SELECT_CLF_MAX_FAMILIES
    if cap > 0 and len(picked) > cap:
        keep = set(sorted(picked, key=lambda f: -prob_map[f])[:cap])
        picked = [f for f in picked if f in keep]

    return SelectResult(ok=True, families=picked, probs=prob_map,
                        threshold=thr, reason="", truncated=truncated)
