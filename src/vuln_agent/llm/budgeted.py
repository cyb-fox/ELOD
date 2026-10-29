"""BudgetedLLM: the single LLM entry point.

- Check token and call limits before every request and raise BudgetExceeded on overflow.
- Add actual usage after every request.
- Extract JSON robustly, tolerating fences, surrounding noise, and empty reasoning-model content.

Budget state lives in the external State.budget dict. This class keeps a reference
and updates it in place, so nodes do not have to write budget state back manually.

Three repeatability layers are centralized here: request-side seed and greedy
sampling, deterministic response replay through ``LLM_CACHE_DIR``, and visible
accounting for malformed JSON. A truncated response is retried with the global
token cap before a format-correction request is attempted.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .. import config
from ..state import Budget


class BudgetExceeded(Exception):
    """Token or call limit exceeded; nodes should route to output with partial=True."""


class LLMCallFailed(Exception):
    """A request still failed after bounded retries; nodes should degrade locally."""


# ---- Optional call trace (full I/O, token counts, and latency) ----
_TRACE_LOCK = threading.Lock()
_TRACE_ENABLED = False
_TRACE_FULL = False  # Whether to include full system/user/response text.
_TRACE_LOCAL = threading.local()


def _local_trace() -> list[dict]:
    records = getattr(_TRACE_LOCAL, "records", None)
    if records is None:
        records = []
        _TRACE_LOCAL.records = records
    return records


def trace_context() -> list[dict]:
    """The record list this thread appends to; hand it to worker threads.

    Trace state is per-thread so that concurrent bench files cannot mix their
    prompts. But node-internal fan-out (``parallel_map``) spawns worker threads
    *inside* one file, and a fresh thread starts with an empty list — every call
    a worker made used to be dropped on the floor, silently reporting a batched
    file as costing zero tokens. Workers therefore adopt their parent's list via
    ``use_trace_context``; appends are already under ``_TRACE_LOCK``.
    """
    return _local_trace()


def use_trace_context(records: list[dict]) -> None:
    """Append to ``records`` from this thread instead of a private list."""
    _TRACE_LOCAL.records = records


def enable_trace(full: bool = True) -> None:
    """Enable tracing and clear history; full=True also records prompt/response text."""
    global _TRACE_ENABLED, _TRACE_FULL
    _TRACE_ENABLED = True
    _TRACE_FULL = full
    clear_trace()


def clear_trace() -> None:
    """Clear only the current thread's records.

    Bench workers use this before each file so concurrent files cannot mix
    prompts or responses in one trace.
    """
    _TRACE_LOCAL.records = []


def get_trace() -> list[dict]:
    with _TRACE_LOCK:
        return list(_local_trace())


def _trace_add(rec: dict) -> None:
    with _TRACE_LOCK:
        _local_trace().append(rec)


# ---- Sampling parameters (repeatability layer 1) ----
# Remove provider-rejected sampling parameters from subsequent requests and retry;
# this costs repeatability but preserves detection.
_DROPPED: set[str] = set()
_DROP_LOCK = threading.Lock()


def _sampling_kwargs() -> dict:
    out: dict = {"temperature": config.LLM_TEMPERATURE}
    if config.LLM_SEED >= 0:
        out["seed"] = int(config.LLM_SEED)
    if str(config.LLM_TOP_P).strip():
        try:
            out["top_p"] = float(config.LLM_TOP_P)
        except ValueError:
            pass
    with _DROP_LOCK:
        return {k: v for k, v in out.items() if k not in _DROPPED}


def _is_upstream_flake(err: Exception) -> bool:
    """Tell a gateway routing miss from a real "no such model" 404.

    A gateway that fans one model name out to several upstream channels answers
    404 with ``type=upstream_error`` when it happens to pick a channel where the
    model is not published; the next attempt reaches a different channel. A wrong
    model id carries the provider's own error type and must stay fatal, so that a
    misconfigured run fails loudly instead of degrading to "found nothing".
    """
    if int(getattr(err, "status_code", 0) or 0) != 404:
        return False
    body = getattr(err, "body", None)
    detail = body.get("error") if isinstance(body, dict) else None
    if not isinstance(detail, dict):
        return False
    return "upstream" in str(detail.get("type") or "").lower()


def _drop_unsupported(err: Exception, kwargs: dict) -> bool:
    """Identify rejected sampling parameters in a 4xx error and remove them permanently.

    Prefer an exact parameter name; for a generic unsupported-parameter response,
    remove all optional knobs while keeping temperature greedy.
    """
    if int(getattr(err, "status_code", 0) or 0) not in (400, 404, 422):
        return False
    msg = str(getattr(err, "message", "") or err).lower()
    opt = [k for k in kwargs if k != "temperature"]
    hit = [k for k in opt if k in msg]
    if not hit and any(w in msg for w in ("unsupported", "unknown", "unrecognized",
                                          "not support", "invalid_request")):
        hit = opt
    if not hit:
        return False
    with _DROP_LOCK:
        _DROPPED.update(hit)
    return True


# ---- Response disk cache (repeatability layer 2) ----

def _cache_key(model: str, system: str, user: str, cap: int, extra: dict) -> str:
    h = hashlib.sha256()
    for part in (model, system, user, str(cap), json.dumps(extra, sort_keys=True)):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_path(key: str) -> Optional[str]:
    d = str(config.LLM_CACHE_DIR or "").strip()
    if not d:
        return None
    sub = os.path.join(d, key[:2])
    try:
        os.makedirs(sub, exist_ok=True)
    except OSError:
        return None
    return os.path.join(sub, key + ".json")


def _cache_read(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except Exception:                     # Treat corrupt cache data as a miss.
        return None
    return rec if isinstance(rec, dict) and "content" in rec else None


def _cache_write(path: Optional[str], rec: dict) -> None:
    if not path:
        return
    try:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, path)             # Atomic replace avoids partial files.
    except Exception:
        pass


def _escalated_cap(cap: int) -> Optional[int]:
    """Next token cap to try after a truncated response, or None if exhausted.

    A truncated response must never be re-requested at the same cap: the model
    would spend the same tokens and be cut at the same place. The ladder doubles
    the cap, never drops below the global default, and stops at
    ``LLM_TOKENS_HARD_CAP`` so a runaway reasoning trace cannot bill without
    bound.
    """
    ceiling = max(int(config.LLM_TOKENS_HARD_CAP or 0), int(config.LLM_MAX_TOKENS))
    nxt = min(ceiling, max(int(config.LLM_MAX_TOKENS), cap * 2))
    return nxt if nxt > cap else None


# ---- Process-wide egress gate ----
# Concurrency is a product of two nested pools: the bench/project file workers
# times ``SINK_MAX_WORKERS`` inside a node. The provider limits tokens per
# minute, so the only number that matters is total in-flight requests. Gating
# here keeps that number in one place instead of asking the operator to hand-tune
# ``--workers`` down until the 429s stop.
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: Optional[threading.Semaphore] = None
_INFLIGHT_LIMIT = 0


def _inflight_gate() -> Optional[threading.Semaphore]:
    """The shared semaphore, rebuilt if the configured limit changed."""
    global _INFLIGHT, _INFLIGHT_LIMIT
    limit = int(config.LLM_MAX_INFLIGHT or 0)
    if limit <= 0:
        return None
    with _INFLIGHT_LOCK:
        if _INFLIGHT is None or _INFLIGHT_LIMIT != limit:
            _INFLIGHT = threading.Semaphore(limit)
            _INFLIGHT_LIMIT = limit
        return _INFLIGHT


def _backoff(seconds: float) -> None:
    """Sleep with proportional jitter so retries do not re-converge.

    Without jitter, every thread caught by one rate-limit window retries in the
    same millisecond and reproduces the window it was waiting out.
    """
    jitter = max(0.0, float(config.LLM_BACKOFF_JITTER or 0.0))
    time.sleep(seconds * (1.0 + random.random() * jitter) if jitter else seconds)


class BudgetedLLM:
    def __init__(self, budget: Budget, tag: str = "", model: Optional[str] = None):
        self._budget = budget
        self._tag = tag  # Node name for trace grouping and model overrides.
        self._model = model or (config.model_for(tag) if tag else config.LLM_MODEL)
        self._lock = threading.Lock()  # Protect in-place budget updates.
        self._client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
            timeout=config.LLM_TIMEOUT,
        )

    # ---- Budget accounting ----
    def _preflight(self) -> None:
        with self._lock:
            b = self._budget
            if b["llm_calls"] >= b["calls_max"]:
                raise BudgetExceeded(
                    f"llm_calls {b['llm_calls']} >= {b['calls_max']}"
                )
            if b["tokens_used"] >= b["tokens_max"]:
                raise BudgetExceeded(
                    f"tokens_used {b['tokens_used']} >= {b['tokens_max']}"
                )

    def _account(self, usage: Any) -> None:
        with self._lock:
            self._budget["llm_calls"] += 1
            if usage is not None:
                self._budget["tokens_used"] += int(getattr(usage, "total_tokens", 0) or 0)

    def _account_tokens(self, total: int) -> None:
        """Account for a cache replay without a request, using cached usage."""
        with self._lock:
            self._budget["llm_calls"] += 1
            self._budget["tokens_used"] += int(total or 0)

    def _count_failed_call(self) -> None:
        with self._lock:
            self._budget["llm_calls"] += 1
            # Keep failed calls separate so evaluation can distinguish an offline
            # run with no output from a genuine negative.
            self._budget["failed_calls"] = self._budget.get("failed_calls", 0) + 1

    def _count_json_failure(self) -> None:
        """Count unparseable JSON, including responses that fail correction."""
        with self._lock:
            self._budget["json_failures"] = self._budget.get("json_failures", 0) + 1

    # ---- Raw text calls ----
    def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        return self.complete_ex(system, user, max_tokens=max_tokens)[0]

    def complete_ex(self, system: str, user: str,
                    max_tokens: Optional[int] = None) -> tuple[str, str]:
        """Like complete, but also return finish_reason for truncation handling."""
        self._preflight()
        cap = int(max_tokens) if max_tokens else config.LLM_MAX_TOKENS
        content, finish = self._call_with_retries(system, user, cap)
        # Reasoning models count reasoning and visible output toward max_tokens.
        # A small cap can be consumed before visible JSON is emitted, so retry once
        # at a raised cap to leave room for the response.
        if not content.strip() and finish == "length":
            higher = _escalated_cap(cap)
            if higher is not None:
                content, finish = self._call_with_retries(system, user, higher)
        return content, finish

    def _call_with_retries(self, system: str, user: str, cap: int) -> tuple[str, str]:
        """One call at a fixed cap with bounded network/rate-limit retries."""
        sampling = _sampling_kwargs()
        path = _cache_path(_cache_key(self._model, system, user, cap, sampling))
        hit = _cache_read(path)
        if hit is not None:
            self._account_tokens(int(hit.get("total_tokens", 0) or 0))
            content = str(hit.get("content") or "")
            finish = str(hit.get("finish") or "")
            if _TRACE_ENABLED:
                self._trace_replay(system, user, content, hit)
            return content, finish
        # Retry transient network/read timeouts and TPM rate limits with backoff.
        last_err: Optional[Exception] = None
        t0 = time.time()
        attempt = 0
        while attempt <= config.LLM_MAX_RETRIES:
            try:
                resp = self._create(system, user, cap, sampling)
                self._account(resp.usage)
                content = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason or ""
                _cache_write(path, {
                    "content": content, "finish": finish,
                    "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
                    "model": self._model, "tag": self._tag,
                })
                if _TRACE_ENABLED:
                    self._trace_ok(system, user, content, resp.usage, t0, attempt)
                return content, finish
            except RateLimitError as e:
                last_err = e
                if attempt < config.LLM_MAX_RETRIES:
                    _backoff(config.LLM_RATELIMIT_BACKOFF * (attempt + 1))
            except APIStatusError as e:
                # Retry in place without rejected seed/top_p. This is parameter
                # negotiation, not a failed request, so it does not consume retry budget.
                # Other status codes (401/5xx) are configuration/service failures
                # and must not be silently downgraded to an empty result.
                if sampling and _drop_unsupported(e, sampling):
                    sampling = _sampling_kwargs()
                    path = _cache_path(
                        _cache_key(self._model, system, user, cap, sampling))
                    continue
                # A gateway routing miss is transient: one bad draw would otherwise
                # abort a whole file after its earlier calls already succeeded.
                if not _is_upstream_flake(e):
                    raise
                last_err = e
                if attempt < config.LLM_MAX_RETRIES:
                    _backoff(2 * (attempt + 1))
            except (APITimeoutError, APIConnectionError) as e:
                last_err = e
                if attempt < config.LLM_MAX_RETRIES:
                    _backoff(2 * (attempt + 1))
            attempt += 1
        # Retries exhausted: account for one failed call and let the caller degrade.
        self._count_failed_call()
        if _TRACE_ENABLED:
            self._trace_fail(system, user, last_err, t0)
        raise LLMCallFailed(str(last_err))

    def _create(self, system: str, user: str, cap: int, sampling: dict):
        """The one place a request leaves the process, held under the egress gate.

        The gate is released as soon as the response returns, so backoff sleeps
        and JSON parsing do not occupy a slot.
        """
        gate = _inflight_gate()
        if gate is None:
            return self._request(system, user, cap, sampling)
        with gate:
            return self._request(system, user, cap, sampling)

    def _request(self, system: str, user: str, cap: int, sampling: dict):
        return self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=cap,
            **sampling,
        )

    # ---- Trace records ----
    def _trace_ok(self, system, user, content, usage, t0, attempt) -> None:
        rec = {
            "tag": self._tag,
            "ok": True,
            "attempt": attempt,
            "elapsed_s": round(time.time() - t0, 2),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        if _TRACE_FULL:
            rec["system"] = system
            rec["user"] = user
            rec["response"] = content
        _trace_add(rec)

    def _trace_fail(self, system, user, err, t0) -> None:
        rec = {
            "tag": self._tag,
            "ok": False,
            "elapsed_s": round(time.time() - t0, 2),
            "error": str(err),
        }
        if _TRACE_FULL:
            rec["system"] = system
            rec["user"] = user
        _trace_add(rec)

    def _trace_replay(self, system, user, content, rec) -> None:
        out = {
            "tag": self._tag,
            "ok": True,
            "cached": True,
            "elapsed_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": int(rec.get("total_tokens", 0) or 0),
        }
        if _TRACE_FULL:
            out["system"] = system
            out["user"] = user
            out["response"] = content
        _trace_add(out)

    def _attempt(self, system: str, user: str, cap: int) -> tuple[str, str]:
        """One budget-checked request at exactly ``cap``, with no cap escalation.

        ``complete_json`` owns its own cap ladder, so it must not go through
        ``complete_ex``: that method escalates internally, which would leave the
        caller's ``cap`` stale and buy a second request at a cap already tried.
        """
        self._preflight()
        return self._call_with_retries(system, user, cap)

    # ---- JSON calls (expand after truncation, then correct format once) ----
    def complete_json(
        self, system: str, user: str, retry: bool = True,
        max_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        """Return parsed JSON, or None after recording a json_failures count.

        ``retry`` controls one format-correction request. Truncated output is
        retried at a raised token cap because another request at the same cap
        would be truncated again — and once the cap ladder is exhausted, the
        format-correction request is skipped as well: re-asking at a cap that has
        already proven too small buys another truncation at full price.
        """
        cap = int(max_tokens) if max_tokens else config.LLM_MAX_TOKENS
        try:
            text, finish = self._attempt(system, user, cap)
        except LLMCallFailed:
            return None                       # Already counted by _count_failed_call.
        parsed = _extract_json(text)
        if parsed is not None:
            return parsed
        # 1) Truncated output: climb the cap ladder; this is a token-cap issue.
        while finish == "length":
            higher = _escalated_cap(cap)
            if higher is None:
                break
            cap = higher
            try:
                text, finish = self._attempt(system, user, cap)
            except (BudgetExceeded, LLMCallFailed):
                self._count_json_failure()
                return None
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
        # 2) Still truncated at the ceiling: a format-correction request would be
        # cut at the same place. Record the failure instead of paying for it.
        if finish == "length":
            self._count_json_failure()
            return None
        # 3) Non-truncated malformed output: one format-correction request.
        if not retry:
            self._count_json_failure()
            return None
        try:
            fix = self.complete(
                system,
                user + "\n\nThe previous response was not valid JSON. Return only valid JSON with no extra text.",
                max_tokens=cap,     # The cap the ladder settled on, not the original.
            )
        except (BudgetExceeded, LLMCallFailed):
            self._count_json_failure()
            return None
        parsed = _extract_json(fix)
        if parsed is None:
            self._count_json_failure()
        return parsed


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Optional[Any]:
    """Extract the first valid JSON value (object or array) from model output."""
    if not text:
        return None
    text = text.strip()
    # 1) Direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) Fenced code block.
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # 3) Scan for the first balanced object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    frag = text[start : i + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        break
    return None
