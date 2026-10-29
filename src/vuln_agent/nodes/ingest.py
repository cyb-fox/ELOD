"""ingest (deterministic): language detection + function splitting + feeding mode
+ cheap file-level prescreen for the scan profile."""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..state import VulnState
from ..tools import canonicalize as cz
from ..tools import prescreen as ps
from ..tools import segmenter as sg


def ingest(state: VulnState) -> dict:
    path = state["target_path"]
    lang = sg.detect_lang(path)
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    # (3) Source canonicalization (line count unchanged): folds statement
    # continuations that span physical lines, removing the purely cosmetic
    # prompt perturbation that patch reordering introduces.  Done before
    # function splitting so that the whole chain (function ranges / risk
    # windows / anchoring / report) uses one coordinate system.
    if lang in ("c", "cpp") and config.CANONICALIZE_SOURCE:
        source = cz.canonicalize_source(source)

    functions = sg.split_functions(source, lang)
    # First round has no sinks, so compress by signature + control flow only.
    units = [sg.prepare_unit(source, f) for f in functions]

    out = {
        "source_text": source,
        "functions": functions,
        "units": units,
        "errors": [],
    }

    # scan profile: cheap file-level prescreen (pure rules, zero token).  With
    # no danger indication at all -> mark for short-circuit, and graph routes
    # straight to output, skipping the whole LLM chain.  The bench profile does
    # not prescreen (protects evaluation recall).
    if config.prescreen_enabled(state.get("profile")):
        is_clean, reasons = ps.prescreen_file(source, functions)
        out["_prescreen_clean"] = is_clean
        out["_prescreen_reasons"] = reasons

    return out
