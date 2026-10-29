"""CWE JSON discovery, whitelist management, and two-level family loading.

Design:
- The whitelist contains agent-ready CWEs with both detection_signals and
  negative_signals. select_knowledge only chooses and loads whitelisted items.
- Loading is top-down within one select_knowledge call:
  top_family_menu lists the nine families; family_member_menu lists member CWEs;
  cwe_family maps a CWE back to its family for deterministic recall fallback.
- load_cwe reads one CWE JSON file and caches it with lru_cache.
- data/cwe_agent_kb/families.json is the authoritative family index.
- family_card returns family-level knowledge (principle / trigger / examples /
  patch / safe_pattern) distilled offline into family_cards.json. It is LLM
  reference material only; no deterministic component reads it.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Optional

from .. import config


@functools.lru_cache(maxsize=None)
def _kb_index() -> dict[str, Path]:
    """Return a cwe_id -> file path index."""
    idx: dict[str, Path] = {}
    for p in sorted(config.KB_DIR.glob("CWE-*.json")):
        idx[p.stem] = p
    return idx


@functools.lru_cache(maxsize=None)
def load_cwe(cwe_id: str) -> Optional[dict[str, Any]]:
    """Read one CWE JSON file; return None when missing or invalid."""
    p = _kb_index().get(cwe_id)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def whitelist() -> list[str]:
    """Return agent-ready CWEs with detection_signals and negative_signals."""
    out = []
    for cwe_id in _kb_index():
        d = load_cwe(cwe_id)
        if d and "detection_signals" in d and "negative_signals" in d:
            out.append(cwe_id)
    return sorted(out, key=lambda c: int(c.split("-")[1]))


# A few KB entries lack a name (for example CWE-20); use a readable menu label.
_FAMILY_NAME_FALLBACK: dict[str, str] = {"CWE-20": "Improper Input Validation"}


@functools.lru_cache(maxsize=1)
def _load_families() -> dict[str, Any]:
    """Read families.json, returning an empty dict on failure."""
    p = config.KB_DIR / "families.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fams = data.get("families")
        return fams if isinstance(fams, dict) else {}
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def top_family_menu() -> list[str]:
    """Return the nine top-level families as ``<family> - <description>`` menu items."""
    out = []
    for fam, info in _load_families().items():
        desc = (info or {}).get("description", "") if isinstance(info, dict) else ""
        out.append(f"{fam} — {desc}" if desc else fam)
    return out


def family_names() -> list[str]:
    """Return top-level family names for whitelist validation and fallback."""
    return list(_load_families().keys())


def family_member_menu(family: str) -> list[tuple[str, str]]:
    """Return whitelisted named members of a family, preserving index order."""
    wl = set(whitelist())
    info = _load_families().get(family) or {}
    members = info.get("members", []) if isinstance(info, dict) else []
    out: list[tuple[str, str]] = []
    for cwe_id in members:
        if cwe_id not in wl:
            continue
        d = load_cwe(cwe_id) or {}
        name = d.get("name") or _FAMILY_NAME_FALLBACK.get(cwe_id, "")
        out.append((cwe_id, name))
    return out


def family_description(family: str) -> str:
    """Return one family's one-line description ("" when unknown)."""
    info = _load_families().get(family) or {}
    desc = info.get("description", "") if isinstance(info, dict) else ""
    return desc if isinstance(desc, str) else ""


@functools.lru_cache(maxsize=1)
def _load_family_cards() -> dict[str, Any]:
    """Read family_cards.json's ``cards`` mapping, {} on any failure.

    The file is an offline distillation asset under data/ (gitignored), so a
    fresh clone will not have it. Absence is a normal state, not a degradation:
    family_card then falls back to description-only.
    """
    p = config.KB_DIR / "family_cards.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cards = data.get("cards")
        return cards if isinstance(cards, dict) else {}
    except Exception:
        return {}


# Card fields fed to prompts, in emission order. `family` is added by
# family_card itself; `probs`-like model internals never appear here.
_CARD_FIELDS = ("principle", "trigger", "examples", "patch", "safe_pattern")


def family_card(family: str) -> dict[str, Any]:
    """Return one family's knowledge card for prompt assembly.

    Always returns at least ``{"family": ..., "description": ...}``. When
    family_cards.json is present, the distilled fields are merged in; empty or
    malformed fields are dropped so the prompt never emits an empty section
    (the "section exists iff data exists" rule).
    """
    card: dict[str, Any] = {"family": family}
    desc = family_description(family)
    if desc:
        card["description"] = desc
    raw = _load_family_cards().get(family)
    if not isinstance(raw, dict):
        return card
    for key in _CARD_FIELDS:
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            card[key] = val.strip()
        elif isinstance(val, list) and val:
            card[key] = val
    return card


def cwe_family(cwe_id: str) -> Optional[str]:
    """Map a CWE to its top-level family for deterministic recall fallback."""
    d = load_cwe(cwe_id) or {}
    fam = (d.get("family") or {}).get("project_family")
    return fam if isinstance(fam, str) and fam else None
