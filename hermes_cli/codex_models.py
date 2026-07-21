"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import os

logger = logging.getLogger(__name__)

DEFAULT_CODEX_MODELS: List[str] = [
    # GPT-5.6 series (Sol/Terra/Luna) — verified live on ChatGPT Codex
    # 2026-07-20. The -pro variants are advertised by the model catalog but
    # rejected by the ChatGPT-backed Codex completion route for this account.
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    # NOTE: gpt-5.3-codex / gpt-5.3-codex-spark / gpt-5.2-codex /
    # gpt-5.1-codex-max / gpt-5.1-codex-mini were previously listed here,
    # but the chatgpt.com Codex backend returns HTTP 400 "The '<model>' model
    # is not supported when using Codex with a ChatGPT account." for these
    # slugs on ChatGPT accounts we've tested. Keeping them in fallback lists
    # leaks dead slugs into /model and crashes on selection.
]

_UNSUPPORTED_CHATGPT_CODEX_MODELS: frozenset[str] = frozenset({
    # The live /backend-api/codex/models endpoint can advertise these even
    # though the chat completion route rejects them for ChatGPT accounts with
    # HTTP 400 "not supported when using Codex with a ChatGPT account".
    "gpt-5.6-sol-pro",
    "gpt-5.6-terra-pro",
    "gpt-5.6-luna-pro",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
})


def _filter_supported_chatgpt_codex_models(model_ids: List[str]) -> List[str]:
    """Remove Codex slugs known to be rejected by ChatGPT-backed Codex."""
    filtered: List[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        slug = (model_id or "").strip()
        if not slug:
            continue
        if slug.lower() in _UNSUPPORTED_CHATGPT_CODEX_MODELS:
            continue
        if slug not in seen:
            filtered.append(slug)
            seen.add(slug)
    return filtered


_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [
    ("gpt-5.6-sol", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-terra", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-luna", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.5", ("gpt-5.4", "gpt-5.4-mini")),
]



def _add_forward_compat_models(model_ids: List[str]) -> List[str]:
    """Add Clawdbot-style synthetic forward-compat Codex models.

    If a newer Codex slug isn't returned by live discovery, surface it when an
    older compatible template model is present. This mirrors Clawdbot's
    synthetic catalog / forward-compat behavior for GPT-5 Codex variants.
    """
    ordered: List[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)

    for synthetic_model, template_models in _FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model in seen:
            continue
        if any(template in seen for template in template_models):
            ordered.append(synthetic_model)
            seen.add(synthetic_model)

    return _filter_supported_chatgpt_codex_models(ordered)


def _fetch_models_from_api(access_token: str) -> List[str]:
    """Fetch available models from the Codex API. Returns visible models sorted by priority."""
    try:
        import httpx
        resp = httpx.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        entries = data.get("models", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.debug("Failed to fetch Codex models from API: %s", exc)
        return []

    sortable = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        # Codex CLI's catalog uses ``supported_in_api`` for the public OpenAI
        # API, not for the OAuth-backed Codex backend that this provider uses.
        # Some valid Codex CLI models (for example gpt-5.3-codex-spark) are
        # marked false here but are still accepted by the Codex route.
        visibility = item.get("visibility", "")
        if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        sortable.append((rank, slug))

    sortable.sort(key=lambda x: (x[0], x[1]))
    return _add_forward_compat_models(_filter_supported_chatgpt_codex_models([slug for _, slug in sortable]))


def _read_default_model(codex_home: Path) -> Optional[str]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib
    except Exception:
        return None
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _read_cache_models(codex_home: Path) -> List[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    sortable = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug.strip():
                continue
            slug = slug.strip()
            # Do not filter on ``supported_in_api`` here.  It describes the
            # public OpenAI API, while Hermes openai-codex talks to the same
            # OAuth-backed Codex backend as Codex CLI.
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
                continue
            priority = item.get("priority")
            rank = int(priority) if isinstance(priority, (int, float)) else 10_000
            sortable.append((rank, slug))

    sortable.sort(key=lambda item: (item[0], item[1]))
    deduped: List[str] = []
    for _, slug in sortable:
        if slug not in deduped:
            deduped.append(slug)
    return _filter_supported_chatgpt_codex_models(deduped)


def get_codex_model_ids(access_token: Optional[str] = None) -> List[str]:
    """Return available Codex model IDs, trying API first, then local sources.
    
    Resolution order: API (live, if token provided) > config.toml default >
    local cache > hardcoded defaults.
    """
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()
    ordered: List[str] = []

    # Try live API if we have a token
    if access_token:
        api_models = _fetch_models_from_api(access_token)
        if api_models:
            # The live Codex model endpoint can omit older-but-still-working
            # ChatGPT-backed slugs (verified: gpt-5.4 and gpt-5.4-mini) while
            # advertising newer slugs that the completion route rejects. Treat
            # the endpoint as discovery input, not the sole allowlist.
            merged = _filter_supported_chatgpt_codex_models(api_models)
            for model_id in DEFAULT_CODEX_MODELS:
                if model_id not in merged:
                    merged.append(model_id)
            return _add_forward_compat_models(merged)

    # Fall back to local sources
    default_model = _read_default_model(codex_home)
    if default_model:
        ordered.append(default_model)

    for model_id in _read_cache_models(codex_home):
        if model_id not in ordered:
            ordered.append(model_id)

    for model_id in DEFAULT_CODEX_MODELS:
        if model_id not in ordered:
            ordered.append(model_id)

    return _add_forward_compat_models(ordered)
