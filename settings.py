"""Shared dashboard and MCP configuration."""
import json
import os
from pathlib import Path

DEFAULT = {"provider": "openai", "model": "text-embedding-3-small", "api_key": "", "llm_provider": "openai", "llm_model": "gpt-4o-mini", "search_mode": "semantic", "search_limit": 6}


def path():
    return Path(os.getenv("VECTORVAULT_DATA_DIR", "data/vectorvault")) / "settings.json"


def load():
    settings = DEFAULT.copy()
    if path().exists():
        settings.update(json.loads(path().read_text(encoding="utf-8")))
    return settings


def save(provider, model, api_key=None):
    if provider not in ("openai", "ollama") or not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise ValueError("Choose a provider and enter a valid model name")
    settings = load()
    settings.update(provider=provider, model=model.strip())
    if api_key is not None:
        if not isinstance(api_key, str) or len(api_key) > 500:
            raise ValueError("Invalid API key")
        settings["api_key"] = api_key.strip()
    return write(settings)


def save_llm(provider, model):
    if provider not in ("openai", "ollama") or not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise ValueError("Choose a provider and enter a valid model name")
    config = load()
    config.update(llm_provider=provider, llm_model=model.strip())
    return write(config)


def save_server(mode, limit):
    if mode not in ("semantic", "hybrid"):
        raise ValueError("Search mode must be semantic or hybrid")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError("Search limit must be a whole number between 1 and 100")
    if not 1 <= limit <= 100:
        raise ValueError("Search limit must be between 1 and 100")
    config = load()
    config.update(search_mode=mode, search_limit=limit)
    return write(config)


def write(settings):
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings), encoding="utf-8")
    os.replace(temporary, target)
    return settings


def key(settings):
    return settings.get("api_key") or os.getenv("OPENAI_API_KEY", "")
