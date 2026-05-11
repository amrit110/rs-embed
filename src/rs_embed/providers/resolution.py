"""Provider backend resolution and lifecycle management."""

from __future__ import annotations

import os
from typing import Any

from ..core.errors import ModelError
from . import get_provider, has_provider, list_providers
from .base import ProviderBase


def _norm(backend: str) -> str:
    return str(backend).strip().lower()


def default_provider_backend_name() -> str | None:
    configured = _norm(os.environ.get("RS_EMBED_DEFAULT_PROVIDER", ""))
    if configured:
        return configured if has_provider(configured) else None
    providers = list_providers()
    if not providers:
        return None
    if "gee" in providers:
        return "gee"
    return str(providers[0]).strip().lower()


def resolve_provider_backend_name(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: str | None = None,
) -> str | None:
    b = _norm(backend)
    if allow_auto and b == "auto":
        resolved_auto = (
            _norm(auto_backend) if auto_backend is not None else default_provider_backend_name()
        )
        if not resolved_auto:
            return None
        b = resolved_auto
    if has_provider(b):
        return b
    return None


def is_provider_backend(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: str | None = None,
) -> bool:
    return (
        resolve_provider_backend_name(
            backend,
            allow_auto=allow_auto,
            auto_backend=auto_backend,
        )
        is not None
    )


def provider_init_kwargs(backend: str) -> dict[str, Any]:
    b = _norm(backend)
    if b == "gee":
        return {"auto_auth": True}
    return {}


def get_cached_provider(
    provider_cache: dict[str, ProviderBase],
    *,
    backend: str,
    allow_auto: bool = True,
    auto_backend: str | None = None,
) -> ProviderBase:
    b = resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    )
    if b is None:
        raise ModelError(f"Unsupported provider backend={backend!r}.")
    p = provider_cache.get(b)
    if p is None:
        p = get_provider(b, **provider_init_kwargs(b))
        provider_cache[b] = p
    p.ensure_ready()
    return p


def create_provider_for_backend(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: str | None = None,
) -> ProviderBase:
    b = resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    )
    if b is None:
        raise ModelError(f"Unsupported provider backend={backend!r}.")
    p = get_provider(b, **provider_init_kwargs(b))
    p.ensure_ready()
    return p
