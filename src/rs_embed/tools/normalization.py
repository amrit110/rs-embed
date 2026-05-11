from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..embedders.catalog import canonical_model_id
from ..providers import has_provider


def normalize_model_name(model: str) -> str:
    """Resolve a model name or alias to its canonical model identifier.

    Parameters
    ----------
    model : str
        Model name or registered alias.

    Returns
    -------
    str
        Canonical model identifier as registered in the catalog.
    """
    return canonical_model_id(model)


def normalize_backend_name(backend: str) -> str:
    """Normalize a backend name to lowercase stripped form.

    Parameters
    ----------
    backend : str
        Raw backend name (e.g. ``"GEE"``, ``" auto "``).

    Returns
    -------
    str
        Lowercase, whitespace-stripped backend name.
    """
    return str(backend).strip().lower()


def normalize_device_name(device: str | None) -> str:
    """Normalize a device string, defaulting to ``"auto"`` when absent.

    Parameters
    ----------
    device : str or None
        Raw device string (e.g. ``"CUDA"``, ``"mps"``, ``None``).

    Returns
    -------
    str
        Lowercase, whitespace-stripped device name, or ``"auto"`` when
        *device* is ``None`` or empty.
    """
    if device is None:
        return "auto"
    dev = str(device).strip().lower()
    return dev or "auto"


def normalize_input_chw(
    x_chw: np.ndarray,
    *,
    expected_channels: int | None = None,
    name: str = "input_chw",
) -> np.ndarray:
    """Validate and cast a CHW input array to float32.

    Parameters
    ----------
    x_chw : np.ndarray
        Input array expected to have shape ``(C, H, W)``.
    expected_channels : int or None
        If provided, raises if the channel dimension does not match.
    name : str
        Label used in error messages (default ``"input_chw"``).

    Returns
    -------
    np.ndarray
        Float32 array with shape ``(C, H, W)``.

    Raises
    ------
    ModelError
        If the array is not 3-D or has the wrong number of channels.
    """
    x = np.asarray(x_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(f"{name} must be CHW with ndim=3, got shape={getattr(x, 'shape', None)}")
    if expected_channels is not None and int(x.shape[0]) != int(expected_channels):
        raise ModelError(
            f"{name} channel mismatch: got C={int(x.shape[0])}, expected C={int(expected_channels)}"
        )
    return x


def normalize_input_array(
    x_input: np.ndarray,
    *,
    expected_channels: int | None = None,
    name: str = "input",
) -> np.ndarray:
    """Validate and cast a CHW or TCHW input array to float32.

    Parameters
    ----------
    x_input : np.ndarray
        Input array expected to have shape ``(C, H, W)`` or
        ``(T, C, H, W)``.
    expected_channels : int or None
        If provided, raises if the channel dimension does not match.
    name : str
        Label used in error messages (default ``"input"``).

    Returns
    -------
    np.ndarray
        Float32 array with shape ``(C, H, W)`` or ``(T, C, H, W)``.

    Raises
    ------
    ModelError
        If the array is not 3-D or 4-D, or has the wrong number of channels.
    """
    x = np.asarray(x_input, dtype=np.float32)
    if x.ndim == 3:
        c = int(x.shape[0])
    elif x.ndim == 4:
        c = int(x.shape[1])
    else:
        raise ModelError(
            f"{name} must be CHW or TCHW with ndim=3/4, got shape={getattr(x, 'shape', None)}"
        )
    if expected_channels is not None and c != int(expected_channels):
        raise ModelError(f"{name} channel mismatch: got C={c}, expected C={int(expected_channels)}")
    return x


def _probe_model_describe(model_n: str) -> dict[str, Any]:
    """Best-effort model describe() probe used for API-level routing decisions."""
    try:
        cls = get_embedder_cls(model_n)
        emb = cls()
        desc = emb.describe() or {}
        return desc if isinstance(desc, dict) else {}
    except Exception as _e:
        return {}


def _default_provider_backend_for_api() -> str:
    from ..providers.resolution import default_provider_backend_name

    return default_provider_backend_name() or "gee"


def _resolve_embedding_api_backend(model_n: str, backend_n: str) -> str:
    """Normalize backend semantics for precomputed models."""
    desc = _probe_model_describe(model_n)
    if str(desc.get("type", "")).strip().lower() != "precomputed":
        return backend_n

    backends = desc.get("backend")
    if not isinstance(backends, list):
        return backend_n
    allowed = [str(b).strip().lower() for b in backends if str(b).strip()]
    if not allowed:
        return backend_n

    provider_allowed = ("provider" in allowed) or ("gee" in allowed)
    if backend_n in allowed:
        if backend_n == "auto" and provider_allowed:
            return _default_provider_backend_for_api()
        return backend_n
    if backend_n == "local" and "auto" in allowed and not provider_allowed:
        return "auto"
    if has_provider(backend_n) and provider_allowed:
        return backend_n

    if backend_n in {"gee", "auto"}:
        if "auto" in allowed:
            return "auto"
        if "local" in allowed:
            return "local"
        if provider_allowed:
            return _default_provider_backend_for_api()

    return backend_n


def coerce_input_to_tchw(
    input_chw: np.ndarray,
    *,
    expected_channels: int,
    n_frames: int,
    model_name: str,
) -> np.ndarray:
    """Normalize user-provided CHW/TCHW into clipped float32 [T,C,H,W]."""
    raw = np.asarray(input_chw, dtype=np.float32)
    t = max(1, int(n_frames))

    if raw.ndim == 3:
        if int(raw.shape[0]) != int(expected_channels):
            raise ModelError(
                f"input_chw must be CHW with C={int(expected_channels)} for {model_name}, "
                f"got {tuple(int(v) for v in raw.shape)}"
            )
        raw_tchw = np.repeat(raw[None, ...], repeats=t, axis=0).astype(np.float32)
    elif raw.ndim == 4:
        if int(raw.shape[1]) != int(expected_channels):
            raise ModelError(
                f"input_chw must be TCHW with C={int(expected_channels)} for {model_name}, "
                f"got {tuple(int(v) for v in raw.shape)}"
            )
        raw_tchw = raw.astype(np.float32, copy=False)
        if raw_tchw.shape[0] < t:
            raw_tchw = np.concatenate(
                [raw_tchw] + [raw_tchw[-1:]] * (t - raw_tchw.shape[0]),
                axis=0,
            )
        elif raw_tchw.shape[0] > t:
            raw_tchw = raw_tchw[:t]
    else:
        raise ModelError(
            f"input_chw must be CHW (C,H,W) or TCHW (T,C,H,W) for {model_name}, "
            f"got {tuple(int(v) for v in raw.shape)}"
        )

    raw_tchw = np.nan_to_num(raw_tchw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(raw_tchw, 0.0, 10000.0).astype(np.float32)


def coerce_single_input_chw(
    input_chw: Any,
    *,
    expected_channels: int | None,
    model_name: str,
) -> np.ndarray:
    """Normalize one user-provided tensor input into float32 CHW."""
    raw = input_chw
    try:
        import torch

        if torch.is_tensor(raw):
            raw = raw.detach().cpu().numpy()
    except Exception as _e:
        pass

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 4:
        raise ModelError(
            f"{model_name} expects single-sample input_chw as CHW (C,H,W), "
            f"got {tuple(int(v) for v in arr.shape)}. "
            "Use get_embeddings_batch_from_inputs(...) for batches."
        )
    if arr.ndim != 3:
        raise ModelError(
            f"{model_name} expects input_chw as CHW (C,H,W), got {tuple(int(v) for v in arr.shape)}"
        )
    if expected_channels is not None and int(arr.shape[0]) != int(expected_channels):
        raise ModelError(
            f"input_chw must be CHW with C={int(expected_channels)} for {model_name}, "
            f"got {tuple(int(v) for v in arr.shape)}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
