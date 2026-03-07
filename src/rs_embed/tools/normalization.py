from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import ModelError
from ..embedders.catalog import canonical_model_id


def normalize_model_name(model: str) -> str:
    return canonical_model_id(model)


def normalize_backend_name(backend: str) -> str:
    return str(backend).strip().lower()


def normalize_device_name(device: Optional[str]) -> str:
    if device is None:
        return "auto"
    dev = str(device).strip().lower()
    return dev or "auto"


def normalize_input_chw(
    x_chw: np.ndarray,
    *,
    expected_channels: Optional[int] = None,
    name: str = "input_chw",
) -> np.ndarray:
    x = np.asarray(x_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(
            f"{name} must be CHW with ndim=3, got shape={getattr(x, 'shape', None)}"
        )
    if expected_channels is not None and int(x.shape[0]) != int(expected_channels):
        raise ModelError(
            f"{name} channel mismatch: got C={int(x.shape[0])}, expected C={int(expected_channels)}"
        )
    return x
