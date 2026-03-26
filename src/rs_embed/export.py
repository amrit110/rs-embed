"""High-level export entrypoints.

`export_npz` is a convenience wrapper around `rs_embed.api.export_batch`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from .core.specs import FetchSpec, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from .core.types import ExportConfig, ExportTarget


def export_npz(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    models: list[str],
    out_path: str,
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: SensorSpec | None = None,
    fetch: FetchSpec | None = None,
    per_model_sensors: dict[str, SensorSpec] | None = None,
    per_model_fetches: dict[str, FetchSpec] | None = None,
    config: ExportConfig = ExportConfig(),
) -> dict[str, Any]:
    """Export inputs + embeddings for one spatial query to a single `.npz`.

    The output format is always ``"npz"`` regardless of any ``config.format``
    value passed in; ``config`` controls all other runtime settings (workers,
    resume, show_progress, input_prep, etc.).
    """
    from .api import export_batch as _api_export_batch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not out_path.endswith(".npz"):
        out_path = out_path + ".npz"

    return _api_export_batch(
        spatials=[spatial],
        temporal=temporal,
        models=models,
        target=ExportTarget.combined(out_path),
        config=replace(config, format="npz"),
        backend=backend,
        device=device,
        output=output,
        sensor=sensor,
        fetch=fetch,
        per_model_sensors=per_model_sensors,
        per_model_fetches=per_model_fetches,
    )
