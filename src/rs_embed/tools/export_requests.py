from __future__ import annotations

import os

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import FetchSpec, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import (
    ExportConfig,
    ExportLayout,
    ExportModelRequest,
    ExportTarget,
    ModelConfig,
)
from ..core.validation import assert_supported
from .checkpoint_utils import is_incomplete_combined_manifest
from .manifest import combined_resume_manifest, load_json_dict
from .model_defaults import resolve_sensor_for_model
from .normalization import _resolve_embedding_api_backend, normalize_model_name
from .runtime import require_model_config_support


def normalize_export_format(format_name: str) -> tuple[str, str]:
    fmt = str(format_name).strip().lower()
    from ..writers import SUPPORTED_FORMATS, get_extension

    if fmt not in SUPPORTED_FORMATS:
        raise ModelError(
            f"Unsupported export format: {format_name!r}. Supported: {SUPPORTED_FORMATS}."
        )
    return fmt, get_extension(fmt)


def normalize_export_target(
    *,
    n_spatials: int,
    ext: str,
    target: ExportTarget,
) -> ExportTarget:
    if not isinstance(target, ExportTarget):
        raise ModelError("target must be an ExportTarget instance.")
    if target.layout == ExportLayout.COMBINED:
        if not target.out_file:
            raise ModelError("ExportTarget.COMBINED requires out_file.")
        out_file = target.out_file if target.out_file.endswith(ext) else (target.out_file + ext)
        return ExportTarget.combined(out_file)
    if target.layout == ExportLayout.PER_ITEM:
        if not target.out_dir:
            raise ModelError("ExportTarget.PER_ITEM requires out_dir.")
        point_names = (
            target.names if target.names is not None else [f"p{i:05d}" for i in range(n_spatials)]
        )
        if len(point_names) != n_spatials:
            raise ModelError("target.names must have the same length as spatials.")
        return ExportTarget.per_item(target.out_dir, names=point_names)
    raise ModelError(f"Unsupported ExportTarget layout: {target.layout!r}.")



def resolve_export_model_configs(
    *,
    models: list[str | ExportModelRequest],
    backend_n: str,
    temporal: TemporalSpec | None,
    output: OutputSpec,
    sensor: SensorSpec | None,
    fetch: FetchSpec | None,
    modality: str | None,
    per_model_sensors: dict[str, SensorSpec] | None,
    per_model_fetches: dict[str, FetchSpec] | None,
    per_model_modalities: dict[str, str] | None,
) -> tuple[list[ModelConfig], dict[str, str]]:
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty list[str] or list[ExportModelRequest].")

    per_model_sensors = per_model_sensors or {}
    per_model_fetches = per_model_fetches or {}
    per_model_modalities = per_model_modalities or {}

    requests: list[ExportModelRequest] = []
    for item in models:
        if isinstance(item, ExportModelRequest):
            requests.append(item)
        elif isinstance(item, str):
            requests.append(ExportModelRequest(name=item))
        else:
            raise ModelError("models entries must be strings or ExportModelRequest instances.")

    model_configs: list[ModelConfig] = []
    resolved_backend: dict[str, str] = {}
    for req in requests:
        model_name = req.name
        model_n = normalize_model_name(model_name)
        eff_backend = _resolve_embedding_api_backend(model_n, backend_n)
        resolved_backend[model_name] = eff_backend
        cls = get_embedder_cls(model_n)
        try:
            emb_check = cls()
            assert_supported(emb_check, backend=eff_backend, output=output, temporal=temporal)
            require_model_config_support(
                embedder=emb_check,
                model_config=req.model_config,
                method_name="get_embedding",
            )
            desc = emb_check.describe() or {}
        except ModelError:
            raise
        except Exception as _e:
            desc = {}

        modality_eff = req.modality
        if modality_eff is None:
            modality_eff = per_model_modalities.get(model_name, modality)

        sensor_eff = req.sensor
        if sensor_eff is None:
            sensor_eff = per_model_sensors.get(model_name, sensor)

        fetch_eff = req.fetch
        if fetch_eff is None:
            fetch_eff = per_model_fetches.get(model_name, fetch)

        model_configs.append(
            ModelConfig(
                name=model_name,
                backend=eff_backend,
                sensor=resolve_sensor_for_model(
                    model_n,
                    sensor=sensor_eff,
                    fetch=fetch_eff,
                    modality=modality_eff,
                    default_when_missing=True,
                ),
                model_config=req.model_config,
                model_type=str(desc.get("type", "")).lower(),
            )
        )

    return model_configs, resolved_backend


def maybe_return_completed_combined_resume(
    *,
    target: ExportTarget,
    config: ExportConfig,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
) -> dict[str, object] | None:
    if target.layout != ExportLayout.COMBINED or not config.resume or not target.out_file:
        return None
    if not os.path.exists(target.out_file):
        return None
    json_path = os.path.splitext(target.out_file)[0] + ".json"
    resume_manifest = load_json_dict(json_path)
    if is_incomplete_combined_manifest(resume_manifest):
        return None
    return combined_resume_manifest(
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend,
        device=device,
        out_file=target.out_file,
    )
