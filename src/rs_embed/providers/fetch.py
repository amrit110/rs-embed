"""Provider fetch helpers and satellite-data normalization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..core.errors import ModelError
from ..core.specs import NormalizationSpec, SensorSpec, SpatialSpec, TemporalSpec
from .base import ProviderBase


def fetch_sensor_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec,
    to_float_image: bool = False,
) -> np.ndarray:
    """Fetch a CHW patch from a concrete SensorSpec, re-raising ProviderError as ModelError."""
    from ..core.errors import ProviderError

    try:
        return provider.fetch_sensor_patch_chw(
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            to_float_image=to_float_image,
        )
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc


def fetch_collection_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    collection: str,
    bands: tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: int | None = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch a provider patch as CHW float32 using shared SensorSpec logic."""
    sensor = SensorSpec(
        collection=str(collection),
        bands=tuple(str(b) for b in bands),
        scale_m=int(scale_m),
        cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),  # type: ignore[arg-type]
        fill_value=float(fill_value),
        composite=str(composite),
    )
    return fetch_sensor_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )


def _stitch_spatial_last2_arrays(
    *,
    a: np.ndarray,
    b: np.ndarray,
    parent_spatial: Any,
    axis: str,
    scale_m: int,
    fill_value: float,
) -> np.ndarray:
    from .gee_utils import _stitch_bbox_split_arrays

    return _stitch_bbox_split_arrays(
        arr_a=np.asarray(a, dtype=np.float32),
        arr_b=np.asarray(b, dtype=np.float32),
        parent_spatial=parent_spatial,
        axis=axis,
        scale_m=scale_m,
        fill_value=fill_value,
    )


def _fetch_spatial_array_with_bbox_fallback(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    scale_m: int,
    fill_value: float,
    fetch_fn: Callable[[SpatialSpec], np.ndarray],
    split_depth: int = 0,
) -> np.ndarray:
    from . import gee_utils as _ah

    try:
        return np.asarray(fetch_fn(spatial), dtype=np.float32)
    except Exception as e:
        if not (
            _ah._looks_like_gee_sample_too_many_pixels(e) and _ah._looks_like_bbox_spatial(spatial)
        ):
            raise
        max_depth = int(getattr(_ah, "_MAX_GEE_BBOX_SPLIT_DEPTH", 12))
        if int(split_depth) >= max_depth:
            raise ModelError(
                f"GEE bbox fallback exceeded max recursive splits ({max_depth})."
            ) from e

        spatial_bbox = _ah._coerce_bbox_like(spatial)
        h_est, w_est = _ah._bbox_span_pixels_estimate(spatial_bbox, scale_m=int(scale_m))
        prefer_axis = "x" if int(w_est) >= int(h_est) else "y"
        a_sp, b_sp, axis = _ah._split_bbox_for_recursive_fetch(
            spatial_bbox, prefer_axis=prefer_axis
        )
        arr_a = _fetch_spatial_array_with_bbox_fallback(
            provider,
            spatial=a_sp,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            fetch_fn=fetch_fn,
            split_depth=int(split_depth) + 1,
        )
        arr_b = _fetch_spatial_array_with_bbox_fallback(
            provider,
            spatial=b_sp,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            fetch_fn=fetch_fn,
            split_depth=int(split_depth) + 1,
        )
        return _stitch_spatial_last2_arrays(
            a=arr_a,
            b=arr_b,
            parent_spatial=spatial_bbox,
            axis=axis,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
        )


def fetch_collection_patch_all_bands_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    collection: str,
    scale_m: int = 10,
    fill_value: float = 0.0,
    composite: str = "median",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fetch all bands for a collection with BBox fallback stitching for large GEE samples."""

    def _fetch_once(sp: SpatialSpec) -> tuple[np.ndarray, tuple[str, ...]]:
        arr, names = provider.fetch_collection_patch_all_bands_chw(
            spatial=sp,
            temporal=temporal,
            collection=str(collection),
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            composite=str(composite),
        )
        return np.asarray(arr, dtype=np.float32), tuple(str(b) for b in names)

    try:
        arr, names = _fetch_once(spatial)
        return np.asarray(arr, dtype=np.float32), tuple(names)
    except Exception as e:
        from . import gee_utils as _ah

        if not (
            _ah._looks_like_gee_sample_too_many_pixels(e) and _ah._looks_like_bbox_spatial(spatial)
        ):
            raise

        def _rec(sp: SpatialSpec, depth: int = 0) -> tuple[np.ndarray, tuple[str, ...]]:
            max_depth = int(getattr(_ah, "_MAX_GEE_BBOX_SPLIT_DEPTH", 12))
            try:
                return _fetch_once(sp)
            except Exception as ee:
                if not (
                    _ah._looks_like_gee_sample_too_many_pixels(ee)
                    and _ah._looks_like_bbox_spatial(sp)
                ):
                    raise
                if int(depth) >= max_depth:
                    raise ModelError(
                        f"GEE bbox fallback exceeded max recursive splits ({max_depth})."
                    ) from ee
                sp_bbox = _ah._coerce_bbox_like(sp)
                h_est, w_est = _ah._bbox_span_pixels_estimate(sp_bbox, scale_m=int(scale_m))
                prefer_axis = "x" if int(w_est) >= int(h_est) else "y"
                a_sp, b_sp, axis = _ah._split_bbox_for_recursive_fetch(
                    sp_bbox, prefer_axis=prefer_axis
                )
                arr_a, names_a = _rec(a_sp, depth + 1)
                arr_b, names_b = _rec(b_sp, depth + 1)
                if tuple(names_a) != tuple(names_b):
                    raise ModelError(
                        "Band names mismatch while stitching all-band bbox tiles."
                    ) from None
                stitched = _stitch_spatial_last2_arrays(
                    a=arr_a,
                    b=arr_b,
                    parent_spatial=sp_bbox,
                    axis=axis,
                    scale_m=int(scale_m),
                    fill_value=float(fill_value),
                )
                return stitched, tuple(names_a)

        return _rec(spatial, 0)


def fetch_s2_rgb_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    """Fetch Sentinel-2 RGB as float32 CHW in [0,1]."""
    raw = fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
        fill_value=0.0,
    )
    return np.clip(raw / 10000.0, 0.0, 1.0).astype(np.float32)


def fetch_s1_vvvh_raw_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    orbit: str | None = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> np.ndarray:
    """Fetch Sentinel-1 VV/VH as raw float32 CHW."""
    arr = _fetch_spatial_array_with_bbox_fallback(
        provider,
        spatial=spatial,
        scale_m=int(scale_m),
        fill_value=float(fill_value),
        fetch_fn=lambda sp: provider.fetch_s1_vvvh_raw_chw(
            spatial=sp,
            temporal=temporal,
            scale_m=int(scale_m),
            orbit=orbit,
            use_float_linear=bool(use_float_linear),
            composite=str(composite),
            fill_value=float(fill_value),
            require_iw=bool(require_iw),
            relax_iw_on_empty=bool(relax_iw_on_empty),
        ),
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fetch_s1_vvvh_raw_chw_with_meta(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    orbit: str | None = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch Sentinel-1 VV/VH as raw float32 CHW together with fetch metadata."""
    if hasattr(provider, "fetch_s1_vvvh_raw_chw_with_meta"):
        arr, meta = provider.fetch_s1_vvvh_raw_chw_with_meta(
            spatial=spatial,
            temporal=temporal,
            scale_m=int(scale_m),
            orbit=orbit,
            use_float_linear=bool(use_float_linear),
            composite=str(composite),
            fill_value=float(fill_value),
            require_iw=bool(require_iw),
            relax_iw_on_empty=bool(relax_iw_on_empty),
        )
    else:
        arr = fetch_s1_vvvh_raw_chw(
            provider,
            spatial=spatial,
            temporal=temporal,
            scale_m=int(scale_m),
            orbit=orbit,
            use_float_linear=bool(use_float_linear),
            composite=str(composite),
            fill_value=float(fill_value),
            require_iw=bool(require_iw),
            relax_iw_on_empty=bool(relax_iw_on_empty),
        )
        meta = {
            "s1_iw_requested": bool(require_iw),
            "s1_iw_applied": bool(require_iw),
            "s1_iw_relaxed_on_empty": False,
            "s1_relax_iw_on_empty": bool(relax_iw_on_empty),
        }
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}")
    meta_out = dict(meta or {})
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), meta_out


def fetch_s2_multiframe_raw_tchw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    bands: Sequence[str],
    n_frames: int = 8,
    collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    scale_m: int = 10,
    cloudy_pct: int | None = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch an S2 time series as raw float32 [T,C,H,W] in [0,10000]."""
    arr = _fetch_spatial_array_with_bbox_fallback(
        provider,
        spatial=spatial,
        scale_m=int(scale_m),
        fill_value=float(fill_value),
        fetch_fn=lambda sp: provider.fetch_multiframe_collection_raw_tchw(
            spatial=sp,
            temporal=temporal,
            collection=str(collection),
            bands=tuple(str(b) for b in bands),
            n_frames=int(n_frames),
            scale_m=int(scale_m),
            cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),
            composite=str(composite),
            fill_value=float(fill_value),
        ),
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise ModelError(f"Expected TCHW array, got shape={getattr(arr, 'shape', None)}")
    if int(arr.shape[1]) != len(tuple(bands)):
        raise ModelError(
            f"Time series channel mismatch: got C={int(arr.shape[1])}, expected C={len(tuple(bands))}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_s1_vvvh_chw(raw_chw: np.ndarray) -> np.ndarray:
    """Convert raw S1 VV/VH to numerically stable [0,1] CHW."""
    arr = np.asarray(raw_chw, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(
            f"Expected raw S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}"
        )
    x = np.log1p(np.maximum(arr, 0.0))
    denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
    denom = float(denom) if float(denom) > 0 else 1.0
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32)


def apply_normalization(raw: np.ndarray, norm: NormalizationSpec) -> np.ndarray:
    """Apply a declarative normalization strategy to raw provider data."""
    arr = np.asarray(raw, dtype=np.float32)
    if norm.mode == "s2_sr_clip":
        return np.clip(arr / 10000.0, 0.0, 1.0).astype(np.float32)
    if norm.mode == "s2_sr_raw":
        return np.clip(arr, 0.0, 10000.0).astype(np.float32)
    if norm.mode == "s1_log_normalize":
        return normalize_s1_vvvh_chw(arr)
    if norm.mode == "none":
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ModelError(f"Unknown normalization mode: {norm.mode!r}")
