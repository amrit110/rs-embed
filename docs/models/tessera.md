# Tessera (`tessera`)

> Precomputed embedding adapter backed by GeoTessera tiles, with strict tile mosaic + ROI crop behavior. Use `backend="auto"`.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `tessera` |
| Family / Source | GeoTessera precomputed embeddings |
| Adapter type | `precomputed` |
| Typical backend | `auto` |
| Primary input | `BBox` / `PointBuffer` in EPSG:4326 (converted to bbox) |
| Default resolution | 10m source product resolution |
| Temporal mode | year-like selection (`year`; `range` uses start year fallback) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none |
| Training alignment (adapter path) | N/A (precomputed product) |

---

## When To Use This Model

Tessera is a strong choice for fast precomputed baselines, large-area ROI extraction without model inference runtime, and workflows that benefit from local tile mosaic and crop behavior rather than provider fetches. It is easy to misuse if you expect arbitrary backends, treat `TemporalSpec.range(...)` as exact temporal semantics, or work across tiles with incompatible CRS or resolution.

---

## Input Contract (Current Adapter Path)

### Spatial

The adapter accepts `BBox` directly and `PointBuffer`, which it converts to `BBox` in EPSG:4326 using an approximate meter-to-degree conversion. Unsupported spatial types raise `ModelError`.

### Temporal

`temporal=None` defaults to year `2021`. `TemporalSpec.year(...)` uses `temporal.year`, and `TemporalSpec.range(start, end)` uses the year parsed from `start`. This is a year selector for tile-product lookup, not scene-level temporal filtering.

### Backend / cache

The backend should be `auto`, although legacy `local` is still accepted for compatibility. The adapter reads the GeoTessera cache from `RS_EMBED_TESSERA_CACHE` or from a per-call override such as `sensor.collection="cache:/path/to/cache"`.

---

## Preprocessing / Retrieval Pipeline (Current rs-embed Path)

1. Convert `SpatialSpec` to `BBox` in EPSG:4326
2. Resolve year from `TemporalSpec` (with fallback behavior)
3. Open/cache `geotessera.GeoTessera` instance by `cache_dir`
4. Query tile blocks covering ROI/year
5. Fetch tile embeddings and normalize array layout (`HWC` or `CHW` -> internal `HWC`)
6. Strict mosaic + crop:
   - requires north-up transforms (no rotation/shear)
   - requires consistent tile CRS and resolution across fetched tiles
   - reprojects ROI bbox into tile CRS if needed (requires `pyproj`)
7. Convert cropped result to `CHW`
8. Return pooled vector or grid

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_TESSERA_CACHE` | unset (GeoTessera default) | Local GeoTessera cache directory |
| `RS_EMBED_TESSERA_BATCH_WORKERS` | `4` | Batch worker count for `get_embeddings_batch(...)` |

Non-env override:

`sensor.collection="cache:/path/to/cache"` overrides the cache directory for one call.

---

## Output Semantics

Tessera follows the standard precomputed-product behavior. `pooled` applies spatial pooling over the cropped embedding grid, and `grid` returns the cropped `(D,H,W)` embedding grid in product pixel space after mosaic and crop rather than raw imagery space.

---

## Examples

### Minimal example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "tessera",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=5000),
    temporal=TemporalSpec.year(2021),
    output=OutputSpec.pooled(),
    backend="auto",
)
```

### Example cache override

```python
# Example (shell):
# export RS_EMBED_TESSERA_CACHE=/data/geotessera
```

---

## Common Failure Modes / Debugging

- backend is not `auto`
- no tiles found for ROI/year
- tile CRS/resolution mismatch during mosaic
- tile transform has rotation/shear (not north-up)
- missing optional deps for CRS reprojection (`pyproj`) on non-4326 tiles

Recommended first checks:

Inspect `preferred_year`, `bbox_4326`, and `chw_shape` first, then inspect crop metadata such as `tile_crs`, `mosaic_hw`, and `crop_px_window`. If no tiles are found, try a larger ROI before assuming the cache is broken.

---

## Reproducibility Notes

Keep the GeoTessera cache snapshot or path, year-selection logic, ROI geometry, CRS, and output mode fixed and recorded.

---

## Source of Truth (Code Pointers)

The main code paths are `src/rs_embed/embedders/catalog.py` for registration and `src/rs_embed/embedders/precomputed_tessera.py` for the adapter implementation.
