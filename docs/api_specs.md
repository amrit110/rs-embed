# API Specs & Data Structures

This page documents the core spec/data types used across the public API.

For task-oriented usage, see [Common Workflows](workflows.md). For exact embedding, export, and inspect functions, see [API: Embedding](api_embedding.md), [API: Export](api_export.md), and [API: Inspect](api_inspect.md).

=== ":material-shape-outline: Spatial"

    Start with `SpatialSpec` (`BBox` or `PointBuffer`) to define the ROI.

=== ":material-calendar-range: Temporal"

    Use `TemporalSpec.year(...)` for precomputed/year-indexed products and `TemporalSpec.range(...)` for provider/on-the-fly fetch windows.

=== ":material-grid: Output"

    Use `OutputSpec.pooled()` first unless you specifically need spatial structure (`grid`).

---

## Data Structures

### SpatialSpec

`SpatialSpec` describes the spatial region for which you want to extract an embedding.

#### `BBox`

```python
BBox(minlon: float, minlat: float, maxlon: float, maxlat: float, crs: str = "EPSG:4326")
```

This is an **EPSG:4326** latitude/longitude bounding box. The current public API supports only EPSG:4326, and `validate()` checks that the bounds are valid.

#### `PointBuffer`

```python
PointBuffer(lon: float, lat: float, buffer_m: float, crs: str = "EPSG:4326")
```

This is a square ROI centered at a point, with size defined in meters and internally projected into the coordinate system required by the provider. `buffer_m` must be greater than zero.

---

### TemporalSpec

`TemporalSpec` describes the time range (by year or by date range).

```python
TemporalSpec(mode: Literal["year", "range"], year: int | None, start: str | None, end: str | None)
```

Recommended constructors:

```python
TemporalSpec.year(2022)
TemporalSpec.range("2022-06-01", "2022-09-01")
```

!!! warning "Temporal range is a window"
    `TemporalSpec.range(start, end)` is treated as a half-open interval `[start, end)`, so `end` is excluded.

Temporal semantics in provider and on-the-fly paths: `TemporalSpec.range(start, end)` is interpreted as a half-open window `[start, end)`, so `end` is excluded. In GEE-backed fetch paths, that window is used to filter an image collection and then apply a compositing reducer such as `median` or `mosaic`. The fetched input is therefore usually a composite over the whole window rather than an automatically selected single-day scene. If you want to approximate a single-day query, use a one-day window such as `TemporalSpec.range("2022-06-01", "2022-06-02")`.

About `input_time` in metadata: many embedders store `meta["input_time"]` as the midpoint date of the temporal window. That midpoint is metadata, and for some models an auxiliary time signal, rather than proof that imagery was fetched from exactly that single date.

!!! note "Common gotcha"
    `input_time` often looks like a single date, but the actual provider fetch may still be a composite over the full temporal window.

---

### SensorSpec

`SensorSpec` is mainly for **on-the-fly** models (fetch a patch from GEE online and feed it into the model). It specifies which collection to pull from, which bands, and what resolution/compositing strategy to use.

```python
SensorSpec(
    collection: str,
    bands: Tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    fill_value: float = 0.0,
    composite: Literal["median", "mosaic"] = "median",
    modality: Optional[str] = None,
    orbit: Optional[str] = None,
    use_float_linear: bool = True,
    s1_require_iw: bool = True,
    s1_relax_iw_on_empty: bool = True,
    check_input: bool = False,
    check_raise: bool = True,
    check_save_dir: Optional[str] = None,
)
```

| Field | Meaning |
|---|---|
| `collection` | GEE collection or image ID. |
| `bands` | Band names as a tuple. |
| `scale_m` | Sampling resolution in meters. |
| `cloudy_pct` | Best-effort cloud filter, subject to collection properties. |
| `fill_value` | No-data fill value. |
| `composite` | Temporal compositing method, usually `median` or `mosaic`. |
| `modality` | Optional model-facing modality selector for models with multiple branches. |
| `orbit` | Optional orbit or pass filter for sensor families that support it. |
| `use_float_linear` | Selects linear-scale floating-point products when a sensor family offers both linear and dB variants. |
| `s1_require_iw` | Prefer Sentinel-1 `IW` scenes only on provider-backed S1 fetch paths. |
| `s1_relax_iw_on_empty` | Retry without the `IW` filter when strict S1 `IW` fetch returns no imagery. |
| `check_*` | Optional input checks and quicklook saving; see [API: Inspect](api_inspect.md#inspect_gee_patch). |

!!! note
    For **precomputed** models (e.g., directly reading offline embedding products), `sensor` is usually ignored or set to `None`.

!!! note
    Public embedding/export APIs also accept a top-level `modality=...` convenience argument.
    When provided, rs-embed resolves it into the model's sensor/input contract and validates that the model explicitly supports that modality.

### FetchSpec

`FetchSpec` is the lightweight public override for **sampling / fetch policy**.
Use it when you want to change common knobs such as resolution or compositing, but do **not** want to define a full `SensorSpec`.

```python
FetchSpec(
    scale_m: int | None = None,
    cloudy_pct: int | None = None,
    fill_value: float | None = None,
    composite: Literal["median", "mosaic"] | None = None,
)
```

| Field | Meaning |
|---|---|
| `scale_m` | Sampling resolution override. |
| `cloudy_pct` | Cloud filter override. |
| `fill_value` | No-data fill override. |
| `composite` | Temporal compositing override. |

Recommended rule: use `fetch=FetchSpec(...)` for normal public API usage. Use `sensor=SensorSpec(...)` only when you need advanced control over `collection`, `bands`, `modality`, or similar source-level details.

Important constraints: `fetch` and `sensor` cannot be passed together in the same request, and `fetch` is always applied on top of the model's default sensor contract. Some models use `scale_m` as more than fetch resolution: for example, `scalemae` uses it as semantic scale conditioning, and `anysat` uses it as both fetch resolution and patch-size control.

Example:

```python
from rs_embed import FetchSpec, get_embedding

emb = get_embedding(
    "prithvi",
    spatial=...,
    temporal=...,
    fetch=FetchSpec(scale_m=10, cloudy_pct=10),
)
```

---

### OutputSpec

`OutputSpec` controls the embedding output shape: a **pooled vector** or a **dense grid**.


```python
OutputSpec(
    mode: Literal["grid", "pooled"],
    pooling: Literal["mean", "max"] = "mean",
    grid_orientation: Literal["north_up", "native"] = "north_up",
)
```

#### Recommended Constructors

=== ":material-vector-line: Pooled (default)"

    ```python
    OutputSpec.pooled(pooling="mean")   # shape: (D,)
    ```

=== ":material-grid: Grid (spatial)"

    ```python
    OutputSpec.grid()         # shape: (D, H, W), normalized to north-up when possible
    OutputSpec.grid(grid_orientation="native")  # keep model/provider native orientation
    ```

Sampling resolution is no longer configured on `OutputSpec`.
Use `fetch=FetchSpec(scale_m=...)` for resolution overrides.


#### `pooled`

> ROI-level Vector Embedding


`pooled` represents one whole ROI (Region of Interest) using a single vector `(D,)`.

Best suited for classification, retrieval, clustering, and most cross-model comparison work. The out put shape is:

```python
Embedding.data.shape == (D,)
```

How `pooled` is produced:

- **ViT / MAE-style models** (e.g., RemoteCLIP / Prithvi / SatMAE / ScaleMAE)
  Start from patch tokens `(N, D)` with an optional CLS token. The adapter removes the CLS token if present, then pools across the token axis, usually with `mean` and optionally with `max`.

Mean-pooling formula:

$$
v_d = \frac{1}{N'} \sum_{i=1}^{N'} t_{i,d}
$$

- **Precomputed embeddings** (e.g., Tessera / GSE / Copernicus)
  These already expose an embedding grid `(D, H, W)`, so pooling happens over the spatial dimensions `(H, W)`.

$$
v_d = \frac{1}{HW} \sum_{y,x} g_{d,y,x}
$$


#### `grid`
> ROI-level Spatial Embedding Field


`grid` returns a spatial embedding field `(D, H, W)`, where each spatial location maps to a vector.

Best suited for spatial visualization, pixel-wise or patch-wise tasks, and intra-ROI structure analysis. The output shape is:

```python
Embedding.data.shape == (D, H, W)
```

!!! note
    `data` can be returned as `xarray.DataArray`, with metadata in `meta` or `attrs`. For precomputed geospatial products, that metadata may include CRS and crop context. For ViT token grids, it usually describes patch-grid structure rather than georeferenced pixel coordinates.

How `grid` is produced:

- **ViT / MAE-style models**
  Start from tokens `(N, D)`. The adapter removes a CLS token if needed, reshapes the remaining tokens from `(N', D)` to `(H, W, D)`, and then reorders them to `(D, H, W)`. Here `(H, W)` comes from the patch layout, such as `8x8` or `14x14`.

- **Precomputed embeddings**
  These already use `(D, H, W)` as the native output shape, so the API can return that structure directly.

---

### InputPrepSpec 
> Optional Large-ROI Input Policy

`InputPrepSpec` controls API-level handling of large on-the-fly inputs before model inference.
This is mainly useful when you want to choose between the model's normal resize path and API-side tiled inference.

```python
InputPrepSpec(
    mode: Literal["resize", "tile", "auto"] = "resize",
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    max_tiles: int = 9,
    pad_edges: bool = True,
)
```

#### Recommended Constructors

```python
InputPrepSpec.resize()               # default behavior (fastest)
InputPrepSpec.tile()                 # tile size inferred from model defaults.image_size when available
InputPrepSpec.auto(max_tiles=4)      # choose tile or resize automatically
InputPrepSpec.tile(tile_size=224)    # explicit tile size override
```

#### String Shorthand

```python
input_prep="resize"   # default
input_prep="tile"
input_prep="auto"
```

#### Current Tiling Behavior

Tile size defaults to `embedder.describe()["defaults"]["image_size"]` when available, unless you override it. Boundary tiles use a cover-shift layout such as `300 -> [0,224]` and `[76,300]` to avoid edge padding when possible, and grid stitching uses midpoint-cut ownership in overlap regions rather than hard overwrite. `tile_stride` currently must equal `tile_size`, so explicit overlap or gap control is not enabled yet, although boundary shifting can still create overlap on the last tile. `auto` is conservative and currently prefers tiling mainly for `OutputSpec.grid()` when tile count is small enough.

![tiles](assets/tiles.png)

<!-- <img src="./docs/assets/tiles.png" width="500" alt="icon" /> -->


---

### ExportTarget / ExportConfig / ExportModelRequest

`export_batch(...)` now uses small public request objects so large export jobs do not need dozens of top-level keywords.

#### Examples

```python
ExportTarget.combined("exports/run")
ExportTarget.per_item("exports/items", names=["p1", "p2"])

ExportConfig(
    save_inputs=True,
    save_embeddings=True,
    chunk_size=32,
    num_workers=8,
    resume=True,
)

ExportModelRequest("remoteclip")
ExportModelRequest("terrafm", modality="s1", sensor=my_s1_sensor)
ExportModelRequest.configure("thor", variant="large")
```

`ExportTarget` controls where outputs are written, `ExportConfig` controls how the export runs, and `ExportModelRequest` carries optional per-model overrides when one job mixes different settings such as sensor, modality, or variant. Use `ExportModelRequest.configure(...)` when you want to pass model settings as keyword arguments.

Legacy `out + layout`, `out_dir` / `out_path`, and per-model dict overrides are still accepted for backward compatibility.

---

### Embedding

`get_embedding` / `get_embeddings_batch` return an `Embedding`:


```python
from rs_embed.core.embedding import Embedding

Embedding(
    data: np.ndarray | xarray.DataArray,
    meta: Dict[str, Any],
)
```

`data` holds the embedding itself as a float32 vector or grid, and `meta` carries model information, optional input information, and export or check reports.

---
