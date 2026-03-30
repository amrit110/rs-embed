# ScaleMAE RGB (`scalemae`)

> Sentinel-2 RGB on-the-fly adapter for ScaleMAE (`rshf.scalemae.ScaleMAE`), with explicit scale conditioning via `sensor.scale_m -> input_res_m`.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `scalemae` |
| Aliases | `scalemae_rgb` |
| Family / Backbone | ScaleMAE via `rshf.scalemae.ScaleMAE` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) + `input_res_m` |
| Default resolution | 10m default provider fetch / semantic scale (`sensor.scale_m`) |
| Temporal mode | range window in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required semantic scale** (`sensor.scale_m` passed as `input_res_m`) |
| Training alignment (adapter path) | Medium-High when `sensor.scale_m` matches the actual input resolution semantics |

---

## When To Use This Model

ScaleMAE is the right choice for RGB experiments where spatial scale conditioning matters, for comparisons against SatMAE or RemoteCLIP with an explicitly scale-aware backbone, and for robustness studies across resolution changes where `scale_m` is part of the logged setup.

It becomes harder to interpret when `sensor.scale_m` is missing or semantically wrong for the input patch. You should also avoid assuming that `grid` is always available, because some wrapper outputs are pooled vectors only, and cross-run comparisons should record the output type through fields such as `tokens_kind`.

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

The current adapter path is provider-only, so use `backend="gee"` or another provider-compatible backend. `TemporalSpec` is normalized through the shared helper, and `TemporalSpec.range(...)` remains the clearest option for reproducible runs.

### Sensor / channels + scale

Default `SensorSpec` if omitted:

The default sensor is `COPERNICUS/S2_SR_HARMONIZED` with bands `("B4", "B3", "B2")`, `scale_m=10`, `cloudy_pct=30`, and `composite="median"`.

`input_chw` contract:

`input_chw` must be `CHW` with 3 channels in `(B4,B3,B2)` order, and the adapter expects raw Sentinel-2 SR values in `0..10000`.

Scale requirement:

The adapter passes `float(sensor.scale_m)` to ScaleMAE as `input_res_m`. If `sensor.scale_m` does not match the actual resolution semantics of the patch, the resulting embeddings are not meaningfully comparable.

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 RGB patch (provider path) or convert `input_chw` raw SR -> `[0,1]` -> `uint8`
2. Resize to `RS_EMBED_SCALEMAE_IMG` (default `224`)
3. Convert to tensor with CLIP-style preprocessing (`rgb_u8_to_tensor_clipnorm`)
4. Build `input_res` tensor from `sensor.scale_m`
5. Call ScaleMAE `forward_features(...)` (preferred) or `forward(...)`
   - adapter handles signature differences across `rshf` versions
   - passes both `patch_size` and `input_res`
6. Normalize output format:
   - `[N,D]` tokens
   - `[D]` pooled vector
   - `[C,H,W]` feature map reshaped to tokens
7. Return pooled vector or token grid

Important:

`grid` output requires a token sequence after adapter normalization. If the model or wrapper returns pooled vectors only, `OutputSpec.grid()` raises a clear error instead of silently fabricating a grid.

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_SCALEMAE_ID` | `MVRL/scalemae-vitlarge-800` | HF model ID for `ScaleMAE.from_pretrained(...)` |
| `RS_EMBED_SCALEMAE_IMG` | `224` | Resize / preprocess image size |
| `RS_EMBED_SCALEMAE_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |
| `RS_EMBED_SCALEMAE_BATCH_SIZE` | CPU:`8`, CUDA:`32` | Inference batch size for batch APIs |

Non-env but critical:

Even though it is not an environment variable, `sensor.scale_m` is a critical runtime setting because it is passed directly as `input_res_m`.

---

## Output Semantics

### `OutputSpec.pooled()`

If the adapter receives a token sequence `[N,D]`, `OutputSpec.pooled()` pools patch tokens with `mean` or `max`. If the model already returns a pooled vector `[D]`, that vector is returned directly and metadata marks the path as `model_pooled`. Metadata also records fields such as `tokens_kind`, `used_patch_size`, and `used_scale_m`.

### `OutputSpec.grid()`

`OutputSpec.grid()` requires a token sequence after adapter normalization and returns a patch-token grid as `xarray.DataArray` with shape `(D,H,W)`. As with the other ViT-like pages, this is model token layout rather than georeferenced raster pixels.

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "scalemae",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example tuning (env + scale semantics)

```python
# Example (shell):
# export RS_EMBED_SCALEMAE_ID=MVRL/scalemae-vitlarge-800
# export RS_EMBED_SCALEMAE_IMG=224
#
# In code, keep sensor.scale_m correct (this is passed to ScaleMAE as input_res_m).
```

---

## Common Failure Modes / Debugging

- backend mismatch (`scalemae` is provider-only)
- wrong `input_chw` shape / band order (`CHW`, 3 channels, `(B4,B3,B2)`)
- missing `rshf.scalemae.ScaleMAE`
- wrapper signature mismatch in older/newer `rshf` versions (adapter has fallbacks, but still possible)
- `grid` requested when model output is pooled vector only
- incorrect `sensor.scale_m` causing silent comparison drift

Recommended first checks:

Start by inspecting metadata such as `tokens_kind`, `used_patch_size`, and `input_res_m` or `used_scale_m`. Then verify `sensor.scale_m` and `RS_EMBED_SCALEMAE_IMG`, and use `OutputSpec.pooled()` first if you are isolating a grid-layout issue.

---

## Reproducibility Notes

For reproducibility, keep `RS_EMBED_SCALEMAE_ID`, `RS_EMBED_SCALEMAE_IMG`, and especially `sensor.scale_m` fixed, along with the temporal window, compositing settings, output mode, pooling choice, and `rshf` version.

---

## Source of Truth (Code Pointers)

The main implementation points are `src/rs_embed/embedders/catalog.py`, `src/rs_embed/embedders/onthefly_scalemae.py`, and the shared helpers in `src/rs_embed/embedders/_vit_mae_utils.py`.
