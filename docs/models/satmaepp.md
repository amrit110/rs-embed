# SatMAE++ Family (`satmaepp`, `satmaepp_s2_10b`)

> `rs-embed` currently exposes two SatMAE++ adapter paths: RGB (`satmaepp`) and Sentinel-2 10-band (`satmaepp_s2_10b`). This page documents the shared behavior first, then the variant-specific details.

## Quick Facts

| Field | `satmaepp` (RGB) | `satmaepp_s2_10b` (S2-10B) |
|---|---|---|
| Canonical ID | `satmaepp` | `satmaepp_s2_10b` |
| Aliases | `satmaepp_rgb`, `satmae++` | `satmaepp_sentinel10`, `satmaepp_s2` |
| Adapter type | `on-the-fly` | `on-the-fly` |
| Typical backend | provider backend (`gee`) | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) | S2 SR 10-band (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) |
| Default resolution | 10m default provider fetch (`sensor.scale_m`) | 10m default provider fetch (`sensor.scale_m`) |
| Temporal mode | range window + single composite | range window + single composite |
| Output modes | `pooled`, `grid` | `pooled`, `grid` |
| Model config keys | none | `model_config["variant"]` (default: `large`; choices: `large`) |
| Core extraction | `forward_encoder(mask_ratio=0.0)` | `forward_encoder(mask_ratio=0.0)` |

---

## When To Use This Family

### Good fit for

- SatMAE-style patch-token features with a familiar MAE encoder path
- comparisons between RGB and multispectral S2 setups within one model family
- workflows that want both pooled vectors and token-grid outputs

### Be careful when

- comparing the two variants as if they differ only by channel count; their preprocessing and runtime loading paths are materially different
- assuming `grid` is georeferenced pixel space; both variants return model token grids
- changing checkpoint or preprocessing settings without recording them

---

## Shared Output Semantics

### `OutputSpec.pooled()`

- Both variants pool patch tokens with `mean` or `max`
- `satmaepp` metadata typically records `patch_mean` / `patch_max`
- `satmaepp_s2_10b` metadata typically records `group_tokens_mean` / `group_tokens_max`

### `OutputSpec.grid()`

- `satmaepp` returns a standard ViT patch-token grid `(D,H,W)`
- `satmaepp_s2_10b` reduces grouped tokens across channel groups, then reshapes to `(D,H,W)`
- Both outputs are model token layouts, not georeferenced raster grids

---

## Variant A: `satmaepp` (RGB)

### Input Contract

Default `SensorSpec`:

- `collection="COPERNICUS/S2_SR_HARMONIZED"`
- `bands=("B4","B3","B2")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`

`input_chw` contract:

- must be 3-channel `CHW`
- band order must be `(B4,B3,B2)`
- raw S2 SR values are expected in `0..10000`

### Preprocessing Pipeline

1. Fetch RGB `uint8` from the provider, or convert `input_chw` from raw SR to `[0,1]` and then to `uint8`
2. Apply SatMAE++ fMoW RGB eval preprocessing:
   - channel order `rgb` or `bgr`
   - `ToTensor -> Normalize(mean/std) -> Resize(short side) -> CenterCrop(image_size)`
3. Run `forward_encoder(mask_ratio=0.0)` to extract tokens
4. Pool tokens or reshape them into a patch grid

### Key Environment Variables

| Env var | Effect |
|---|---|
| `RS_EMBED_SATMAEPP_ID` | HF model ID / checkpoint selector |
| `RS_EMBED_SATMAEPP_IMG` | Eval image size |
| `RS_EMBED_SATMAEPP_CHANNEL_ORDER` | `rgb` or `bgr` preprocessing order |
| `RS_EMBED_SATMAEPP_BGR` | Legacy BGR toggle |
| `RS_EMBED_SATMAEPP_FETCH_WORKERS` | Provider prefetch workers for batch APIs |
| `RS_EMBED_SATMAEPP_BATCH_SIZE` | Inference batch size for batch APIs |

### Common Failure Modes

- wrong `input_chw` shape or band order
- checkpoint preprocessing mismatch because `CHANNEL_ORDER` changed
- missing `rshf` / SatMAE++ wrapper dependencies
- unexpected token shape causing grid reshape failures

---

## Variant B: `satmaepp_s2_10b` (Sentinel-2 10-band)

### Input Contract

Default `SensorSpec`:

- `collection="COPERNICUS/S2_SR_HARMONIZED"`
- `bands=("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

Strict requirements:

- `sensor.bands` must exactly match the 10-band order above
- `input_chw` must be 10-channel `CHW`
- raw S2 SR values are expected in `0..10000`

### Preprocessing + Runtime Loading

1. Fetch 10-band `CHW`, or reuse `input_chw`
2. Apply source-style Sentinel statistics mapping (`mean ± 2*std`) to `uint8`
3. Apply eval transforms: `ToTensor -> Resize(short side) -> CenterCrop(image_size)`
4. Download or load runtime weights
5. Import vendored grouped-channel runtime
6. Construct the grouped-channel model with channel groups `((0,1,2,6),(3,4,5,7),(8,9))`
7. Run `forward_encoder(mask_ratio=0.0)` to extract grouped tokens
8. Reduce grouped tokens for pooled or grid output

### Key Environment Variables

| Env var | Effect |
|---|---|
| `RS_EMBED_SATMAEPP_S2_CKPT_REPO` | Checkpoint repo/source |
| `RS_EMBED_SATMAEPP_S2_CKPT_FILE` | Checkpoint filename |
| `RS_EMBED_SATMAEPP_S2_MODEL_FN` | Model constructor name |
| `RS_EMBED_SATMAEPP_S2_IMG` | Eval image size |
| `RS_EMBED_SATMAEPP_S2_PATCH` | Patch size |
| `RS_EMBED_SATMAEPP_S2_GRID_REDUCE` | Group reduction mode for grid output |
| `RS_EMBED_SATMAEPP_S2_WEIGHTS_ONLY` | Weights-only checkpoint loading toggle |
| `RS_EMBED_SATMAEPP_S2_FETCH_WORKERS` | Provider prefetch workers for batch APIs |
| `RS_EMBED_SATMAEPP_S2_BATCH_SIZE` | Inference batch size for batch APIs |

### Common Failure Modes

- `sensor.bands` order differs from the strict expected 10-band layout
- vendored runtime import fails or checkpoint download is unavailable
- grouped-token reshape assumptions do not match the loaded checkpoint/config
- `GRID_REDUCE` changes representation semantics across experiments

---

## Examples

### Minimal pooled examples

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.range("2022-06-01", "2022-09-01")

emb_rgb = get_embedding(
    "satmaepp",
    spatial=spatial,
    temporal=temporal,
    output=OutputSpec.pooled(),
    backend="gee",
)

emb_s2 = get_embedding(
    "satmaepp_s2_10b",
    spatial=spatial,
    temporal=temporal,
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example tuning knobs (env-controlled)

```python
# RGB variant:
# export RS_EMBED_SATMAEPP_ID=...
# export RS_EMBED_SATMAEPP_CHANNEL_ORDER=bgr
#
# S2-10B variant:
# export RS_EMBED_SATMAEPP_S2_IMG=224
# export RS_EMBED_SATMAEPP_S2_GRID_REDUCE=mean
```

### Example with `model_config`

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.range("2022-06-01", "2022-09-01")

emb_s2 = get_embedding(
    "satmaepp_s2_10b",
    spatial=spatial,
    temporal=temporal,
    output=OutputSpec.grid(),
    backend="gee",
    model_config={"variant": "large"},
)
```

For export jobs, the same setting goes through
`ExportModelRequest("satmaepp_s2_10b", model_config={"variant": "large"})`.

---

## Reproducibility Notes

Keep fixed and record:

- which variant you used (`satmaepp` vs `satmaepp_s2_10b`)
- checkpoint source
- image size, patch size, and channel/group reduction settings
- temporal window and provider compositing settings
- output mode (`pooled` vs `grid`) and pooling choice

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- RGB adapter: `src/rs_embed/embedders/onthefly_satmaepp.py`
- S2-10B adapter: `src/rs_embed/embedders/onthefly_satmaepp_s2.py`
