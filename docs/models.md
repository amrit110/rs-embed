# Supported Models (Overview)

This page is the **model selection entry point**.

Use it to answer one question quickly:

- Which models should I shortlist for this task?

Once you have a shortlist:

- use [Supported Models (Advanced Reference)](models_reference.md) for side-by-side comparison
- use the per-model detail pages linked below for exact contracts, examples, and caveats

---

## How To Use This Page

Recommended flow:

1. Pick a shortlist from the overview tables below
2. Validate temporal + input assumptions in [Advanced Reference](models_reference.md)
3. Open the linked detail page for your final candidate before production or benchmarking

Canonical model IDs use the short public names shown on this page, such as `remoteclip`, `prithvi`, `terrafm`, and `thor`.
Some detail-page filenames still use older names for compatibility, but the canonical IDs above are the names users should copy into code.

---

## Quick Chooser by Goal

| Goal | Good starting models | Why |
|---|---|---|
| Fast baseline / simple pipeline | `tessera`, `gse`, `copernicus` | Precomputed embeddings, fewer runtime dependencies |
| Simple S2 RGB on-the-fly experiments | `remoteclip`, `satmae`, `satmaepp`, `scalemae` | Straightforward RGB input paths |
| Time-series temporal modeling | `agrifm`, `anysat`, `galileo` | Native multi-frame temporal packaging |
| Multispectral / strict spectral semantics | `satmaepp_s2_10b`, `dofa`, `terramind`, `thor`, `satvision` | Strong channel/schema assumptions |
| Mixed-modality experiments (S1/S2) | `terrafm` | Supports S2 or S1 path (per call) |

## Model Catalog Snapshot

### Precomputed Embeddings

| Model ID | Type | Primary Input / Source | Outputs | Temporal mode | Notes | Detail |
|---|---|---|---|---|---|---|
| `tessera` | Precomputed | GeoTessera embedding tiles | `pooled`, `grid` | yearly coverage product | Fast baseline, source-fixed precomputed workflow | [detail](models/tessera.md) |
| `gse` | Precomputed | Google Satellite Embedding (annual) | `pooled`, `grid` | `TemporalSpec.year(...)` | Annual product via provider path | [detail](models/gse.md) |
| `copernicus` | Precomputed | Copernicus embeddings | `pooled`, `grid` | limited (2021) | Coarse resolution product | [detail](models/copernicus.md) |

### On-the-fly Foundation Models

| Model ID | Primary Input | Temporal style | Outputs | Notable requirements | Detail |
|---|---|---|---|---|---|
| `remoteclip` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | provider backend; RGB preprocessing | [detail](models/remoteclip.md) |
| `satmae` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | RGB path; ViT token/grid behavior | [detail](models/satmae.md) |
| `satmaepp` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | SatMAE++ fMoW-style eval preprocessing; channel order control | [detail](models/satmaepp.md) |
| `satmaepp_s2_10b` | S2 SR 10-band (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) | single composite window | `pooled`, `grid` | strict 10-band order; grouped-channel token handling | [detail](models/satmaepp.md) |
| `scalemae` | S2 RGB + scale | single composite window | `pooled`, `grid` | requires `sensor.scale_m` / `input_res_m` | [detail](models/scalemae.md) |
| `wildsat` | S2 RGB | single composite window | `pooled`, `grid` | normalization options | [detail](models/wildsat.md) |
| `prithvi` | S2 6-band | single composite window | `pooled`, `grid` | required temporal + location side inputs | [detail](models/prithvi.md) |
| `terrafm` | S2 12-band or S1 VV/VH | single composite window | `pooled`, `grid` | choose modality per call | [detail](models/terrafm.md) |
| `terramind` | S2 SR 12-band | single composite window | `pooled`, `grid` | strict normalization/channel semantics | [detail](models/terramind.md) |
| `dofa` | Multispectral + wavelengths | single composite window | `pooled`, `grid` | wavelength vector required/inferred | [detail](models/dofa.md) |
| `fomo` | S2 12-band | single composite window | `pooled`, `grid` | normalization mode choices | [detail](models/fomo.md) |
| `thor` | S2 SR 10-band | single composite window | `pooled`, `grid` | strict stats-based normalization | [detail](models/thor.md) |
| `satvision` | TOA 14-channel | single composite window | `pooled`, `grid` | strict channel order + calibration | [detail](models/satvision.md) |
| `anysat` | S2 10-band time series | multi-frame | `pooled`, `grid` | frame dates (`s2_dates`) side input | [detail](models/anysat.md) |
| `galileo` | S2 10-band time series | multi-frame | `pooled`, `grid` | month tokens + grouped tensors | [detail](models/galileo.md) |
| `agrifm` | S2 10-band time series | multi-frame | `pooled`, `grid` | fixed `T` frame stack behavior | [detail](models/agrifm.md) |

---

## Temporal and Comparison Notes (What People Usually Miss)

- `TemporalSpec.range(start, end)` is usually a **window for compositing**, not a single-scene acquisition selector.
- `OutputSpec.grid()` may be a **token/patch grid**, not a georeferenced raster grid (especially for ViT-like backbones).
- Cross-model comparisons are easiest with `OutputSpec.pooled()` and fixed ROI/temporal/compositing settings.
- Multi-frame models (`agrifm`, `anysat`, `galileo`) need extra attention to frame count and temporal side inputs.

Read the details in [Supported Models (Advanced Reference)](models_reference.md).

---

## Adding Model Pages

This repo now includes a reusable template page for documenting each model consistently:

- [Model Detail Template](model_detail_template.md)

Recommended fields for each model page:

- what the model expects (`input` contract, band order, temporal mode)
- what rs-embed currently feeds (current adapter behavior)
- preprocessing defaults and env knobs
- output semantics (`pooled` vs `grid` details)
- caveats for reproducibility / fair benchmarking

---

## Source of Truth in Code

Model registration source of truth:

- `src/rs_embed/embedders/catalog.py` (`MODEL_SPECS`)

Implementation details for each adapter live in:

- `src/rs_embed/embedders/onthefly_*.py`
- `src/rs_embed/embedders/precomputed_*.py`

When docs and code disagree, check code first and update docs accordingly.
