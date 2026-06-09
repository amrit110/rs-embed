"""Tests for OlmoEarth embedder (onthefly_olmoearth.py)."""
from __future__ import annotations

import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders import onthefly_olmoearth as oe


# ---------------------------------------------------------------------------
# Variant normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("nano", "nano"),
        ("NANO", "nano"),
        ("Nano", "nano"),
        ("nano_v1", "nano"),
        ("tiny", "tiny"),
        ("base", "base"),
        ("large", "large"),
        ("large_v1", "large"),
        ("nano_v1_1", "nano_v1_1"),
        ("NANO_V1_1", "nano_v1_1"),
        ("nano_11", "nano_v1_1"),
        ("tiny_v1_1", "tiny_v1_1"),
        ("tiny_11", "tiny_v1_1"),
        ("base_v1_1", "base_v1_1"),
        ("base_11", "base_v1_1"),
    ],
)
def test_normalize_variant_valid(raw, expected):
    assert oe._normalize_variant(raw) == expected


def test_normalize_variant_raises_on_unknown():
    with pytest.raises(ModelError, match="Unknown OlmoEarth variant"):
        oe._normalize_variant("xlarge")


# ---------------------------------------------------------------------------
# _resolve_variant
# ---------------------------------------------------------------------------


def test_resolve_variant_from_model_config():
    assert oe._resolve_variant({"variant": "base"}) == "base"
    assert oe._resolve_variant({"variant": "nano_v1_1"}) == "nano_v1_1"


def test_resolve_variant_defaults_to_nano():
    assert oe._resolve_variant(None) == "nano"
    assert oe._resolve_variant({}) == "nano"


def test_resolve_variant_env_fallback(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_VARIANT", "tiny")
    assert oe._resolve_variant(None) == "tiny"


def test_resolve_variant_model_config_overrides_env(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_VARIANT", "tiny")
    assert oe._resolve_variant({"variant": "base"}) == "base"


def test_resolve_variant_env_auto_gives_default(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_VARIANT", "auto")
    assert oe._resolve_variant(None) == "nano"


# ---------------------------------------------------------------------------
# _resolve_patch_size / _resolve_image_size
# ---------------------------------------------------------------------------


def test_resolve_patch_size_defaults():
    assert oe._resolve_patch_size(None) == oe._DEFAULT_PATCH_SIZE


def test_resolve_patch_size_from_config():
    assert oe._resolve_patch_size({"patch_size": 8}) == 8
    assert oe._resolve_patch_size({"patch_size": "2"}) == 2


def test_resolve_patch_size_rejects_out_of_range():
    with pytest.raises(ModelError, match="patch_size must be 1"):
        oe._resolve_patch_size({"patch_size": 9})


def test_resolve_image_size_defaults():
    assert oe._resolve_image_size(None) == oe._DEFAULT_IMAGE_SIZE


def test_resolve_image_size_from_config():
    assert oe._resolve_image_size({"image_size": 128}) == 128


# ---------------------------------------------------------------------------
# _date_to_timestamp
# ---------------------------------------------------------------------------


def test_date_to_timestamp_known_date():
    day, month, year = oe._date_to_timestamp("2022-07-15")
    assert day == 15
    assert month == 6   # 0-indexed: July = 6
    assert year == 2022


def test_date_to_timestamp_january():
    day, month, year = oe._date_to_timestamp("2020-01-01")
    assert month == 0   # January = 0


def test_date_to_timestamp_none_returns_default():
    day, month, year = oe._date_to_timestamp(None)
    assert isinstance(day, int)
    assert isinstance(month, int)
    assert isinstance(year, int)


# ---------------------------------------------------------------------------
# _normalize_s2_chw
# ---------------------------------------------------------------------------


def test_normalize_s2_chw_output_shape_and_dtype():
    raw = np.random.uniform(0, 3000, (12, 32, 32)).astype(np.float32)
    out = oe._normalize_s2_chw(raw)
    assert out.shape == (12, 32, 32)
    assert out.dtype == np.float32


def test_normalize_s2_chw_no_nans():
    raw = np.zeros((12, 16, 16), dtype=np.float32)
    out = oe._normalize_s2_chw(raw)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


# ---------------------------------------------------------------------------
# _prepare_chw
# ---------------------------------------------------------------------------


def test_prepare_chw_resizes_to_image_size():
    raw = np.random.uniform(0, 3000, (12, 64, 64)).astype(np.float32)
    out = oe._prepare_chw(raw, image_size=256, patch_size=4)
    assert out.shape == (12, 256, 256)


def test_prepare_chw_wrong_channels_raises():
    bad = np.zeros((6, 32, 32), dtype=np.float32)
    with pytest.raises(ModelError, match="12-band"):
        oe._prepare_chw(bad, image_size=256, patch_size=4)


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def test_olmoearth_registered_in_catalog():
    from rs_embed.embedders.catalog import MODEL_SPECS
    assert "olmoearth" in MODEL_SPECS
    module, cls_name = MODEL_SPECS["olmoearth"]
    assert module == "onthefly_olmoearth"
    assert cls_name == "OlmoEarthEmbedder"


def test_olmoearth_lazy_loads_via_registry():
    from rs_embed.core.registry import get_embedder_cls
    cls = get_embedder_cls("olmoearth")
    assert cls is oe.OlmoEarthEmbedder


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


def test_describe_returns_expected_keys():
    emb = oe.OlmoEarthEmbedder()
    info = emb.describe()
    assert info["type"] == "onthefly"
    assert "pooled" in info["output"]
    assert "grid" in info["output"]
    assert "model_config" in info
    assert "variant" in info["model_config"]
    assert len(info["input_bands"]) == 12


# ---------------------------------------------------------------------------
# get_embedding — mocked forward
# ---------------------------------------------------------------------------


def _fake_encoder_output(batch_size: int = 1, hw: int = 4, d: int = 128):
    """Return a fake tokens_and_masks NamedTuple-like object."""
    import torch

    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue  # type: ignore
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (  # type: ignore
        TokensAndMasks,
    )

    s2 = torch.randn(batch_size, hw, hw, 1, 3, d)  # (B, H', W', T, S, D)
    s2_mask = torch.zeros(batch_size, hw, hw, 1, 3)  # all ONLINE_ENCODER
    ts = TokensAndMasks(sentinel2_l2a=s2, sentinel2_l2a_mask=s2_mask)
    return ts


def test_get_embedding_pooled_returns_embedding(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())

    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )

    fake_model = object()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (fake_model, {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert isinstance(out, Embedding)
    assert out.data.ndim == 1
    assert out.data.dtype == np.float32
    assert out.meta["model"] == "olmoearth"
    assert "variant" in out.meta
    assert "hf_repo" in out.meta


def test_get_embedding_grid_returns_dataarray(monkeypatch):
    xr = pytest.importorskip("xarray")
    emb = oe.OlmoEarthEmbedder()

    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    fake_model = object()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (fake_model, {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
    )

    assert isinstance(out, Embedding)
    assert isinstance(out.data, xr.DataArray)
    assert set(out.data.dims) == {"d", "y", "x"}
    assert out.meta["grid_kind"] == "spatial_tokens"


def test_get_embedding_uses_model_config_variant(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 8, 768),
    )
    seen = {}

    def _fake_load(variant, device):
        seen["variant"] = variant
        return object(), {"hf_repo": "allenai/OlmoEarth-v1-Base"}, "cpu"

    monkeypatch.setattr(oe, "_load_olmoearth", _fake_load)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        model_config={"variant": "base"},
    )

    assert seen["variant"] == "base"
    assert out.meta["variant"] == "base"
    assert out.meta["model_size"] == "base"
    assert out.meta["model_version"] == "v1"


def test_get_embedding_accepts_input_chw(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    x_chw = np.full((12, 64, 64), 2000.0, dtype=np.float32)
    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=x_chw,
    )

    assert isinstance(out, Embedding)
    assert out.data.ndim == 1


def test_get_embedding_wrong_input_chw_raises(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    with pytest.raises(ModelError, match="12 bands"):
        emb.get_embedding(
            spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
            temporal=TemporalSpec.year(2022),
            sensor=None,
            output=OutputSpec.pooled(),
            backend="gee",
            input_chw=np.zeros((6, 32, 32), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# get_embeddings_batch — mocked paths
# ---------------------------------------------------------------------------


def test_get_embeddings_batch_prefetch_and_dispatch(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )

    captured = {}

    def _fake_batch_from_inputs(**kwargs):
        captured["input_chws"] = kwargs["input_chws"]
        return [
            Embedding(data=np.array([float(i)], dtype=np.float32), meta={})
            for i in range(len(kwargs["spatials"]))
        ]

    monkeypatch.setattr(emb, "get_embeddings_batch_from_inputs", _fake_batch_from_inputs)

    spatials = [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(3)]
    out = emb.get_embeddings_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    # Inputs should be raw DN arrays (not pre-normalized)
    assert len(captured["input_chws"]) == 3
    for arr in captured["input_chws"]:
        assert arr.shape == (12, 64, 64)


def test_get_embeddings_batch_from_inputs_pooled(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    call_count = {"n": 0}

    def _fake_encoder_forward(model, sample, *, patch_size, device):
        batch_size = sample.sentinel2_l2a.shape[0]
        call_count["n"] += batch_size
        return _fake_encoder_output(batch_size, 4, 128)

    monkeypatch.setattr(oe, "_encoder_forward", _fake_encoder_forward)

    spatials = [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(4)]
    input_chws = [np.full((12, 64, 64), float(i + 1) * 500, dtype=np.float32) for i in range(4)]

    out = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 4
    assert call_count["n"] == 4
    for emb_out in out:
        assert isinstance(emb_out, Embedding)
        assert emb_out.data.ndim == 1
        assert emb_out.data.shape == (128,)


def test_get_embeddings_batch_from_inputs_grid(monkeypatch):
    pytest.importorskip("xarray")
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(sample.sentinel2_l2a.shape[0], 4, 128),
    )

    spatials = [PointBuffer(lon=0.0, lat=0.0, buffer_m=256)]
    input_chws = [np.full((12, 64, 64), 2000.0, dtype=np.float32)]

    out = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.grid(),
        backend="gee",
    )

    import xarray as xr

    assert len(out) == 1
    assert isinstance(out[0].data, xr.DataArray)
    assert out[0].data.dims == ("d", "y", "x")


def test_get_embeddings_batch_from_inputs_length_mismatch_raises():
    emb = oe.OlmoEarthEmbedder()
    with pytest.raises(ModelError, match="length mismatch"):
        emb.get_embeddings_batch_from_inputs(
            spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=256)],
            input_chws=[],
            temporal=TemporalSpec.year(2022),
            output=OutputSpec.pooled(),
            backend="gee",
        )


def test_get_embeddings_batch_returns_empty_for_empty_spatials():
    emb = oe.OlmoEarthEmbedder()
    out = emb.get_embeddings_batch(
        spatials=[],
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="gee",
    )
    assert out == []


# ---------------------------------------------------------------------------
# Meta fields
# ---------------------------------------------------------------------------


def test_embedding_meta_has_required_fields(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=5.0, lat=10.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    meta = out.meta
    assert meta["model"] == "olmoearth"
    assert meta["type"] == "on_the_fly"
    assert meta["backend"] == "gee"
    assert meta["variant"] == "nano"
    assert meta["model_size"] == "nano"
    assert meta["model_version"] == "v1"
    assert meta["patch_size"] == oe._DEFAULT_PATCH_SIZE
    assert meta["hf_repo"] is not None
    assert "temporal_range" in meta
    assert "image_size" in meta
