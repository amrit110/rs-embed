import rs_embed.api as api
from rs_embed.core.specs import FetchSpec, OutputSpec, PointBuffer
from rs_embed.core.types import ExportConfig, ExportTarget, ModelConfig


def test_export_batch_normalizes_combined_target_and_config(monkeypatch, tmp_path):
    captured = {}

    def _fake_resolve_export_model_configs(**kwargs):
        captured["resolve_kwargs"] = kwargs
        return [ModelConfig(name="mock_model", backend="auto", model_type="mock")], {
            "mock_model": "auto"
        }

    def _fake_run(self):
        captured["spatials"] = self.spatials
        captured["target"] = self.target
        captured["config"] = self.config
        return {"ok": True}

    monkeypatch.setattr(api, "_resolve_export_model_configs", _fake_resolve_export_model_configs)
    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    out = api.export_batch(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=10)],
        temporal=None,
        models=["mock_model"],
        target=ExportTarget.combined(str(tmp_path / "one")),
        config=ExportConfig(save_inputs=False, save_embeddings=False, save_manifest=False),
    )

    assert out == {"ok": True}
    assert captured["config"].format == "npz"
    assert captured["spatials"]
    assert isinstance(captured["target"], ExportTarget)
    assert captured["target"].out_file.endswith(".npz")


def test_export_batch_passes_fetch_to_model_resolution(monkeypatch, tmp_path):
    captured = {}

    def _fake_resolve_export_model_configs(**kwargs):
        captured.update(kwargs)
        return [ModelConfig(name="mock_model", backend="auto", model_type="mock")], {
            "mock_model": "auto"
        }

    def _fake_run(self):
        return {"ok": True}

    monkeypatch.setattr(api, "_resolve_export_model_configs", _fake_resolve_export_model_configs)
    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    out = api.export_batch(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=10)],
        temporal=None,
        models=["mock_model"],
        target=ExportTarget.combined(str(tmp_path / "one")),
        fetch=FetchSpec(scale_m=20),
    )

    assert out == {"ok": True}
    assert captured["fetch"] == FetchSpec(scale_m=20)


def test_export_batch_respects_configured_export_format(monkeypatch, tmp_path):
    captured = {}

    def _fake_resolve_export_model_configs(**kwargs):
        return [ModelConfig(name="mock_model", backend="auto", model_type="mock")], {
            "mock_model": "auto"
        }

    def _fake_run(self):
        captured["target"] = self.target
        captured["config"] = self.config
        return {"ok": True}

    monkeypatch.setattr(api, "_resolve_export_model_configs", _fake_resolve_export_model_configs)
    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    api.export_batch(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=10)],
        temporal=None,
        models=["mock_model"],
        target=ExportTarget.combined(str(tmp_path / "one")),
        config=ExportConfig(format="netcdf"),
        output=OutputSpec.pooled(),
    )

    assert captured["config"].format == "netcdf"
    assert captured["target"].out_file.endswith(".nc")
