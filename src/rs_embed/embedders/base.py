from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import FetchResult
from ..providers.base import ProviderBase


def _method_accepts_parameter(obj: Any, method_name: str, param_name: str) -> bool:
    fn = getattr(type(obj), method_name, None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    if param_name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


class EmbedderBase:
    """Base interface for all embedder implementations.

    Subclasses implement model-specific inference while keeping a common call
    signature used by higher-level API and pipeline code.
    """

    model_name: str = "base"
    _allow_auto_backend: bool = True

    def __init__(self) -> None:
        self._providers: dict[str, ProviderBase] = {}

    def _get_provider(self, backend: str) -> ProviderBase:
        from .runtime_utils import get_cached_provider

        return get_cached_provider(
            self._providers, backend=backend, allow_auto=self._allow_auto_backend
        )

    def describe(self) -> dict[str, Any]:
        """Return model capabilities and input/output requirements.

        Returns
        -------
        dict[str, Any]
            Metadata describing supported backends, outputs, and temporal hints.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete embedder subclasses.
        """
        raise NotImplementedError

    def fetch_input(
        self,
        provider: ProviderBase,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
    ) -> FetchResult | None:
        """Fetch model input with model-specific logic.

        The default implementation returns ``None``, signaling callers to
        use the generic provider fetch path.  Subclasses override this to
        own their input semantics (fallback chains, sensor-specific
        filters, fetch-time metadata).

        Both ``get_embedding()`` and ``export_batch()`` call this method,
        ensuring identical fetch behavior across single and batch paths.

        Parameters
        ----------
        provider : ProviderBase
            Ready provider instance.
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec or None
            Optional temporal filter.
        sensor : SensorSpec
            Sensor/source definition.

        Returns
        -------
        FetchResult or None
            Model-specific fetch result, or ``None`` to fall back to
            generic fetch.
        """
        return None

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: np.ndarray | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> Embedding:
        """Compute a single embedding.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec or None
            Optional temporal filter.
        sensor : SensorSpec or None
            Optional sensor override for provider-backed models.
        model_config : dict[str, Any] or None
            Optional model-specific settings such as architecture variant.
        output : OutputSpec
            Requested output layout.
        backend : str
            Backend/provider selector.
        device : str
            Target inference device.
        input_chw : np.ndarray or None
            Optional prefetched CHW input to bypass provider fetch.

        Returns
        -------
        Embedding
            Embedding payload and metadata.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete embedder subclasses.
        """

        raise NotImplementedError

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        """Default batch implementation: loop over spatials.

        Embedders that can do true batching (e.g. torch models) should override.

        Parameters
        ----------
        spatials : list[SpatialSpec]
            Spatial requests to process.
        temporal : TemporalSpec or None
            Optional temporal filter applied to all inputs.
        sensor : SensorSpec or None
            Optional sensor override.
        model_config : dict[str, Any] or None
            Optional model-specific settings.
        output : OutputSpec
            Requested output layout.
        backend : str
            Backend/provider selector.
        device : str
            Target inference device.

        Returns
        -------
        list[Embedding]
            Embeddings in the same order as ``spatials``.
        """
        out: list[Embedding] = []
        for s in spatials:
            kwargs: dict[str, Any] = {
                "spatial": s,
                "temporal": temporal,
                "sensor": sensor,
                "output": output,
                "backend": backend,
                "device": device,
            }
            if model_config is not None and _method_accepts_parameter(
                self,
                "get_embedding",
                "model_config",
            ):
                kwargs["model_config"] = model_config
            out.append(self.get_embedding(**kwargs))
        return out

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        """Batch inference with prefetched CHW inputs.

        Default implementation keeps existing behavior by looping through
        get_embedding(..., input_chw=...).
        Embedders that can do true batched model forward should override.

        Parameters
        ----------
        spatials : list[SpatialSpec]
            Spatial requests aligned with ``input_chws``.
        input_chws : list[np.ndarray]
            Prefetched CHW arrays, one per spatial.
        temporal : TemporalSpec or None
            Optional temporal filter.
        sensor : SensorSpec or None
            Optional sensor override.
        model_config : dict[str, Any] or None
            Optional model-specific settings.
        output : OutputSpec
            Requested output layout.
        backend : str
            Backend/provider selector.
        device : str
            Target inference device.

        Returns
        -------
        list[Embedding]
            Embeddings in the same order as inputs.

        Raises
        ------
        ValueError
            If ``spatials`` and ``input_chws`` lengths differ.
        """
        if len(spatials) != len(input_chws):
            raise ValueError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        out: list[Embedding] = []
        for s, x in zip(spatials, input_chws, strict=False):
            kwargs: dict[str, Any] = {
                "spatial": s,
                "temporal": temporal,
                "sensor": sensor,
                "output": output,
                "backend": backend,
                "device": device,
                "input_chw": x,
            }
            if model_config is not None and _method_accepts_parameter(
                self,
                "get_embedding",
                "model_config",
            ):
                kwargs["model_config"] = model_config
            out.append(self.get_embedding(**kwargs))
        return out
