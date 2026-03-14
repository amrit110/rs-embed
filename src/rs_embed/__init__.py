"""rs-embed — location embeddings from remote-sensing imagery.

This top-level package re-exports the public API surface so that users can
write ``from rs_embed import Model, BBox`` without reaching into subpackages.

See :mod:`rs_embed.api` for the function-based API and :class:`rs_embed.Model`
for the class-based (stateful) interface.
"""

from .core.specs import (
    BBox,
    PointBuffer,
    TemporalSpec,
    SensorSpec,
    OutputSpec,
    InputPrepSpec,
)
from .core.types import (
    ExportConfig,
    ExportLayout,
    ExportModelRequest,
    ExportTarget,
    ModelConfig,
)
from .api import export_batch, get_embedding, get_embeddings_batch, list_models
from .model import Model
from .pipelines.exporter import BatchExporter
from .inspect import inspect_gee_patch, inspect_provider_patch
from .export import export_npz

__all__ = [
    # Specs
    "BBox",
    "PointBuffer",
    "TemporalSpec",
    "SensorSpec",
    "OutputSpec",
    "InputPrepSpec",
    # Types
    "ExportConfig",
    "ExportLayout",
    "ExportModelRequest",
    "ExportTarget",
    "ModelConfig",
    # Embedding API (class-based)
    "Model",
    "BatchExporter",
    # Embedding API (function-based, backward compat)
    "get_embedding",
    "get_embeddings_batch",
    "list_models",
    # Export API
    "export_batch",
    "export_npz",
    # Inspection
    "inspect_provider_patch",
    # Backward-compatible alias for inspect_provider_patch
    "inspect_gee_patch",
]
__version__ = "0.1.0"
