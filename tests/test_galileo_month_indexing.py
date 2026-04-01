import importlib
import sys
import types

import numpy as np

from rs_embed.core.specs import TemporalSpec

sys.modules.setdefault("xarray", types.SimpleNamespace(DataArray=object))
gal = importlib.import_module("rs_embed.embedders.onthefly_galileo")
_frame_month_sequence = gal._frame_month_sequence
_month_from_iso = gal._month_from_iso


def test_galileo_month_from_iso_is_zero_based():
    assert _month_from_iso("2022-01-15") == 0
    assert _month_from_iso("2022-06-15") == 5
    assert _month_from_iso("2022-12-15") == 11


def test_galileo_frame_month_sequence_stays_within_embedding_range():
    months = _frame_month_sequence(
        TemporalSpec.range("2022-01-01", "2023-01-01"),
        n_frames=8,
    )

    assert months.dtype == np.int64
    assert months.shape == (8,)
    assert int(months.min()) >= 0
    assert int(months.max()) <= 11
    assert 11 in months.tolist()
