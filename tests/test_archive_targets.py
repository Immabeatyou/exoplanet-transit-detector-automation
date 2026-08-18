import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import web.app as app_module


def test_fetch_archive_targets_tops_up_partial_cache(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        if kwargs["randomize"] is False:
            return pd.DataFrame({"filename": ["cached.fits"]})
        return pd.DataFrame({"filename": ["cached.fits", "remote.fits"]})

    monkeypatch.setattr(app_module, "fetch_kepler_llc_from_archive", fake_fetch)

    result = app_module.fetch_archive_targets(2, str(tmp_path))

    assert result["filename"].tolist() == ["cached.fits", "remote.fits"]
    assert calls[1]["target_count"] == 1
    assert calls[1]["exclude_filenames"] == {"cached.fits"}