import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.pipeline import fetch_kepler_llc_from_archive


def test_fetch_kepler_llc_from_archive_uses_local_cache_first(tmp_path):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()

    test_file = cache_dir / 'local_test_file_llc.fits'
    test_file.write_bytes(b'not a real fits file')

    result = fetch_kepler_llc_from_archive(
        target_count=1,
        download_dir=str(cache_dir),
        max_buckets=1,
        randomize=False,
        random_seed=0,
        exclude_filenames=set(),
    )

    assert not result.empty
    assert result['filename'].tolist() == ['local_test_file_llc.fits']
    assert result['local_path'].tolist() == [str(test_file.resolve())]
