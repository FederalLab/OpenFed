import time

from openfed import Meta


def test_meta():
    meta = Meta()
    assert 'mode' in meta
    assert 'version' in meta
    assert hasattr(meta, 'mode')
    assert hasattr(meta, 'version')

    meta.tic = time.time()
    time.sleep(0.1)
    meta['toc'] = time.time()

    duration = meta.toc - meta['tic']

    assert duration > 0.0

    del meta.tic  # type: ignore
    del meta['toc']
