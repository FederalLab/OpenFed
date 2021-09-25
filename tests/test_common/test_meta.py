# @Author            : FederalLab
# @Date              : 2021-09-25 16:56:38
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:56:38
# Copyright (c) FederalLab. All rights reserved.
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
