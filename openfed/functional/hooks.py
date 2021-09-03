# Copyright (c) FederalLab. All rights reserved.
import torch
from openfed.core.const import DefaultMaintainer


def device_alignment():
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    def package(state, p):
        for k, v in state.items():
            if v is not None:
                state[k] = v.to(p)
        return state

    _default_maintainer.register_package_hook(nice=100, package_hook=package)

    def unpackage(state, p):
        for k, v in state.items():
            if v is not None:
                state[k] = v.to(p)
        return state

    _default_maintainer.register_unpackage_hook(nice=0,
                                                unpackage_hook=unpackage)


def sign_gradient_clip(epsilon=0.001):
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if _default_maintainer.leader:

        def unpackage(state, p):
            state['param'] = p - epsilon * state['param']
            return state

        _default_maintainer.register_unpackage_hook(nice=80,
                                                    unpackage_hook=unpackage)
    else:

        def package(state, p):
            assert 'original_param' in state
            state['param'] = torch.sign(state['param'] -
                                        state['original_param'])
            return state

        _default_maintainer.register_package_hook(nice=20,
                                                  package_hook=package)
