# Copyright (c) FederalLab. All rights reserved.

from addict import Dict as AttrDict

from openfed.utils import FMT, tablist


class Meta(AttrDict):
    '''A attributed dictionary to delivery message. It contains :attr:`mode` and
    :attr:`version`.

    .. Example::

        >>> meta = Meta()
        >>> meta
        <OpenFed> Meta
        +-------+---------+
        |  mode | version |
        +-------+---------+
        | train |    -1   |
        +-------+---------+

        >>> meta['instance'] = 100
        >>> meta.instance
        100
        >>> meta.accuracy = 0.98
        >>> meta['accuracy']
        0.98
        >>> meta
        <OpenFed> Meta
        +-------+---------+----------+----------+
        |  mode | version | instance | accuracy |
        +-------+---------+----------+----------+
        | train |    -1   |   100    |   0.98   |
        +-------+---------+----------+----------+

        >>> del meta.instance
        >>> del meta['accuracy']
        >>> meta
        <OpenFed> Meta
        +-------+---------+
        |  mode | version |
        +-------+---------+
        | train |    -1   |
        +-------+---------+
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'mode' not in self:
            self.mode = 'train'
        if 'version' not in self:
            self.version = -1

    def __repr__(self):
        head = list(self.keys())
        data = list(self.values())
        description = tablist(head, data, items_per_row=10)

        return FMT.openfed_class_fmt.format(
            class_name=self.__class__.__name__,
            description=description,
        )
