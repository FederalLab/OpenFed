# Copyright (c) FederalLab. All rights reserved.

from addict import Dict as AttrDict
from openfed.utils import openfed_class_fmt, tablist


class Meta(AttrDict):
    """A class to store different task inforation, such as instance number, 
    train and test accuracy.

    .. Example::
        >>> task_info = Meta(part_id=1)
        >>> task_info
        {'part_id': 1}
        >>> task_info['accuracy'] = 0.98
        >>> task_info.accuracy
        0.98
        >>> task_info
        {'part_id': 1, 'accuracy': 0.98}
        >>> task_info.instance = 100
        >>> task_info['instance']
        100
        >>> task_info
        {'part_id': 1, 'accuracy': 0.98, 'instance': 100}
        >>> del task_info['instance']
        >>> task_info
        {'part_id': 1, 'accuracy': 0.98}
        >>> del task_info.accuracy
        >>> task_info
        {'part_id': 1}
    
    .. note::
        Meta will have a default attribute of `train`.
        It will be used to control some aggregating process.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a train flag to task info.
        if 'mode' not in self:
            self.mode = 'train'
        if 'version' not in self:
            self.version = '-1'

    def __str__(self):
        return openfed_class_fmt.format(class_name='Meta',
                                        description=tablist(
                                            list(self.keys()),
                                            list(self.values()),
                                            items_per_row=10,
                                        ))
