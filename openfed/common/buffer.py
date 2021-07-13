from typing import List, Any
from openfed.utils import convert_to_list


class Buffer(object):
    """Used for Optimizer/Aggregator format class to clear buffer
    """
    param_groups: Any
    state: Any

    def clear_buffer(self, keep_keys: List[str] = None):
        """Clear state buffers.
        Args:
            keep_keys: if not specified, we will directly remove all buffers.
                Otherwise, the key in keep_keys will be kept.
        """
        keep_keys = convert_to_list(keep_keys)

        for group in self.param_groups:
            if 'keep_keys' in group:
                if keep_keys is None:
                    keys = group['keep_keys']
                elif group['keep_keys'] is None:
                    keys = keep_keys
                else:
                    keys = keep_keys + group['keep_keys']
            else:
                keys = keep_keys

            for p in group["params"]:
                if p in self.state[p]:
                    if keys is None:
                        del self.state[p]
                    else:
                        for k in self.state[p].keys():
                            if k not in keys:
                                del self.state[p][k]
