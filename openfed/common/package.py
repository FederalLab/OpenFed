from typing import Dict, List, Union


def _check_state_keys(obj, keys: Union[str, List[str]], mode: str):
    keys = [keys] if isinstance(keys, str) else keys

    keys = keys if keys else getattr(obj, mode, None)

    return keys


class Package(object):
    # 提供打包和解包statedict的能力
    state: Dict

    def pack_state(self, obj, keys: Union[str, List[str]] = None):
        """将obj中的state根据指定的key，pack到对应的数据流中。
        """
        keys = _check_state_keys(obj, keys, mode='package_key_list')
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: state[k] for k in keys}
                    self.pack(p, rdict)

    def unpack_state(self, obj, keys: Union[str, List[str]] = None):
        keys = _check_state_keys(obj, keys, mode="unpackage_key_list")
        if keys:
            for group in obj.param_groups:
                for p in group["params"]:
                    state = obj.state[p]
                    rdict = {k: None for k in keys}
                    rdict = self.unpack(p, rdict)
                    state.update(rdict)
