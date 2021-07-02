from typing import List


class Wrapper(object):
    """Provide some method to wrap a class with support of Package.
    """
    pack_key_list: List = None
    unpack_key_list: List = None

    def add_pack_key(self, key: str):
        assert isinstance(key, str), "Only string format keys are supported."
        if self.pack_key_list is None:
            self.pack_key_list = []
        if key in self.pack_key_list:
            raise KeyError(f"Duplicate key: {key}.")
        self.pack_key_list.append(key)

    def add_unpack_key(self, key: str):
        assert isinstance(key, str), "Only string format keys are supported."
        if self.unpack_key_list is None:
            self.unpack_key_list = []
        if key in self.unpack_key_list:
            raise KeyError(f"Duplicate key: {key}.")
        self.unpack_key_list.append(key)

    def add_pack_key_list(self, keys: List[str]):
        [self.add_pack_key(key) for key in keys]

    def add_unpack_key_list(self, keys: List[str]):
        [self.add_unpack_key(key) for key in keys]
