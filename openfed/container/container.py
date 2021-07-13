from openfed.common import glue

from .agg import Agg
from .reducer import Reducer


class Container(Agg, Reducer):
    """Glue Agg and Reducer into a single class, named Container.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_container` to create a container.")


def build_container(aggregator: Agg, reducer: Reducer = None) -> Container:
    reducer = Reducer() if reducer is None else reducer
    return glue(aggregator, reducer)
