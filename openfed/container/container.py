from openfed.common import glue

from .agg import Agg
from .reducer import Reducer


class Container(Agg, Reducer):
    """Glue Agg and Reducer into a single class, named Container.

    .. warn::
        Container is an abstract class, it would not be instant.
    """
    def __new__(cls):
        raise RuntimeError("Call `build_container` to create a container.")


def build_container(aggregator: Agg, reducer: Reducer = None) -> Container:
    """Build a container to glue aggregator and reducer.

    Args:
        aggregator: The aggregator to aggregate tensor.
        reducer: The reducer to reduce task information.
    """
    # An empty reducer is needed.
    reducer = Reducer() if reducer is None else reducer
    return glue(aggregator, reducer)
