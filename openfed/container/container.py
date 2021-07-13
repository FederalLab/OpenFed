from openfed.utils import glue

from .agg import Agg
from .reducer import Reducer


class Container(Agg, Reducer):
    """Glue Agg and Reducer into a single class, named Container.
    """


def build_container(aggregator: Agg, reducer: Reducer) -> Container:
    return glue(aggregator, reducer)
