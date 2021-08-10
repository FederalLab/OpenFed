from .agg import Agg
from .average_agg import AverageAgg
from .elastic_agg import ElasticAgg
from .naive_agg import NaiveAgg

aggregators = [
    Agg, ElasticAgg, AverageAgg, NaiveAgg
]
