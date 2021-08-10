from .penal import Penalizer, PenalizerList
from .elastic_penal import ElasticPenalizer
from .prox_penal import ProxPenalizer
from .scaffold_penal import ScaffoldPenalizer

penalizers = [
    Penalizer, ElasticPenalizer, ProxPenalizer, ScaffoldPenalizer
]
