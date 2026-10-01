"""Paper-grounded RIMER; additive API, independent of legacy BRBModel."""
from .config import RIMERAttribute,RIMERConfig
from .model import RIMERModel
from .results import RIMERResult
from .hierarchy import RIMERNode,HierarchicalRIMER
from .trainer import RIMERTrainer
from .objectives import RIMERSupervision
