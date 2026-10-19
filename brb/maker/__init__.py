"""Additive implementation of Yang & Xu (2025), DOI 10.1016/j.artint.2025.104289."""
from .state_space import StateSpace
from .evidence import Evidence,EvidenceTable,JointEvidence
from .results import MAKERResult
from .combination import combine_pair,extend_result,dependency_indices
from .model import MAKERModel
from .training_data import ObservationGroup,MAKERTrainingData
from .objectives import paper_objectives
from .trainer import MAKERTrainer
from .acquisition import EvidenceAcquisition
from .torch_backend import TorchMAKERBackend
from .torch_trainer import TorchMAKERTrainer
