"""Explicit configuration for constrained device training."""
from dataclasses import dataclass, field
import math
import torch

@dataclass
class TrainingConfig:
    optimizer: str = 'slsqp'
    device: str = 'cpu'
    dtype: str = 'float64'
    maxiter: int = 500
    lr: float = 1e-3
    seed: int = 42
    ftol: float = 1e-9
    feasibility_tol: float = 1e-7
    batch_size: int = 2048
    record_every: int = 1
    checkpoint_every: int = 10
    profile_compute: bool = False
    optimizer_options: dict = field(default_factory=dict)
    constraint_options: dict = field(default_factory=dict)

    def __post_init__(self):
        self.optimizer=self.optimizer.lower().replace('-', '')
        if self.optimizer not in {'slsqp','adam','adamw','sgd','lbfgs'}: raise ValueError('Unknown optimizer.')
        if self.dtype not in {'float32','float64'}: raise ValueError('dtype must be float32 or float64.')
        if torch.device(self.device).type not in {'cpu','cuda'}: raise ValueError('Only CPU/CUDA devices are supported.')
        for name in ['maxiter','seed','batch_size','record_every','checkpoint_every']:
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<(0 if name in {'maxiter','seed'} else 1): raise ValueError('Invalid '+name)
        for name in ['lr','ftol','feasibility_tol']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError('Invalid '+name)
        if self.dtype=='float32' and self.feasibility_tol<1e-6:
            raise ValueError('float32 requires feasibility_tol >= 1e-6.')
        if set(self.constraint_options)-{'outer_steps','penalty','penalty_growth'}: raise ValueError('Unknown constraint option.')
        if self.constraint_options.get('outer_steps',5)<1 or self.constraint_options.get('penalty',10.)<=0 or self.constraint_options.get('penalty_growth',10.)<1: raise ValueError('Invalid augmented Lagrangian settings.')

    @property
    def torch_dtype(self): return getattr(torch,self.dtype)

    def check_device(self):
        if torch.device(self.device).type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; select CPU explicitly.')
