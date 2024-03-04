from enum import Enum
from typing import Optional

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import List, Tuple, Union

class Solver(Enum):
    Cholesky = 0
    ConjugateGradient = 1

# These lowercase Optimizers are from optax
# They are matched to optax optimizers by name, so
# if adding another it's gotta match.
# The parameter names here should match optax too

# They can be used by themselves, or they can be used as the regularization
# Component in the second order optimization

@dataclass
class LRSchedule:
    '''
    This is to configure a learning rate schedule with Optax
    Optax is fancy but this is not, only linear are supported 

    The point of this is to save run time by not requiring restarts
    to modify delta or epsilon
    '''
    init_values: List[float] = field(default_factory = lambda : [1e-3, 1e-3, 3e-4, 3e-4, 5e-5])
    end_values:  List[float] = field(default_factory = lambda : [1e-3, 3e-4, 3e-4, 5e-5, 5e-5])
    steps:         List[int] = field(default_factory = lambda : [1000,  500, 1000,  500, 1000])


cs = ConfigStore.instance()

@dataclass
class Optimizer:
    # b1 is the rms prop decay.  Set to 0 to not store this state
    b1:                float = 0.9
    # b2 is the momentum term.  Set to 0 to not use momentum
    b2:                float = 0.0
    # Solver is Cholesky or CG, set to None to use 1st order grads
    # solver: Optional[Solver] = Solver.Cholesky
    solver: Optional[Solver] = Solver.ConjugateGradient
    # Adaptive will take the gradients determined above and vary the learning rate
    # to reach the best next step
    adaptive:           bool = False
    # delta is the learning rate
    delta:        LRSchedule = field(default_factory = lambda: LRSchedule())
    # Epsilon is the regularization for the 2nd order matrix inversion
    epsilon:      LRSchedule = field(default_factory = lambda: LRSchedule())
    # Optionally, toggle a prediction of the next energy state:
    predict_energy:     bool = False
    # Optional, make snapshots every step of the input to the optimizer for each rank.
    make_snapshots:     bool = False


cs.store(group="optimizer", name="optimizer",         node=Optimizer)
