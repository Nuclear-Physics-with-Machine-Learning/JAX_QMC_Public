from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

from .wavefunction import ManyBodyCfg
from .hamiltonian  import Hamiltonian
from .optimizer    import Optimizer




@dataclass
class Sampler:
    n_thermalize:              int = 5000
    n_void_steps:              int = 1000
    n_walkers:                 int = 2000
    n_particles:               int = 1
    n_dim:                     int = 3
    n_spin_up:                 int = 1
    n_protons:                 int = 1
    kick_size:               float = 0.2
    walk_precision:            str = "float64"
    global_precision:          str = "float64"

    def __post_init__(self):
        """
        Check that the z projection is physical
        """
        if abs(self.n_spin_up) > self.n_particles:
            raise AttributeError("N spin up particles must be less than or equal to total particles")
        if abs(self.n_protons) > self.n_particles:
            raise AttributeError("N protons particles must be less than or equal to total particles")

cs = ConfigStore.instance()

cs.store(group="sampler", name="sampler", node=Sampler)

cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)


defaults = [
    {"hamiltonian"  : "nuclear"},
    {"optimizer"    : "optimizer"},
    {"wavefunction" : "many_body"},
    {"sampler"      : "sampler"},
    {"wavefunction/antisymmetry" : "slater"},
    {"wavefunction/graph_net"    : "none"},
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    hamiltonian:   Hamiltonian = MISSING
    optimizer:       Optimizer = MISSING
    wavefunction:  ManyBodyCfg = MISSING
    sampler:           Sampler = MISSING

    run_id:       str = MISSING
    iterations:   int = 200
    seed:         int = -1
    profile:     bool = False
    distributed: bool = False
    save_dir:     str = "output/"
    restore_path: Any = None
    model_name:   str = "model"
    save_walkers: bool = False

cs.store(name="base_config", node=Config)
