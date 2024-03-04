from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from typing import List, Tuple, Union

@dataclass
class MLPConfig():
    layers:     List[int] = field(default_factory=lambda: [16,])
    bias:            bool = False
    last_activation: bool = False


@dataclass
class DeepSetsCfg():
    individual_cfg: MLPConfig = field(default_factory= lambda : MLPConfig(layers=[16,16,16]))
    aggregate_cfg:  MLPConfig = field(default_factory= lambda : MLPConfig(layers=[16,16,1 ]))
    active:              bool = True

class AntisymmetryKind(Enum):
    NONE     = 0
    Slater   = 1


@dataclass
class AntiSymmetryCfg():
    form: AntisymmetryKind = AntisymmetryKind.NONE
    active:           bool = True
    mlp_cfg:     MLPConfig = field(default_factory= lambda : MLPConfig(layers=[16,16,1]))

@dataclass
class SlaterCfg(AntiSymmetryCfg):
    form: AntisymmetryKind = AntisymmetryKind.Slater

class GraphKind(Enum):
    NONE      = 0
    Attention = 1

@dataclass
class StatePrep():
    padding: Union[int, None] = 16
    edges:               bool = True

@dataclass
class GraphNetCfg():
    form:       GraphKind = GraphKind.NONE
    state_prep: StatePrep = field(default_factory=lambda: StatePrep())


@dataclass
class GraphAttentionConfig(GraphNetCfg):
    form:     GraphKind = GraphKind.Attention
    heads:    List[int] = field(default_factory= lambda :[8,4,])
    features: List[int] = field(default_factory= lambda :[16,16,])


@dataclass
class ManyBodyCfg():
    mean_subtract:           bool = True
    backflow:                bool = True
    correlator_cfg:   DeepSetsCfg = field(default_factory = lambda: DeepSetsCfg() )
    activation:               str = "gelu"
    confinement:            float = 0.01
    antisymmetry: AntiSymmetryCfg = field(default_factory= lambda : AntiSymmetryCfg())
    graph_net:        GraphNetCfg = field(default_factory = lambda: GraphNetCfg() )
    time_reversal:           bool = False
    mirror:                  bool = False

cs = ConfigStore.instance()

cs.store(group="wavefunction/graph_net", name="none",            node=GraphNetCfg)
cs.store(group="wavefunction/graph_net", name="graph_attention", node=GraphAttentionConfig)

cs.store(group="wavefunction/antisymmetry", name="slater",   node=SlaterCfg)

cs.store(group="wavefunction", name="many_body", node=ManyBodyCfg)
