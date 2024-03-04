from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Potential(Enum):
    NuclearPotential  = 0
    HarmonicOscillator = 1
    AtomicPotential   = 2
    NuclearPotentialSpinless  = 3

@dataclass
class Hamiltonian:
    form: Potential = MISSING

@dataclass
class NuclearHamiltonian(Hamiltonian):
    """
    This class describes a nuclear hamiltonian.
    Total particles is set elsewhere, so only need to specify how many protons
    """
    form: Potential = Potential.NuclearPotential
    unroll:     int = 1

@dataclass
class NuclearHamiltonianSpinless(Hamiltonian):
    """
    This class describes a nuclear hamiltonian.
    Total particles is set elsewhere, so only need to specify how many protons
    """
    form: Potential = Potential.NuclearPotentialSpinless


@dataclass
class AtomicHamiltonian(Hamiltonian):

    form: Potential = Potential.AtomicPotential


@dataclass
class HarmonicOscillatorHamiltonian(Hamiltonian):
    form:  Potential = Potential.HarmonicOscillator


cs = ConfigStore.instance()
cs.store(group="hamiltonian", name="nuclear",  node=NuclearHamiltonian)
cs.store(group="hamiltonian", name="nuclear_spinless",  node=NuclearHamiltonianSpinless)
cs.store(group="hamiltonian", name="atomic",   node=AtomicHamiltonian)
cs.store(group="hamiltonian", name="harmonic", node=HarmonicOscillatorHamiltonian)
