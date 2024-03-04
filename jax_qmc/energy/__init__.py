from .hamiltonian import kinetic_energy_log_psi
from .hamiltonian import kinetic_energy_jf_log_psi


import jax
import jax.numpy as numpy

def build_energy_function(cfg, function_registry):

    # Load the hamiltonian function:
    from .. config import Potential

    if cfg.hamiltonian.form == Potential.NuclearPotential:
        if cfg.sampler.n_particles < 2:
            raise Exception("Can't use the nuclear potential without multiple particles.")
        from . nuclear_potential import close_over_energy


    elif cfg.hamiltonian.form == Potential.NuclearPotentialSpinless:
        if cfg.sampler.n_particles < 2:
            raise Exception("Can't use the nuclear potential without multiple particles.")
        from . nuclear_potential_spinless import close_over_energy

    elif cfg.hamiltonian.form == Potential.AtomicPotential:
        from . atomic_potential import close_over_energy

    elif cfg.hamiltonian.form == Potential.HarmonicOscillator:
        from . harmonic_oscillator_potential import close_over_energy
    else:
        raise Exception("Can't identify the right hamiltonian form.")

    return close_over_energy(cfg, function_registry["wavefunction"])
