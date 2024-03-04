import jax.numpy as numpy

from . hamiltonian import kinetic_energy_jf_log_psi, kinetic_energy_log_psi
from dataclasses import dataclass, field

@dataclass(frozen=True)
class h_params_template:
    # why? it's passed as a static argument in a partial function closure so it needs to be hashable
    # in practice, that means immutable (frozen=true)
    mass:   float
    hbar:   float
    omega:  float

h_params = h_params_template(
    mass  = 1.0,
    hbar  = 1.0,
    omega = 1.0,
)

def close_over_energy(config, wavefunction):

    def potential_energy(x, M, omega):

        """Return potential energy for the harmonic oscillator

        Calculate and return the PE.

        Arguments:
            x {} -- Tensor of shape [N, npart, dimension]
            M {} -- Mass, floating point
            omega {} -- HO Omega term, floating point
        Returns:
            DeviceArray - potential energy of shape [N]
        """

        # Potential calculation
        # < x | H | psi > / < x | psi > = < x | 1/2 w * x**2  | psi > / < x | psi >  = 1/2 w * x**2
        # print("Enter pe call")

        # x Squared needs to contract over spatial dimensions:
        x_squared = numpy.sum(x**2, axis=(1, 2))
        pe = (0.5 * M * omega**2 ) * x_squared

        return pe





    def compute_energy(x, spin, isospin, logw_of_x, sign, dlogw_dx, d2logw_dx2, w_params=None):

        n_particles = x.shape[1]

        # Potential energy depends only on the wavefunction
        pe = potential_energy(x=x, M=h_params.mass, omega=h_params.omega)

        # KE by parts needs only one derivative
        ke_jf = kinetic_energy_jf_log_psi(dlogw_dx=dlogw_dx,
            M=h_params.mass, HBAR=h_params.hbar)

        # True, directly, uses the second derivative
        ke_direct = kinetic_energy_log_psi(ke_jf = ke_jf, d2logw_dx2 = d2logw_dx2,
            M=h_params.mass, HBAR=h_params.hbar)



        energy_jf = pe + ke_jf
        energy    = pe + ke_direct


        return {
            "energy"    : energy,
            "energy_jf" : energy_jf,
            "ke_jf"     : ke_jf,
            "ke_direct" : ke_direct,
            "pe"        : pe
        }

    return compute_energy
