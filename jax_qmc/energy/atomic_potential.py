import jax.numpy as numpy
from jax import jit, vmap

from . hamiltonian import kinetic_energy_jf_log_psi, kinetic_energy_log_psi
from dataclasses import dataclass, field

from jax_qmc.spatial import generate_possible_swaps


@dataclass(frozen=True)
class h_params_template:
    # why? it's passed as a static argument in a partial function closure so it needs to be hashable
    # in practice, that means immutable (frozen=true)
    mass:   float
    hbar:   float
    e:      float

h_params = h_params_template(
    mass = 1.0,
    hbar = 1.0,
    e    = 1.0,
)



from jax import jit

def close_over_energy(config, wavefunction):
    
    # @partial(jit, static_argnums=(6,))
    @jit
    def potential_pairwise_single(x, pair):
        # Difference in ij coordinates:
        i = pair[0]; j = pair[1];
        # Difference vector between the two particles:
        x_ij = x[:,i,:] - x[:,j,:]

        # Slicing like this ^ leaves x as shape [n_walkers, n_dim]

        # Take the magnitude of that difference across dimensions
        r_ij = numpy.sqrt(numpy.sum(x_ij**2,axis=1))
        # Now, r_ij should be a float valued vector of shape [n_walkers]

        return 1./(r_ij + 1e-8)
        
    # Vectorize this but only over the last axis, the pairs:
    potential_pairwise = vmap(potential_pairwise_single,
            in_axes=(
                None, # x
                0,    # pair
            )
        )

    def potential_energy(x, Z, ELECTRON_CHARGE):

        # Potential energy is, for n particles, two pieces:
        # Potential from the nucleus:
        # Compute r_i, where r_i = sqrt(sum(x_i^2, y_i^2, z_i^2)) (in 3D)
        # PE_1 = -(Z e^2)/(4 pi eps_0) * sum_i (1/r_i)
        #
        # Potential from electron interactions:
        # Second, compute r_ij, for all i != j, and then
        # PE_2 = -(e^2) / (4 pi eps_0) * sum_{i!=j} (1 / r_ij)
        # where r_ij = sqrt( [xi - xj]^2 + [yi - yj] ^2 + [zi - zj]^2)
        #
        r = numpy.sqrt(numpy.sum(x**2, axis=2))
        # This is the sum of 1/r for all particles with the nucleus:
        pe_1 = - (Z * ELECTRON_CHARGE**2 ) * numpy.sum( 1. / (r + 1e-8), axis=1 )
        
        #nwalkers   = x.shape[0]
        nparticles = x.shape[1]
        pairs    = generate_possible_swaps(nparticles)

        #gr3b = numpy.zeros(shape=[nparticles, nwalkers,], dtype=x.dtype)
        all_r_ij = potential_pairwise(x, pairs)   

        # Sum this over pairs:
        pe_2 = (ELECTRON_CHARGE**2) * numpy.sum(all_r_ij, axis=0 )

        # This is the sum of 1/r for all particles with other particles.
        # n_particles = inputs.shape[1]
        # for i_particle in range(n_particles):
        #     centroid = inputs[:,i_particle,:]
        #
        #     r = tf.math.sqrt(tf.reduce_sum((inputs -centroid)**2, axis=2))
        #     pe_2 = -0.5* (ELECTRON_CHARGE**2 ) * tf.reduce_sum( 1. / (r + 1e-8), axis=1 )
        #     # Because this force is symmetric, I'm multiplying by 0.5 to prevent overflow
        #pe_2 = 0.
        
        return pe_1 + pe_2



    def compute_energy(x, spin, isospin, logw_of_x, sign, dlogw_dx, d2logw_dx2, w_params=None):

        n_particles = x.shape[1]

        # Potential energy depends only on the wavefunction
        pe = potential_energy(x=x, Z=n_particles, ELECTRON_CHARGE=h_params.e)

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
