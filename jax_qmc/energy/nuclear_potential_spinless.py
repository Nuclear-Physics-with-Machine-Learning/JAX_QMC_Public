import jax.numpy as numpy
from jax import jit, vmap
from functools import partial

from . hamiltonian import kinetic_energy, kinetic_energy_jf

# These are globals in this file that get picked from for the hamiltonian.
v0r_dict = {
    2 : -133.3431,
    4 : -487.6128,
    6 : -1064.5010,
}
v0s_dict = {
    2 : -9.0212,
    4 : -17.5515,
    6 : -26.0830,
}
ar3b_dict = {
    2 : 8.27577,
    4 : 26.03457,
    6 : 51.50389,
}

mode = 4

output = {
    "hbar"   : numpy.asarray(197.327,            dtype = dtype),
    "alpha"  : numpy.asarray(1./137.03599,       dtype = dtype),
    "v0r"    : numpy.asarray(v0r_dict[mode],     dtype = dtype),
    "v0s"    : numpy.asarray(v0s_dict[mode],     dtype = dtype),
    "ar3b"   : numpy.asarray(ar3b_dict[mode],    dtype = dtype),
    "vkr"    : numpy.asarray(mode,               dtype = dtype),
}

from dataclasses import dataclass, field



@dataclass(frozen=True)
class h_params_template:
    # why? it's passed as a static argument in a partial function closure so it needs to be hashable
    # in practice, that means immutable (frozen=true)
    mass:   float
    hbar:   float
    alpha:  float
    v0r:    float
    v0s:    float
    ar3b:   float

h_params = h_params_template(
    mass    =  938.95,
    hbar    = 197.327,
    alpha   = 1./137.03599,
    v0r     = v0r[mode],
    v0s     = v0s[mode],
    ar3b    = ar3b[mode],
)



# @partial(jit, static_argnums=1)
@jit
def pionless_2b(r_ij):

    x = h_params.vkr * r_ij

    vr = numpy.exp(-x**2. / 4.0)

    return h_params.v0r*vr, h_params.v0s*vr

@jit
def pionless_3b(r_ij):

    x = h_params.vkr * r_ij
    vr = numpy.exp(-x**2. / 4.0)

    pot_3b = vr * h_params.ar3b
    return pot_3b

@partial(jit, static_argnums=0)
def potential_energy(wavefunction, w_params, x, spin, isospin, w_of_x):
    """Return potential energy

    Calculate and return the PE.

    """
    # Potential calculation

    # Prepare buffers for the output:
    # (Walker shape is (self.nwalkers, self.nparticles, self.n) )
    nwalkers   = x.shape[0]
    nparticles = x.shape[1]

    if nparticles == 2:
        alpha = 1.0
    else:
        alpha = -1.0

    pairs    = generate_possible_swaps(n_particles)


    # The potential is ultimately just _one_ number per walker.
    # But, to properly compute the 3-body term, we use a per-particle
    # accumulater (gr3b) for each walker.
    gr3b = numpy.zeros(shape=[nparticles, nwalkers,], dtype=x.dtype)
    V_ijk = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # three body potential terms
    v_ij  = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # 2 body potential terms:


    # We need to flatten this loop.


    # Here we compute the pair-wise interaction terms
    for pair in pairs:

        i = pair[0]; j = pair[1];
        # Difference vector between the two particles:
        x_ij = x[:,i,:] - x[:,j,:]


        r_ij = numpy.sqrt(numpy.sum(x_ij**2,axis=1))

        vrr, vrs = pionless_2b(r_ij)

        v_ij = v_ij + vrr + alpha*vrs


        if (nparticles > 2 ):
            t_ij = pionless_3b(r_ij)

            # Compute the 3 particle component which runs cyclically
            gr3b = gr3b.at[i].add(t_ij)
            gr3b = gr3b.at[j].add(t_ij)

            V_ijk = V_ijk - t_ij**2

    # stack up gr3b:
    V_ijk += 0.5 * numpy.sum(gr3b**2, axis = 0)

    pe = v_ij + V_ijk

    # print(pe)
    return pe



from functools import partial
from jax import jit
@partial(jit, static_argnums=6)
def compute_energy(x, spin, isospin, w_of_x, dw_dx, d2w_dx2, wavefunction, w_params):


    # Potential energy depends only on the wavefunction, no derivatives
    pe = potential_energy(wavefunction, w_params,
        x=x, spin=spin, isospin=isospin, w_of_x = w_of_x,
    )

    # KE by parts needs only one derivative
    ke_jf = kinetic_energy_jf(w_of_x=w_of_x, dw_dx=dw_dx,
        M=mass, HBAR=hbar)

    # True, directly, uses the second derivative
    ke_direct = kinetic_energy(w_of_x = w_of_x, d2w_dx2 = d2w_dx2,
        M=mass, HBAR=hbar)


    energy_jf = pe + ke_jf
    energy    = pe + ke_direct


    return {
        "energy"    : energy,
        "energy_jf" : energy_jf,
        "ke_jf"     : ke_jf,
        "ke_direct" : ke_direct,
        "pe"        : pe
    }
