import jax.numpy as numpy
from jax import jit, vmap
from jax.lax import scan

import math

from . hamiltonian import kinetic_energy_log_psi, kinetic_energy_jf_log_psi
from jax_qmc.spatial import generate_possible_swaps

# These are globals in this file that get picked from for the hamiltonian.
C01_dict = {
    'a' : -4.38524414,
    'b' : -5.72220536,
    'c' : -7.00250932,
    'd' : -8.22926713,
    'o' : -5.27518671,
}
C10_dict = {
    'a' : -8.00783936,
    'b' : -9.34392090,
    'c' : -10.7734100,
    'd' : -12.2993164,
    'o' : -7.04040080,
}
R0_dict = {
    'a' : 1.7,
    'b' : 1.9,
    'c' : 2.1,
    'd' : 2.3,
    'o' : 1.54592984,
}
R1_dict = {
    'a' : 1.5,
    'b' : 2.0,
    'c' : 2.5,
    'd' : 3.0,
    'o' : 1.83039397
}

from dataclasses import dataclass, field

@dataclass(frozen=True)
class h_params_template:
    # why? it's passed as a static argument in a partial function closure so it needs to be hashable
    # in practice, that means immutable (frozen=true)
    mass:   float
    hbar:   float
    alpha:  float
    C01:    float
    C10:    float
    R0:     float
    R1:     float
    b:      float
    c_pref: float
    R3:     float
    alpha_3body: float

################# Here is where to set the mode:
mode = "o"
mass =  938.95

ce3b = 1.2945
fpi  = 92.4
pi   = 3.14159

hbar = 197.327
R3   = 1.1
alpha_3body = math.sqrt( (ce3b/ 1000. / (fpi)**4) * ((hbar)**6 / (pi**3 * R3**6) ) )




h_params = h_params_template(
    mass    = mass,
    hbar    = hbar,
    alpha   = 1./137.03599,
    C01     = C01_dict[mode],
    C10     = C10_dict[mode],
    R0      = R0_dict[mode],
    R1      = R1_dict[mode],
    b       = 4.27,
    c_pref  = 5.568327996831708,
    R3      = R3,
    alpha_3body = alpha_3body,
)


# @partial(jit, static_argnums=1)
@jit
def pionless_2b(r_ij):

    # logger.info("pionless_2b")
    # These C functions are equation 2.7 from https://arxiv.org/pdf/2102.02327.pdf
    c0_r = (1./(h_params.c_pref * h_params.R0**3 )) * \
        numpy.exp(- numpy.power((r_ij / h_params.R0), 2))
    c1_r = (1./(h_params.c_pref * h_params.R1**3 )) * \
        numpy.exp(- numpy.power((r_ij / h_params.R1), 2))

    # Computing several functions here (A26 to A29 in https://arxiv.org/pdf/2102.02327.pdf):
    v_c         = h_params.hbar *  (3./16.) * \
        (     h_params.C01 * c1_r +     h_params.C10 * c0_r)
    v_sigma     = h_params.hbar *  (1./16.) * \
        (-3.* h_params.C01 * c1_r +     h_params.C10 * c0_r)
    v_tau       = h_params.hbar *  (1./16.) * \
        (     h_params.C01 * c1_r - 3.* h_params.C10 * c0_r)
    v_sigma_tau = h_params.hbar * -(1./16.) * \
        (     h_params.C01 * c1_r +     h_params.C10 * c0_r)

    return v_c, v_sigma, v_tau, v_sigma_tau

@jit
def pionless_3b(r_ij):
    # logger.info("pionless_3b")
    x = r_ij / h_params.R3
    vr = numpy.exp(-x**2)
    pot_3b = vr * h_params.alpha_3body
    return pot_3b

@jit
def potential_em(r_ij):
    r_m = numpy.maximum(r_ij, 0.0001)
    br  = h_params.b * r_m
    f_coul = 1 - (1 + (11./16.)*br + (3./16)*numpy.power(br,2) + \
        (1./48)*numpy.power(br,3))*numpy.exp(-br)
    return h_params.alpha * h_params.hbar * f_coul / r_m

from jax_qmc.spatial import exchange_spins_same_spin_all_walkers
from jax import tree_util

def close_over_energy(config, wavefunction):

    # @partial(jit, static_argnums=(6,))
    @jit
    def potential_pairwise_single(w_params, logw_of_x, sign, x, spin, isospin, pair):
        # Difference in ij coordinates:
        i = pair[0]; j = pair[1];
        # Difference vector between the two particles:
        x_ij = x[:,i,:] - x[:,j,:]

        # Slicing like this ^ leaves x as shape [n_walkers, n_dim]

        # Take the magnitude of that difference across dimensions
        r_ij = numpy.sqrt(numpy.sum(x_ij**2,axis=1))
        # Now, r_ij should be a float valued vector of shape [n_walkers]


        # Compute the Vrr and Vrs terms for this pair of particles:
        v_c, v_sigma, v_tau, v_sigma_tau = pionless_2b(r_ij=r_ij)

        v_em = potential_em(r_ij=r_ij)

        # Now, we need to exchange the spin and isospin of this pair of particles

        swapped_spin    = exchange_spins_same_spin_all_walkers(spin, pair)
        swapped_isospin = exchange_spins_same_spin_all_walkers(isospin, pair)


        # Compute the wavefunction under all these exchanges:
        logw_of_x_swap_spin,    s_s  = wavefunction(w_params, x, swapped_spin, isospin)
        logw_of_x_swap_isospin, s_i  = wavefunction(w_params, x, spin,         swapped_isospin)
        logw_of_x_swap_both ,   s_si = wavefunction(w_params, x, swapped_spin, swapped_isospin)

        # Now compute several ratios:
        ratio_swapped_spin    = numpy.exp(logw_of_x_swap_spin    - logw_of_x) * sign * s_s
        ratio_swapped_isospin = numpy.exp(logw_of_x_swap_isospin - logw_of_x) * sign * s_i
        ratio_swapped_both    = numpy.exp(logw_of_x_swap_both    - logw_of_x) * sign * s_si 

        spin_factor     = numpy.reshape(2*ratio_swapped_spin - 1,      (-1,))
        isospin_factor  = numpy.reshape(2*ratio_swapped_isospin - 1,   (-1,))
        both_factor     = numpy.reshape(4*ratio_swapped_both - 2*ratio_swapped_spin - 2*ratio_swapped_isospin + 1,  (-1,))


        # Em force only applies to protons, so apply that:
        proton = (1./4)*(1 + isospin[:,i])*(1 + isospin[:,j])

        # We accumulate the pairwise interaction of these two nucleons:
        v_ij = v_c
        v_ij += v_sigma     * spin_factor
        v_ij += v_tau       * isospin_factor
        v_ij += v_sigma_tau * both_factor
        v_ij += v_em        * proton

        t_ij = pionless_3b(r_ij)

        return v_ij, t_ij

    # Vectorize this but only over the last axis, the pairs:
    potential_pairwise = vmap(potential_pairwise_single,
            in_axes=(
                None, # w_params
                None, # logw_of_x
                None, # sign
                None, # x
                None, # spin
                None, # isospin
                0,     # pair
            )
        )


    # @partial(jit, static_argnums=(0,))
    @jit
    def potential_energy(w_params, x, spin, isospin, logw_of_x, sign):
        """Return potential energy

        Calculate and return the PE.

        """
        # Potential calculation

        # Prepare buffers for the output:
        # (Walker shape is (self.nwalkers, self.nparticles, self.n) )
        nwalkers   = x.shape[0]
        nparticles = x.shape[1]
        pairs    = generate_possible_swaps(nparticles)


        # The potential is ultimately just _one_ number per walker.
        # But, to properly compute the 3-body term, we use a per-particle
        # accumulater (gr3b) for each walker.
        gr3b = numpy.zeros(shape=[nparticles, nwalkers,], dtype=x.dtype)
        V_ijk = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # three body potential terms
        v_ij  = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # 2 body potential terms:

        # Here we compute the pair-wise interaction terms
        for i_pair in range(len(pairs)):

            pair = pairs[i_pair]

            # Compute the single potential:
            this_v_ij, t_ij = potential_pairwise_single(
                w_params, logw_of_x, sign, x, spin, isospin, pair)

            v_ij = v_ij + this_v_ij

            if (nparticles > 2 ):
                i = pair[0]
                j = pair[1]

                # Compute the 3 particle component which runs cyclically
                gr3b = gr3b.at[i].add(t_ij)
                gr3b = gr3b.at[j].add(t_ij)

                V_ijk = V_ijk - t_ij**2

        # stack up gr3b:
        V_ijk += 0.5 * numpy.sum(gr3b**2, axis = 0)

        pe = v_ij + V_ijk

        return pe



    # @partial(jit, static_argnums=(0,))
    @jit
    def potential_energy_new(w_params, x, spin, isospin, logw_of_x, sign):
        """Return potential energy

        Calculate and return the PE.

        """
        # Potential calculation

        # Prepare buffers for the output:
        # (Walker shape is (self.nwalkers, self.nparticles, self.n) )
        nwalkers   = x.shape[0]
        nparticles = x.shape[1]
        pairs    = generate_possible_swaps(nparticles)


        # The potential is ultimately just _one_ number per walker.
        # But, to properly compute the 3-body term, we use a per-particle
        # accumulater (gr3b) for each walker.
        gr3b = numpy.zeros(shape=[nparticles, nwalkers,], dtype=x.dtype)
        # V_ijk = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # three body potential terms
        # v_ij  = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # 2 body potential terms:


        all_v_ij, all_t_ij = potential_pairwise(w_params, logw_of_x, sign, x, spin, isospin, pairs)

        # gr3b is of shape [N_part, N_walkers], and to fill it we loop
        # over all pairs of particles (k) and fill gr3b[i] and gr3b[j] with t_i_ij[k],
        # where i = pairs[k][0] and j = pairs[k][1]
        # We can do this by slicing:
        p0 = pairs[:,0]
        p1 = pairs[:,1]
        gr3b = gr3b.at[p0].add(all_t_ij)
        gr3b = gr3b.at[p1].add(all_t_ij)

        # Sum this over pairs:
        V_ijk = -numpy.sum(all_t_ij**2, axis=0)

        # Sum this over particles:
        V_ijk += 0.5 * numpy.sum(gr3b**2, axis = 0)

        # Reduce over pairs:
        v_ij = numpy.sum(all_v_ij, axis=0)


        pe = v_ij + V_ijk

        return pe


    @jit
    def potential_energy_scan(w_params, x, spin, isospin, logw_of_x, sign):
        """Return potential energy

        Calculate and return the PE.

        """
        # Potential calculation

        # Prepare buffers for the output:
        # (Walker shape is (self.nwalkers, self.nparticles, self.n) )
        nwalkers   = x.shape[0]
        nparticles = x.shape[1]
        pairs    = generate_possible_swaps(nparticles)

        n_pairs  = len(pairs)

        # all_vt_ij = numpy.zeros((nwalkers, n_pairs,2))
        # all_t_ij = numpy.zeros((nwalkers, pairs))

        def scan_fun(inputs, pair):

            v_ij, t_ij = potential_pairwise_single(
                inputs["w_params"], 
                inputs["logw_of_x"], 
                inputs["sign"], 
                inputs["x"], 
                inputs["spin"], 
                inputs["isospin"], pair)

            return inputs, numpy.stack([v_ij, t_ij])

        scan_fun = jit(scan_fun, donate_argnums=0)

        inputs = {
            "w_params"  : w_params,
            "logw_of_x" : logw_of_x,
            "sign"      : sign,
            "x"         : x,
            "spin"      : spin,
            "isospin"   : isospin,
        }

        # The potential is ultimately just _one_ number per walker.
        # But, to properly compute the 3-body term, we use a per-particle
        # accumulater (gr3b) for each walker.
        gr3b = numpy.zeros(shape=[nparticles, nwalkers,], dtype=x.dtype)
        # V_ijk = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # three body potential terms
        # v_ij  = numpy.zeros(shape=[nwalkers,], dtype=x.dtype) # 2 body potential terms:

        _, all_vt_ij = scan(
            f      = scan_fun,
            init   = inputs,
            xs     = pairs,
            unroll = config.hamiltonian.unroll,
        )
        # print(all_vt_ij)
        # print(all_vt_ij.shape)
        all_v_ij = all_vt_ij[:,0,:]
        all_t_ij = all_vt_ij[:,1,:]

        # all_v_ij, all_t_ij = potential_pairwise(w_params, w_of_x, x, spin, isospin, pairs)

        # gr3b is of shape [N_part, N_walkers], and to fill it we loop
        # over all pairs of particles (k) and fill gr3b[i] and gr3b[j] with t_i_ij[k],
        # where i = pairs[k][0] and j = pairs[k][1]
        # We can do this by slicing:
        p0 = pairs[:,0]
        p1 = pairs[:,1]
        gr3b = gr3b.at[p0].add(all_t_ij)
        gr3b = gr3b.at[p1].add(all_t_ij)

        # Sum this over pairs:
        V_ijk = -numpy.sum(all_t_ij**2, axis=0)

        # Sum this over particles:
        V_ijk += 0.5 * numpy.sum(gr3b**2, axis = 0)

        # Reduce over pairs:
        v_ij = numpy.sum(all_v_ij, axis=0)


        pe = v_ij + V_ijk

        return pe


    @jit
    def compute_energy(x, spin, isospin, logw_of_x, sign, dlogw_dx, d2logw_dx2, w_params):

        # Potential energy depends only on the wavefunction, no derivatives

        # # Reduced precision version:
        # precision = numpy.float32

        # w_params_reduced = tree_util.tree_map(
        #     lambda _x: _x.astype(precision), w_params)

        # # Reduced precision:
        # pe = potential_energy_scan(
        #     w_params_reduced,
        #     x           = x.astype(precision), 
        #     spin        = spin.astype(precision), 
        #     isospin     = isospin.astype(precision), 
        #     logw_of_x   = logw_of_x.astype(precision), 
        #     sign        = sign.astype(precision)
        # )

        # Full precision version:
        pe = potential_energy_scan(w_params,
            x=x, spin=spin, isospin=isospin, 
            logw_of_x = logw_of_x, sign=sign
            )

        # KE by parts needs only one derivative
        ke_jf = kinetic_energy_jf_log_psi(dlogw_dx=dlogw_dx,
            M=h_params.mass, HBAR=h_params.hbar)

        # Get the the KE:
        ke_direct = kinetic_energy_log_psi(d2logw_dx2 = d2logw_dx2, ke_jf = ke_jf,
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

    # This is the end of the closure:
    return compute_energy