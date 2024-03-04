import jax.numpy as numpy
import jax.random as random

from jax import jit, vmap
from jax import tree_util
from jax import lax

from . spatial_manipulation import multisplit, mean_subtract_walkers
from . spatial_manipulation import select_and_exchange_spins
from functools import partial
# original timing was about 2.874 s / iteration

@partial(jit, static_argnums=(1,2))
def multi_uniform(keys, shape, dtype):
    l = jit(vmap(lambda x : random.uniform(x, shape=shape, dtype=dtype)))
    return l(keys)

@partial(jit, static_argnums=(1,2))
def multi_normal(keys, shape, dtype):
    l = jit(vmap(lambda x : random.normal(x, shape=shape, dtype=dtype)))
    return l(keys)

@jit
def compute_point_to_set_distance(point, set2):
    # This is one step to compute the arithmetic average distance.
    # Gets vmap'd over the point to apply this to two distributions

    # point is shape (n_particles, n_dim)
    # set2 is shape (n_walkers, n_particles, n_dim)

    # n_walk = set2.shape[0]
    n_part = set2.shape[1]
    n_dim  = set2.shape[2]

    norm_vec = (point - set2)**2 / (n_dim*n_part)
    difference = numpy.sum(norm_vec, axis=(0,1,2))

    return difference

@jit
def compute_set_to_set_distance(set1, set2):

    n1 = set1.shape[0]
    n2 = set2.shape[0]
    unsummed_dist = jit(vmap(compute_point_to_set_distance, in_axes=(0,None)))(set1, set2)

    return numpy.sum(unsummed_dist) / (n1*n2)

# compute_set_to_set_distance = jit(vmap(compute_point_to_set_distance, in_axes=(0,None)))
# compute_arithmetic_average_distance = lambda x1, x2 : numpy.mean(compute_set_to_set_distance(x1, x2) )

@jit
def energy_distance(set1, set2):
    # Compute the energy distance between two distributions of walkers


    A = compute_set_to_set_distance(set1, set2)
    B = compute_set_to_set_distance(set1, set1)
    C = compute_set_to_set_distance(set2, set2)

    return 2*A - B - C

@jit
def compute_ratio(logw_of_x1, logw_of_x2):

    # Ratio of the wavefunction in the new walkers position:
    probability_walkers = numpy.exp(2 * (logw_of_x1 -  logw_of_x2))
    # print("Probability walkers: ", probability_walkers)
    return probability_walkers

def close_walk_over_wavefunction(wavefunction, precision, mean_subtract=True):
    # Here, we implement the metropolis walk as a private function

    # This creates a closure at runtime over a specific wavefunction object

    # you ought to need to call this again to actually re-JIT

    @jit
    def single_kick(i, kick_state):
        '''
        Kick all the walkers exactly once.
        '''

        ##########################################################
        # Get the prelimary shape / datatype info:
        ##########################################################


        n_particles = kick_state["spin"].shape[-1]
        # First, split the key and update the state:
        split_keys, walker_keys = multisplit(kick_state["keys"])

        # This determines the precision and shape of the kick
        kick_shape = kick_state["x"][0].shape
        kick_dtype = kick_state["x"].dtype

        # Compute the kick, sampling a normal distribution * scale_factor
        kick = kick_state["kick_size"] * multi_normal(split_keys, kick_shape, kick_dtype)
        kicked_walkers = kick_state["x"] + kick

        # print("Kick: ", kick)
        # print("kicked_walkers: ", kicked_walkers)

        # if mean_subtract:
        #     kicked_walkers = mean_subtract_walkers(kicked_walkers)

        # How many walkers  are there?
        n_walkers = kick_state["x"].shape[0]
        n_dim     = kick_state["x"].shape[2]

        ##########################################################
        # Move the walkers only, first, and accept/reject the move:
        ##########################################################

        # Compute the wavefunction of the kicked walkers:
        kicked_wavefunction, kicked_sign = wavefunction(
            kick_state["w_params"], kicked_walkers, kick_state["spin"], kick_state["isospin"])
        # print("Kicked sign: ", kicked_sign)
        # print("Kicked wavefunction:", kicked_wavefunction)
        # print("Orig wavefunction:", kick_state["logpsi"])

        # Ratio of the wavefunction in the new walkers position:
        probability_walkers = compute_ratio(kicked_wavefunction, kick_state["logpsi"])


        # Sample the uniform distribution to check acceptance:
        split_keys, walker_keys = multisplit(walker_keys)
        uniform_numbers = multi_uniform(split_keys, shape=(), dtype=probability_walkers.dtype)

        accept_walkers = probability_walkers >= uniform_numbers

        # Update the acceptance value:
        kick_state["accept_x"] = accept_walkers + kick_state["accept_x"]


        # Store the new configurations with a where function:
        kick_state["logpsi"] = numpy.where(accept_walkers, kicked_wavefunction, kick_state["logpsi"])
        # Need to reshape the accept value to enable broadcasting.
        kick_state["x"]       = numpy.where(
            accept_walkers.reshape(-1, 1, 1), kicked_walkers,  kick_state["x"])

        # # if needed, mean subtract:
        # if mean_subtract:
        #     mean = kick_state["x"].mean(axis=1).reshape((n_walkers, -1, n_dim))
        #     kick_state["x"] = kick_state["x"] - mean

        ##########################################################
        # Now, kick the spin / isospin values second:
        ##########################################################

        # Split the keys and kick the spins
        if n_particles > 1:
            split_keys, walker_keys = multisplit(walker_keys)
            kicked_spins = select_and_exchange_spins(
                split_keys, kick_state["spin"])
        else:
            kicked_spins = kick_state["spin"]

        # Split the key and kick the isospins
        if n_particles > 1:
            split_keys, walker_keys = multisplit(walker_keys)
            kicked_isospins = select_and_exchange_spins(
                split_keys, kick_state["isospin"])
        else:
            kicked_isospins = kick_state["isospin"]

        ##########################################################
        # Compute the wavefunction of the kicked walkers:
        ##########################################################
        kicked_wavefunction_spins, _ = wavefunction(
            kick_state["w_params"], kick_state["x"], kicked_spins, kicked_isospins)


        # Probability of acceptance is the ratio of wave functions squared
        probability_spins = compute_ratio(kicked_wavefunction_spins, kick_state["logpsi"])

        # Compute ratio of wave functions squared in log space:

        # Accept if the prob is higher than a random uniform number:
        split_keys, walker_keys = multisplit(walker_keys)
        accept_spins = probability_spins >= multi_uniform(split_keys, shape=(), dtype=probability_spins.dtype)

        # Store the new configurations with a where function:
        kick_state["logpsi"] = numpy.where(accept_spins, kicked_wavefunction_spins, kick_state["logpsi"])

        # Need to reshape the accept_spins value to enable broadcasting.
        if n_particles > 1:
            kick_state["spin"]    = numpy.where(
                accept_spins.reshape(-1, 1),    kicked_spins,    kick_state["spin"])
            kick_state["isospin"] = numpy.where(
                accept_spins.reshape(-1, 1),    kicked_isospins, kick_state["isospin"])

        # Repack the state:
        kick_state["keys"] = walker_keys

        # Acceptance becomes a moving sum:
        kick_state["accept_spin"] = accept_spins.mean() + kick_state["accept_spin"]


        return kick_state

    # @jit
    def metropolis_walk(kicker_keys, parameters, kick_size, x, spin, isospin, nkicks):

        # Store the split into the state:

        original_precision = x.dtype

        if mean_subtract:
            x = mean_subtract_walkers(x)

        # Drop the weights and inputs into reduced precision:
        x_reduced = lax.stop_gradient(x).astype(precision)

        spin_reduced = lax.stop_gradient(spin).astype(precision)
        isospin_reduced = lax.stop_gradient(isospin).astype(precision)
        params_reduced = tree_util.tree_map(
            lambda _x: _x.astype(precision), parameters)
        kick_size = kick_size.astype(precision)



        # The sign is irrelevant in the walk, so throw it away
        logpsi, sign = wavefunction(params_reduced, x_reduced, spin_reduced, isospin_reduced)

        # print("Original logpsi: ", logpsi)
        # print("Original sign: ", sign)
        logpsi, sign = wavefunction(parameters, x, spin, isospin)
        # print("Original logpsi: ", logpsi)
        # print("Original sign: ", sign)

        # Build the initial kick state:
        kick_state = {}
        kick_state["w_params"]  = params_reduced
        kick_state["kick_size"] = kick_size
        kick_state["logpsi"]       = logpsi
        kick_state["x"]         = x_reduced
        kick_state["spin"]      = spin_reduced
        kick_state["isospin"]   = isospin_reduced
        kick_state["keys"]      = kicker_keys

        # Acceptance becomes a moving sum:
        kick_state["accept_x"]    = numpy.zeros((x.shape[0],), dtype=precision)
        kick_state["accept_spin"] = numpy.zeros((), dtype=precision)

        # unif = multi_uniform(kicker_keys, shape=(), dtype=x.dtype)
        # norm = multi_normal(kicker_keys, shape=(), dtype=x.dtype)

        # print("unif: ", numpy.min(unif), numpy.mean(unif), numpy.max(unif), numpy.std(unif))
        # print("norm: ", numpy.min(norm), numpy.mean(norm), numpy.max(norm), numpy.std(norm))

        # print("Kick State: ", kick_state['accept_x'])
        final_state = lax.fori_loop(lower=0, upper=nkicks, body_fun=single_kick, init_val=kick_state)

        # print("Final state: ", final_state['accept_x'])
        # # What's the energy difference between the initial and final states?
        # edist = energy_distance(x, final_state["x"].astype(original_precision))
        # print(f"Walked edist is: {edist:.6f}")
        # mean = final_state["x"].mean(axis=1)
        # print(mean)
        # print(mean.shape)
        # final_state2 = lax.fori_loop(lower=0, upper=nkicks, body_fun=single_kick, init_val=final_state)

        # # What's the energy difference between the initial and final states?
        # edist = energy_distance(final_state2["x"].astype(original_precision),
        #                         final_state["x"].astype(original_precision)
        #                     )
        # print(f"Walked edist 2 is: {edist:.6f}")

        # final_state = final_state2
        # Return things back in the original precision:
        x       = final_state["x"].astype(original_precision)
        spin    = final_state["spin"].astype(original_precision)
        isospin = final_state["isospin"].astype(original_precision)
        acceptance = {}
        acceptance['x']    = final_state["accept_x"].mean() / nkicks
        acceptance['spin'] = final_state["accept_spin"] / nkicks


        # We need to return the spatial states and nothing else:
        return acceptance, x, spin, isospin

    # def adaptive_metropolis_walk(walker_keys, parameters, kick_size, x, spin, isospin, nkicks):


    #     original_precision = x.dtype

    #     # Drop the weights and inputs into reduced precision:
    #     x_reduced = x.astype(precision)

    #     spin_reduced = spin.astype(precision)
    #     isospin_reduced = isospin.astype(precision)
    #     params_reduced = tree_util.tree_map(
    #         lambda _x: _x.astype(precision), parameters)
    #     kick_size = kick_size.astype(precision)

    #     logpsi = wavefunction(params_reduced, x_reduced, spin_reduced, isospin_reduced)

    #     # Build the initial kick state:
    #     kick_state = {}
    #     kick_state["w_params"]  = params_reduced
    #     kick_state["kick_size"] = kick_size
    #     kick_state["logpsi"]       = logpsi
    #     kick_state["x"]         = x_reduced
    #     kick_state["spin"]      = spin_reduced
    #     kick_state["isospin"]   = isospin_reduced
    #     kick_state["keys"]      = walker_keys

    #     # Acceptance becomes a moving sum:
    #     kick_state["accept_x"]    = numpy.zeros((), dtype="float32")
    #     kick_state["accept_spin"] = numpy.zeros((), dtype="float32")


    #     final_state = lax.fori_loop(lower=0, upper=nkicks, body_fun=single_kick, init_val=kick_state)


    #     # # What's the energy difference between the initial and final states?
    #     # edist = energy_distance(x, final_state["x"].astype(original_precision))
    #     # print(f"Walked edist is: {edist:.6f}")
    #     # mean = final_state["x"].mean(axis=1)
    #     # print(mean)
    #     # print(mean.shape)
    #     # final_state2 = lax.fori_loop(lower=0, upper=nkicks, body_fun=single_kick, init_val=final_state)

    #     # # What's the energy difference between the initial and final states?
    #     # edist = energy_distance(final_state2["x"].astype(original_precision),
    #     #                         final_state["x"].astype(original_precision)
    #     #                     )
    #     # print(f"Walked edist 2 is: {edist:.6f}")

    #     # final_state = final_state2
    #     # Return things back in the original precision:
    #     x = final_state["x"].astype(original_precision)
    #     spin = final_state["spin"].astype(original_precision)
    #     isospin = final_state["isospin"].astype(original_precision)
    #     acceptance_x = final_state["accept_x"] / nkicks
    #     acceptance_spin = final_state["accept_spin"] / nkicks


    #     # We need to return the spatial states and nothing else:
    #     return acceptance_x, acceptance_spin, x, spin, isospin



    return metropolis_walk
