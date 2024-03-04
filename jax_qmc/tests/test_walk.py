import jax.numpy as numpy
import jax.random as random

from jax_qmc.config import Sampler, ManyBodyCfg

import pytest

from jax_qmc.spatial import spatial_initialization
from jax_qmc.spatial import close_walk_over_wavefunction
from jax_qmc.spatial import multisplit, multi_normal, multi_uniform
from jax_qmc.spatial import initialize_spin_until_non_zero
from jax_qmc.spatial import select_and_exchange_spins

from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn

from jax.config import config; config.update("jax_enable_x64", True)

@pytest.mark.parametrize("n_keys", [100])
@pytest.mark.parametrize("dtype", [numpy.float64,numpy.float32,numpy.float16, numpy.bfloat16])
def test_multi_normal(sampler_config, n_keys, seed, dtype):

    key = random.PRNGKey(int(seed))


    shape = (sampler_config.n_particles, sampler_config.n_dim)

    def multi_normal_slow(_keys, shape, dtype):
        out_norm = []
        for _k in _keys:
            _norm = random.normal(_k, shape=shape, dtype=dtype)
            out_norm.append(_norm)

        return numpy.stack(out_norm)

    multikeys = random.split(key, n_keys)

    slow_normal = multi_normal_slow(multikeys, shape, dtype)


    fast_normal = multi_normal(multikeys, shape, dtype)

    assert (slow_normal == fast_normal).all()

@pytest.mark.parametrize("n_keys", [10])
@pytest.mark.parametrize("dtype", [numpy.float64,numpy.float32,numpy.float16, numpy.bfloat16])
def test_multi_uniform(n_keys, seed, dtype):

    if dtype == numpy.float16:
        pytest.skip("Skipping fp16 when fp64 is enabled")

    key = random.PRNGKey(int(seed))

    shape = ()

    def multi_uniform_slow(_keys, shape, dtype):
        out_uniform = []
        for _k in _keys:
            _uniform = random.uniform(_k, shape=shape, dtype=dtype)
            out_uniform.append(_uniform)
            print(_uniform)
        return numpy.stack(out_uniform)

    multikeys = random.split(key, n_keys)

    slow_uniform = multi_uniform_slow(multikeys, shape, dtype)

    fast_uniform = multi_uniform(multikeys, shape, dtype)

    assert (slow_uniform == fast_uniform).all()

def test_wavefunction_with_swapped_spins(sampler_config, network_config, seed):

    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")



    # Get the wavefunction:
    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    logw_of_x, sign = wavefunction(w_params, x, spin, isospin)

    walker_keys, spin_swap_keys = multisplit(walker_keys)

    spin = initialize_spin_until_non_zero(
        spin_swap_keys, x, spin, isospin, wavefunction, w_params)
    logw_of_x, sign = wavefunction(w_params, x, spin, isospin)

    # Create the kick function:




    # If we start at non-zero values, we should finish with non zero values no matter what switches we make:

    # Kick the particles:
    shape = (sampler_config.n_particles, sampler_config.n_dim)
    walker_keys, subkeys = multisplit(walker_keys)
    kick = 0.2*multi_normal(subkeys, shape, dtype=x.dtype)


    #Swap some spins:
    walker_keys, subkeys = multisplit(walker_keys)
    swapped_spin = select_and_exchange_spins(subkeys, spin)

    #Swap some isospins:
    walker_keys, subkeys = multisplit(walker_keys)
    swapped_isospin = select_and_exchange_spins(subkeys, isospin)

    # print("Spin: ", spin)
    # print("Isospin: ", isospin)
    # print("w_of_x: ", w_of_x)

    # Now test that we don't get any zeros:
    logw_of_x_kicked_x, sign = wavefunction(w_params, x + kick, spin, isospin)
    # print("Kicked x: ", w_of_x_kicked_x)
    p_x = numpy.exp(2*(logw_of_x_kicked_x - logw_of_x) )
    # print("Kicked x prob ratio: ", numpy.sum(p_x > 0.5)/ n_total_walkers)
    assert (logw_of_x_kicked_x != 0.0).all()

    logw_of_x_spin, sign = wavefunction(w_params, x, swapped_spin, isospin)

    # print("Spin diff: ", spin - swapped_spin)
    # print("Swapped spin: ", logw_of_x_spin)
    p_spin = numpy.exp(2*(logw_of_x_spin / logw_of_x) )
    # print("swapped_spin prob ratio: ", p_spin )
    # print("spin acceptance: ", numpy.sum(p_spin >= 0.5 ) / n_total_walkers)
    # SOME Particles should move to a higher value:
    assert (p_spin >=  1.0).any()

    logw_of_x_isospin, sign = wavefunction(w_params, x, spin, swapped_isospin)
    # print("Swapped spin: ", logw_of_x_spin)
    p_isospin = numpy.exp(2*(logw_of_x_isospin - logw_of_x))
    # print("swapped_spin prob ratio: ", p_isospin)
    # print("isospin acceptance: ", numpy.sum(p_isospin >= 0.5) / n_total_walkers )
    assert (p_isospin >=  1.0).any()

    logw_of_x_spin_isospin, sign = wavefunction(w_params, x, swapped_spin, swapped_isospin)
    # print("Swapped both: ", w_of_x_spin_isospin)
    p_both = numpy.exp(2*(logw_of_x_spin_isospin - logw_of_x) )
    # print("swapped both prob ratio: ", p_both)
    # print("both acceptance: ", numpy.sum(p_both >= 0.5) / n_total_walkers)
    assert (p_both >=  1.0).any()



def test_metropolis_walk_k(sampler_config, network_config, seed):


    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")


    # Get the wavefunction:
    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    w_of_x = wavefunction(w_params, x, spin, isospin)

    walker_keys, spin_swap_keys = multisplit(walker_keys)

    new_spin = initialize_spin_until_non_zero(
        spin_swap_keys, x, spin, isospin, wavefunction, w_params)

    # Create the kick function:

    metropolis_walk = close_walk_over_wavefunction(wavefunction, "float32")

    walker_keys, kicker_keys = multisplit(walker_keys)

    # Perform a number of kicks on this set:
    acceptance, kicked_x, kicked_spin, kicked_isospin = \
        metropolis_walk(kicker_keys, w_params,
                        numpy.asarray(sampler_config.kick_size),
                        x, spin, isospin, 500)


    assert acceptance["x"] > 0.0
    assert acceptance["x"] < 1.0

    assert acceptance["spin"] > 0.0
    assert acceptance["spin"] <= 1.0

    # Shape is unchanged:
    assert x.shape       == x.shape
    assert spin.shape    == spin.shape
    assert isospin.shape == isospin.shape

    # make sure that some states are different:
    assert (x != kicked_x).any()
    if sampler_config.n_particles > 1:
        # Can't be swapping spins if there's only one particle...
        assert (spin    != kicked_spin).any()
        assert (isospin != kicked_isospin).any()



@pytest.mark.parametrize("n_ranks", [4, 16])
def test_metropolis_walk_parallel(sampler_config, network_config, seed, n_ranks):

    # Artifically ensure we have an even split:
    sampler_config.n_walkers = sampler_config.n_walkers * n_ranks



    nwalkers_per_rank = sampler_config.n_walkers

    n_total_walkers = nwalkers_per_rank * n_ranks


    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    # Spin up some initial points:
    x_global, spin_global, isospin_global = \
        spatial_initialization(walker_keys, sampler_config, "float64")


    # Initialize the wavefunction globally:
        # Get the wavefunction:
    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = \
        init_jit_and_vmap_nn(key, x_global, spin_global, isospin_global, wf)

    # Initialize the wavefunction to non-zero values:
    logw_of_x, sign = wavefunction(w_params, x_global, spin_global, isospin_global)

    walker_keys, spin_swap_keys = multisplit(walker_keys)
    spin_global = initialize_spin_until_non_zero(
        spin_swap_keys, x_global, spin_global, isospin_global, wavefunction, w_params)

    # Create the kick function:
    metropolis_walk = close_walk_over_wavefunction(wavefunction, "float64")

    walker_keys, kicker_keys = multisplit(walker_keys)

    # Perform a number of kicks on this set:
    acceptance, global_kicked_x, global_kicked_spin, global_kicked_isospin = \
        metropolis_walk(kicker_keys, w_params,
                        numpy.asarray(sampler_config.kick_size),
                        x_global, spin_global, isospin_global, 50)



    # Next, split the original keys into N_RANKS, and re-shuffle the spins
    # once per "rank".  Check that when gathered back up, it matches the global
    # values of both the spin and wavefunction


    split_keys     = numpy.split(kicker_keys, n_ranks)

    local_xs       = numpy.split(x_global, n_ranks)
    local_spins    = numpy.split(spin_global, n_ranks)
    local_isospins = numpy.split(isospin_global, n_ranks)

    new_local_xs = []
    new_local_spins = []
    new_local_isospins = []

    for x_local, spin_local, isospin_local, keys_local in \
        zip(local_xs, local_spins, local_isospins, split_keys):

        acceptance, kicked_x, kicked_spin, kicked_isospin = \
            metropolis_walk(keys_local, w_params,
                            numpy.asarray(sampler_config.kick_size),
                            x_local, spin_local, isospin_local, 50)

        new_local_xs.append(kicked_x)
        new_local_spins.append(kicked_spin)
        new_local_isospins.append(kicked_isospin)

    # Gather them up and then then compare.

    gathered_x = numpy.concatenate(new_local_xs)
    assert (global_kicked_x == gathered_x).all()

    gathered_spin = numpy.concatenate(new_local_spins)
    assert (global_kicked_spin == gathered_spin).all()

    gathered_isospin = numpy.concatenate(new_local_isospins)
    assert (global_kicked_isospin == gathered_isospin).all()
