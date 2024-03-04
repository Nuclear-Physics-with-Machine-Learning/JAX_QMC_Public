import jax.numpy as numpy
import jax.random as random
import jax

from jax_qmc.config import Sampler, ManyBodyCfg

import pytest
import time

from jax_qmc.spatial      import spatial_initialization, multisplit
from jax_qmc.spatial      import initialize_spin_until_non_zero
from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn

def test_wavefunction_init(sampler_config, network_config, seed):

    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")

    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction_fn, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)



    # w_params, wavefunction_fn, compute_derivatives, compute_jacobian =  \
    #     init_many_body_wf(subkey, x, spin, isospin, sampler_config, network_config)

    log_w_of_x, sign = wavefunction_fn(w_params, x, spin, isospin)
    assert log_w_of_x.shape == (n_walkers,)
    assert sign.shape == (n_walkers,)



from jax_qmc.spatial import select_and_swap_particles

def test_wavefunction_antisymmetry(sampler_config, network_config, seed):

    if sampler_config.n_particles < 2:
        pytest.skip("Can't swap less than two particles")

    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")
    key, subkey = random.split(key)


    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction_fn, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    multikey, spin_init_key = multisplit(multikey)

    spin = initialize_spin_until_non_zero(
        spin_init_key, x, spin, isospin, wavefunction_fn, w_params)


    log_w_of_x, sign = wavefunction_fn(w_params, x, spin, isospin)

    key, subkey = random.split(key)

    keys = random.split(subkey, x.shape[0])
    swapped_x, swapped_spin, swapped_isospin = select_and_swap_particles(keys, x, spin, isospin)

    log_w_swapped, sign_swapped = wavefunction_fn(w_params, \
        swapped_x, swapped_spin, swapped_isospin)

    print(log_w_of_x)
    print(log_w_swapped)
    # print(network_config.antisymmetry.active)
    # assert False

    # for i in range(10):
    #     a_prime = w(inputs, spins, isospins).numpy()
    # By switching two particles, we should have inverted the sign.

    # By switching two particles, we should have inverted the sign.
    if network_config.antisymmetry.active:
        # THis should change sign but maintain magnitude:
        assert numpy.allclose(sign, -sign_swapped)
    else:
        # This should be the same:
        assert numpy.allclose(sign, sign_swapped)

    # Always do this check:
    assert numpy.allclose(numpy.exp(log_w_of_x), numpy.exp(log_w_swapped))


# from jax_qmc.wavefunction import init_many_body_wf_test_functions
from jax_qmc.spatial import select_and_swap_particles
from jax_qmc.wavefunction.deepsets import init_deep_sets
from jax_qmc.wavefunction.utils    import concat_inputs

import flax
from jax import jit, vmap

def test_deep_sets(sampler_config, network_config, seed):

    nwalkers = sampler_config.n_walkers
    nparticles = sampler_config.n_particles
    ndim = sampler_config.n_dim

    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")

    # Create the wavefunction:
    key, subkey = random.split(key)

    import flax.linen as nn
    if network_config.activation == "leaky_tanh":
        from jax_qmc.wavefunction.wavefunction import leaky_tanh
        activation = leaky_tanh
    else:
        activation = getattr(nn, network_config.activation)


    correlator = init_deep_sets(network_config.correlator_cfg, activation)

    merged_inputs = concat_inputs(x, spin, isospin)

    w_params = correlator.init(key, merged_inputs[0])

    wavefunction_fn = jit(vmap(flax.linen.apply(type(correlator).__call__, correlator), in_axes=[None, 0]) )


    correlation = wavefunction_fn(w_params, merged_inputs)
    # correlation = wavefunction_fn(w_params, x, spin, isospin)

    # assert (correlation > 0.0).all()

    # If we have more than one particle, we can swap particles around.  The correlater should NOT change value.

    if sampler_config.n_particles > 1:


        multikey, subkeys = multisplit(multikey)

        swapped_x, swapped_spin, swapped_isospin = \
            select_and_swap_particles(subkeys, x, spin, isospin)

        swapped_merged_inputs = concat_inputs(swapped_x, swapped_spin, swapped_isospin)
        swapped_correlation = wavefunction_fn(w_params, swapped_merged_inputs)

        assert (numpy.abs(correlation - swapped_correlation) < 1e-6).all()


from jax_qmc.spatial import select_and_exchange_spins

def test_antisymmetry_module(sampler_config, network_config, seed):

    nwalkers = sampler_config.n_walkers
    nparticles = sampler_config.n_particles
    ndim = sampler_config.n_dim

    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")

    # w_params, wavefunction_fn, compute_derivatives, compute_jacobian =  \
    #     init_many_body_wf(subkey, x, spin, isospin, sampler_config, network_config)


    # Create the wavefunction:
    key, subkey = random.split(key)

    import flax.linen as nn
    from jax_qmc.wavefunction import init_antisymmetry

    if network_config.activation == "leaky_tanh":
        from jax_qmc.wavefunction.wavefunction import leaky_tanh
        activation = leaky_tanh
    else:
        activation = getattr(nn, network_config.activation)

    antisymmetry_function = init_antisymmetry(network_config.antisymmetry, activation)
    merged_inputs = concat_inputs(x, spin, isospin)

    w_params = antisymmetry_function.init(key, merged_inputs[0])

    wavefunction_fn = jit(vmap(flax.linen.apply(
            type(antisymmetry_function).__call__, 
            antisymmetry_function), 
        in_axes=[None, 0]) )

    sign, log_output = wavefunction_fn(w_params, merged_inputs)


    assert(sign != 0).all()
    assert(sign*numpy.exp(log_output) != 0).all()

    keys = random.split(subkey, x.shape[0])
    swapped_x, swapped_spin, swapped_isospin = select_and_swap_particles(keys, x, spin, isospin)
    merged_swapped = concat_inputs(swapped_x, swapped_spin, swapped_isospin)

    sign_swapped, log_output_swapped = wavefunction_fn(w_params, merged_swapped)


    # By switching two particles, we should have inverted the sign.
    print(network_config.antisymmetry.form)
    print(network_config.antisymmetry.active)
    print(sign)
    print(sign_swapped)
    print(log_output)
    print(log_output_swapped)
    print(log_output_swapped - log_output)
    print((log_output_swapped - log_output) / (log_output))
    if network_config.antisymmetry.active:
        # THis should change sign but maintain magnitude:
        assert numpy.allclose(sign, -sign_swapped)
    else:
        # This should be the same:
        assert numpy.allclose(sign, sign_swapped)



    # Always do this check:
    assert numpy.allclose(numpy.exp(log_output), numpy.exp(log_output_swapped), atol=1e-4)


