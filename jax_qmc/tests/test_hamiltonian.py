import jax.numpy as numpy
import jax.random as random

from jax_qmc.spatial      import spatial_initialization
from jax_qmc.energy       import kinetic_energy_jf_log_psi, kinetic_energy_log_psi
from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn
from jax_qmc.spatial      import initialize_spin_until_non_zero, multisplit

import pytest
import time


def test_ke_jf(sampler_config, network_config, seed):
    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")
    key, subkey = random.split(key)

    # Get the wavefunction:
    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    log_w_of_x, sign = wavefunction(w_params, x, spin, isospin)

    multikey, submultikey = multisplit(multikey)

    spin = initialize_spin_until_non_zero(
        multikey, x, spin, isospin, wavefunction, w_params)

    dlogw_dx = g_fn(w_params, x, spin, isospin)

    ke_jf = kinetic_energy_jf_log_psi(dlogw_dx, M=1.0, HBAR=1.0)


    # assert ke.sum() > 0


def test_ke_direct(sampler_config, network_config, seed):

    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")
    key, subkey = random.split(key)

    # Get the wavefunction:

    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    log_w_of_x, sign = wavefunction(w_params, x, spin, isospin)


    multikey, submultikey = multisplit(multikey)
    spin = initialize_spin_until_non_zero(
        submultikey, x, spin, isospin, wavefunction, w_params)

    dlogw_dx = g_fn(w_params, x, spin, isospin)
    d2logw_dx2 = d2_fn(w_params, x, spin, isospin)


    # In the cases without spin where the w_of_x can be zero (not thermalized),
    # Add a small term to ensure we don't divide by 0
    ke_jf = kinetic_energy_jf_log_psi(dlogw_dx, M=1.0, HBAR=1.0)

    ke = kinetic_energy_log_psi(ke_jf, d2logw_dx2, M=1.0, HBAR=1.0)

    # assert (ke.sum() > 0).all()

