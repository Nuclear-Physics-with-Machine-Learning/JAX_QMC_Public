import jax.numpy as numpy
import jax.random as random

from jax_qmc.spatial import spatial_initialization

import pytest
import time


def test_walker_initialization(sampler_config, seed):

    # Sampler_config is a fixture from conftest.py

    # # Create the sampler config:
    # sampler_config = Sampler()

    nwalkers = sampler_config.n_walkers
    nparticles = sampler_config.n_particles
    ndim = sampler_config.n_dim
    n_spin_up = sampler_config.n_spin_up
    n_protons = sampler_config.n_protons


    key = random.PRNGKey(int(time.time()))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, nwalkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(walker_keys, sampler_config, "float64")

    # Check the shapes:
    assert x.shape == (nwalkers, nparticles, ndim)
    assert spin.shape == (nwalkers, nparticles)
    assert isospin.shape == (nwalkers, nparticles)

    # We should have +1 for n_spin_up, -1 for n_spin_down
    # (similar for n_protons)
    # We can check this by summing these:

    expected_sum = nwalkers * (2*n_spin_up - nparticles)
    assert expected_sum == numpy.sum(spin)

    expected_sum = nwalkers * (2*n_protons - nparticles)
    assert expected_sum == numpy.sum(isospin)

@pytest.mark.parametrize("n_ranks", [4, 16])
def test_walker_distributed_init(sampler_config, seed, n_ranks):

    # We're emulating MPI here.  Start with one key, split to N_TOTAL walkers
    # and then initialize everythng.

    # Artifically ensure we have an even split:
    sampler_config.n_walkers = sampler_config.n_walkers * n_ranks


    nwalkers_per_rank = int(sampler_config.n_walkers / n_ranks)

    n_total_walkers = nwalkers_per_rank * n_ranks


    key = random.PRNGKey(int(time.time()))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    # Spin up some initial points:
    x_global, spin_global, isospin_global = \
        spatial_initialization(walker_keys, sampler_config, "float64")



    # Next, split the original keys into N_RANKS, and re-initialize the walkers
    # once per "rank"

    split_keys = numpy.split(walker_keys, n_ranks)

    local_xs       = []
    local_spins    = []
    local_isospins = []
    for local_keys in split_keys:
        x_local, spin_local, isospin_local = \
            spatial_initialization(local_keys, sampler_config, "float64")
        local_xs.append(x_local)
        local_spins.append(spin_local)
        local_isospins.append(isospin_local)

    # Gather them up and then then compare.

    gathered_x = numpy.concatenate(local_xs)
    assert (x_global == gathered_x).all()

    gathered_spin = numpy.concatenate(local_spins)
    assert (spin_global == gathered_spin).all()

    gathered_isospin = numpy.concatenate(local_isospins)
    assert (isospin_global == gathered_isospin).all()
