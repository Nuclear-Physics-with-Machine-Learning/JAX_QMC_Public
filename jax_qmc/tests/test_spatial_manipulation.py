import jax.numpy as numpy
from jax import random

import pytest
import time

from jax_qmc.spatial import spatial_initialization
from jax_qmc.spatial import select_and_swap_particles_single_walker
from jax_qmc.spatial import multisplit
from jax_qmc.spatial import mean_subtract_walker, mean_subtract_walkers


@pytest.mark.parametrize("n_keys", [1,10,100])
def test_multisplit(n_keys, seed):

    key = random.PRNGKey(int(seed))

    k, s = random.split(key)


    def multisplit_slow(_keys):
        out_k = []
        out_s = []
        for _k in _keys:
            _sk, _ss = random.split(_k)
            out_k.append(_sk)
            out_s.append(_ss)

        return numpy.stack(out_k), numpy.stack(out_s)

    multikeys = random.split(key, 100)

    slow_keys, slow_subkeys = multisplit_slow(multikeys)


    fast_keys, fast_subkeys = multisplit(multikeys)

    assert (slow_keys == fast_keys).all()
    assert (slow_subkeys == fast_subkeys).all()

from jax_qmc.spatial import generate_random_rotation_matrix
def test_rotation_matrix(seed):

    key = random.PRNGKey(int(seed))

    k, s = random.split(key)

    mat1 = generate_random_rotation_matrix(k)
    mat2 = generate_random_rotation_matrix(s)

    assert (mat1 != mat2).all()

    # Test orthogonality of the matrixes:

    ortho_test_1 = mat1 @ mat1.T
    ortho_test_2 = mat1.T @ mat1

    assert numpy.allclose(ortho_test_1, numpy.eye(3))
    assert numpy.allclose(ortho_test_2, numpy.eye(3))

    ortho_test_3 = mat2 @ mat2.T
    ortho_test_4 = mat2.T @ mat2

    assert numpy.allclose(ortho_test_3, numpy.eye(3))
    assert numpy.allclose(ortho_test_4, numpy.eye(3))

from jax_qmc.spatial import random_rotate_all_walkers

def test_walker_rotation(sampler_config, seed):

    n_dim = sampler_config.n_dim
    if n_dim != 3:
        pytest.skip("Can't rotate outside of 3D")

    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")

    r_pre = numpy.sum(x**2, axis=-1)

    x_rotated = random_rotate_all_walkers(walker_keys, x)
    
    assert x.shape == x_rotated.shape

    r_post = numpy.sum(x_rotated**2, axis=-1)

    assert numpy.allclose(r_pre,r_post)


def test_mean_subtract(sampler_config, seed):

    nparticles = sampler_config.n_particles
    if nparticles < 2:
        pytest.skip("Can't swap less than two particles")

    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")

    single_x = x[0]
    single_spin = spin[0]
    single_isospin = isospin[0]

    mean_sub_single = mean_subtract_walker(single_x)
    assert (numpy.abs(mean_sub_single.sum(axis=-2)) < 1e-9).all()

    mean_sub_all = mean_subtract_walkers(x)
    assert (mean_sub_all[0] == mean_sub_single).all()

    assert (numpy.abs(mean_sub_all.sum(axis=-2)) <  1e-9).all()


def test_particle_swap_individual(sampler_config, seed):


    nparticles = sampler_config.n_particles
    if nparticles < 2:
        pytest.skip("Can't swap less than two particles")

    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")

    single_x = x[0]
    single_spin = spin[0]
    single_isospin = isospin[0]

    swapped_x, swapped_spin, swapped_isospin = \
        select_and_swap_particles_single_walker(
            key, single_x, single_spin, single_isospin)

    # Get the same ij by not updating the key
    ij = random.choice(key, nparticles, shape=(2,), replace=False)

    # How to check the state is correct?
    # We can compute the difference between the two states.
    # For all particles that didn't get swapped, it should be the same
    # For all particles that did get swapped, the difference at i == - difference at j


    difference_x = single_x - swapped_x
    difference_spin = single_spin - swapped_spin
    difference_isospin = single_isospin - swapped_isospin


    not_swapped_mask = numpy.ones((nparticles,), bool)
    not_swapped_mask = not_swapped_mask.at[ij[0]].set(False)
    not_swapped_mask = not_swapped_mask.at[ij[1]].set(False)
    swapped_mask = numpy.logical_not(not_swapped_mask)

    # The swapped tensors should have an inner shape of (2,), index easily:
    assert (difference_x[swapped_mask][0] == \
        - difference_x[swapped_mask][1]).all()
    assert (difference_x[not_swapped_mask] == 0.0).all()

    assert (difference_spin[swapped_mask][0] == \
        - difference_spin[swapped_mask][1]).all()
    assert (difference_spin[not_swapped_mask] == 0.0).all()

    assert (difference_isospin[swapped_mask][0] == \
        - difference_isospin[swapped_mask][1]).all()
    assert (difference_isospin[not_swapped_mask] == 0.0).all()

from jax_qmc.spatial import select_and_exchange_spins_single_walker

def test_spin_exchange_individual(sampler_config, seed):


    nparticles = sampler_config.n_particles
    if nparticles < 2:
        pytest.skip("Can't swap less than two particles")

    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")


    single_x = x[0]
    single_spin = spin[0]
    single_isospin = isospin[0]

    swapped_spin = select_and_exchange_spins_single_walker(key, single_spin)

    # Get the same ij by not updating the key
    ij = random.choice(key, nparticles, shape=(2,), replace=False)

    # How to check the state is correct?
    # We can compute the difference between the two states.
    # For all particles that didn't get swapped, it should be the same
    # For all particles that did get swapped, the difference at i == - difference at j

    difference = swapped_spin - single_spin


    not_swapped_mask = numpy.ones((nparticles,), bool)
    not_swapped_mask = not_swapped_mask.at[ij[0]].set(False)
    not_swapped_mask = not_swapped_mask.at[ij[1]].set(False)
    swapped_mask = numpy.logical_not(not_swapped_mask)

    # The swapped tensors should have an inner shape of (2,), index easily:
    assert (difference[swapped_mask][0] == - difference[swapped_mask][1]).all()
    assert (difference[not_swapped_mask] == 0.0).all()


from jax_qmc.spatial import select_and_swap_particles

def test_particle_swap(sampler_config, seed):

    nwalkers = sampler_config.n_walkers

    nparticles = sampler_config.n_particles
    ndim       = sampler_config.n_dim

    if nparticles < 2:
        pytest.skip("Can't swap less than two particles")


    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")

    walker_keys, subkeys = multisplit(walker_keys)

    swapped_x, swapped_spin, swapped_isospin = \
        select_and_swap_particles(subkeys, x, spin, isospin)


    difference_x        = x - swapped_x
    difference_spin     = spin - swapped_spin
    difference_isospin  = isospin - swapped_isospin

    # Get the same ij by not updating the key
    # ij = random.choice(key, nparticles, shape=(nwalkers,2), replace=False)
    # ij = random.choice(key, nparticles, shape=(2,), replace=False)
    ij = numpy.stack([
        random.choice(_k, nparticles, shape=(2,), replace=False)
        for _k in subkeys
    ])
    # How to check the state is correct?
    # We can compute the difference between the two states.
    # For all particles that didn't get swapped, it should be the same
    # For all particles that did get swapped, the difference at i == - difference at j


    not_swapped_mask = numpy.ones((nwalkers,nparticles,), bool)
    not_swapped_mask_x = numpy.ones((nwalkers,nparticles,ndim), bool)
    for i, _ij_pair in enumerate(ij):
        not_swapped_mask = not_swapped_mask.at[i,_ij_pair[0]].set(False)
        not_swapped_mask = not_swapped_mask.at[i,_ij_pair[1]].set(False)
        not_swapped_mask_x = not_swapped_mask_x.at[i,_ij_pair[1],:].set(False)
        not_swapped_mask_x = not_swapped_mask_x.at[i,_ij_pair[1],:].set(False)
    swapped_mask = numpy.logical_not(not_swapped_mask)
    swapped_mask_x = numpy.logical_not(not_swapped_mask_x)

    # print(x[0], "vs", swapped_x[0])
    # print(spin[0], "vs", swapped_spin[0])
    # print(isospin[0], "vs", swapped_isospin[0])

    # print(difference_x[0])

    # print(x[1], "vs", swapped_x[1])
    # print(spin[1], "vs", swapped_spin[1])
    # print(isospin[1], "vs", swapped_isospin[1])

    # print(difference_x[1])

    # print(swapped_mask_x.shape)
    # print(swapped_mask_x[0])

    # print(difference_x.shape)
    # print(difference_x[swapped_mask_x].shape)

    # swapped_difference = difference_x[swapped_mask_x].reshape((nwalkers, ndim))
    # print(swapped_difference.shape)
    # print(swapped_difference)

    assert (difference_x[swapped_mask][0] == \
        - difference_x[swapped_mask][1]).all()
    assert (difference_x[not_swapped_mask] == 0.0).all()

    assert (difference_spin[swapped_mask][0] == \
        - difference_spin[swapped_mask][1]).all()
    assert (difference_spin[not_swapped_mask] == 0.0).all()

    assert (difference_isospin[swapped_mask][0] == \
        - difference_isospin[swapped_mask][1]).all()
    assert (difference_isospin[not_swapped_mask] == 0.0).all()




from jax_qmc.spatial import select_and_exchange_spins

def test_spin_exchange(sampler_config, seed):

    nwalkers = sampler_config.n_walkers

    nparticles = sampler_config.n_particles
    if nparticles < 2:
        pytest.skip("Can't swap less than two particles")


    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")

    walker_keys, subkeys = multisplit(walker_keys)

    swapped_spin = select_and_exchange_spins(subkeys, spin)

    difference = swapped_spin - spin

    # The swap should be one pair per row.  So let's check.
    # The end result ought to be that the difference is 0 in every spot, except
    # one spot could be 2 and then one spot would be -2.


    summed_diff = numpy.sum(difference, axis=-1)
    assert (summed_diff == 0).all()

    # We should have no more than 1 "2" per row:

    plus = (difference == 2).sum(axis=1)
    assert (plus <= 1).all()

    minus = (difference == -2).sum(axis=1)
    assert (minus <= 1).all()


from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn
from jax_qmc.spatial      import initialize_spin_until_non_zero, multisplit

def test_initialize_until_nonzero(sampler_config, network_config, seed):

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)



    n_total_walkers = sampler_config.n_walkers

    key = random.PRNGKey(int(seed))
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    x, spin, isospin = spatial_initialization(walker_keys, sampler_config,  "float64")



    # Get the wavefunction:
    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    log_w_of_x, sign = wavefunction(w_params, x, spin, isospin)
    w_of_x = sign* numpy.exp(log_w_of_x)

    # original_zeros = w_of_x == 0

    key, subkey = random.split(key)
    walker_keys, spin_swap_keys = multisplit(walker_keys)

    new_spin = initialize_spin_until_non_zero(
        spin_swap_keys, x, spin, isospin, wavefunction, w_params)
    log_w_of_x, sign = wavefunction(w_params, x, new_spin, isospin)
    w_of_x = sign* numpy.exp(log_w_of_x)

    final_zeros = w_of_x == 0

    total_zeros = final_zeros.sum()
    assert(total_zeros == 0)

@pytest.mark.parametrize("n_ranks", [4,16])
def test_parallel_initialize_until_nonzero(sampler_config, network_config, seed, n_ranks):



    nwalkers_per_rank = sampler_config.n_walkers

    n_total_walkers = nwalkers_per_rank * n_ranks


    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    walker_keys = random.split(subkey, n_total_walkers)

    # Spin up some initial points:
    x_global, spin_global, isospin_global = \
        spatial_initialization(walker_keys, sampler_config, "float64")


    # Initialize the wavefunction globally:
    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction, g_fn, d2_fn, J_fn = \
        init_jit_and_vmap_nn(key, x_global, spin_global, isospin_global, wf)

    # Initialize the wavefunction to non-zero values:
    log_w_of_x, sign = wavefunction(w_params, x_global, spin_global, isospin_global)
    w_of_x = sign * numpy.exp(log_w_of_x)

    walker_keys, spin_swap_keys = multisplit(walker_keys)
    new_spin_global = initialize_spin_until_non_zero(
        spin_swap_keys, x_global, spin_global, isospin_global, wavefunction, w_params)
    log_w_of_x_global, sign_global = wavefunction(w_params, x_global, new_spin_global, isospin_global)
    w_of_x_global = sign_global * numpy.exp(log_w_of_x_global)


    # Next, split the original keys into N_RANKS, and re-shuffle the spins
    # once per "rank".  Check that when gathered back up, it matches the global
    # values of both the spin and wavefunction

    split_keys     = numpy.split(spin_swap_keys, n_ranks)

    local_xs       = numpy.split(x_global, n_ranks)
    local_spins    = numpy.split(spin_global, n_ranks)
    local_isospins = numpy.split(isospin_global, n_ranks)

    new_local_spins = []
    w_of_x_locals = []

    for x_local, spin_local, isospin_local, keys_local in \
        zip(local_xs, local_spins, local_isospins, split_keys):

        new_spin_local = initialize_spin_until_non_zero(
            keys_local, x_local, spin_local, isospin_local, wavefunction, w_params)

        log_w_of_x_local, sign_local = wavefunction(w_params, x_local, new_spin_local, isospin_local)
        w_of_x_local = sign_local * numpy.exp(log_w_of_x_local)
        new_local_spins.append(new_spin_local)
        w_of_x_locals.append(w_of_x_local)


    # Gather them up and then then compare.

    gathered_spin = numpy.concatenate(new_local_spins)
    assert (new_spin_global == gathered_spin).all()

    gathered_w_of_x = numpy.concatenate(w_of_x_locals)
    print(w_of_x_global)
    print(gathered_w_of_x)
    print(w_of_x_global - gathered_w_of_x)
    assert (numpy.abs(w_of_x_global - gathered_w_of_x) < 1e-6).all()



