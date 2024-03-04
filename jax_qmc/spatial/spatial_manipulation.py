import jax.numpy as numpy
from jax import random

from jax import jit, vmap

@jit
def multisplit(keys):
    out = jit(vmap(random.split, in_axes=(0,)))(keys)
    return out[:,0,:], out[:,1,:]


@jit
def swap_particles_single_walker(x, spin, isospin, ij):
    # Switch exactly two particles, i != j:

    new_x = x.at[ij[0],:].set(x[ij[1],:])
    new_x = new_x.at[ij[1],:].set(x[ij[0],:])

    new_spin = spin.at[ij[0]].set(spin[ij[1]])
    new_spin = new_spin.at[ij[1]].set(spin[ij[0]])

    new_isospin = isospin.at[ij[0]].set(isospin[ij[1]])
    new_isospin = new_isospin.at[ij[1]].set(isospin[ij[0]])

    return new_x, new_spin, new_isospin

@jit
def exchange_spins_single_walker(spin_vector, ij):
    new_spin  = spin_vector.at[ij[0]].set(spin_vector[ij[1]])
    new_spin = new_spin.at[ij[1]].set(spin_vector[ij[0]])
    return new_spin

exchange_spins = jit(vmap(exchange_spins_single_walker, in_axes=(0,0)))

exchange_spins_same_spin_all_walkers = \
    jit(vmap(exchange_spins_single_walker, in_axes=(0, None)))

# Define the jit'd and vmap'd version of these functions:
swap_particles = jit(vmap(swap_particles_single_walker, in_axes=(0,0,0,0)))

@jit
def select_and_swap_particles_single_walker(key, x, spin, isospin):
    """
    Swap two random particles, based on the

    :param      key:      The random key
    :param      state:    The walker state (just one configuration)
    """
    # We expect a single walker configuration here:
    nparticles = x.shape[0]
    # Generate the swap, without replacement:
    ij = random.choice(key, nparticles, shape=(2,), replace=False)

    # Call the individual swap:
    return swap_particles_single_walker(x, spin, isospin, ij)

select_and_swap_particles = jit(vmap(select_and_swap_particles_single_walker))

@jit
def select_and_exchange_spins_single_walker(key, spin):
    """
    swap two random spin components:

    :param      key:   The key
    :param      spin:  The spin
    """
    nparticles = spin.shape[-1]
    ij = random.choice(key, nparticles, shape=(2,), replace=False)

    return exchange_spins_single_walker(spin, ij)

select_and_exchange_spins = jit(vmap(select_and_exchange_spins_single_walker))


def generate_possible_swaps(n_particles):
    swap_i = []
    swap_j = []
    max_index = n_particles
    i = 0
    while i < n_particles:
        for j in range(i + 1, max_index):
            swap_i.append(i)
            swap_j.append(j)
        i += 1

    swap_i = numpy.asarray(swap_i, dtype="int32")
    swap_j = numpy.asarray(swap_j, dtype="int32")


    swaps =  numpy.stack([swap_i, swap_j], axis=1)
    return swaps


import logging
logger = logging.getLogger()
import time

def initialize_spin_until_non_zero(keys, x, spin, isospin, wavefunction, w_params):

    if spin.shape[-1] < 2:
        return spin

    assert keys.shape[0] == x.shape[0]

    start = time.time()

    # TODO: the jitting of this function is problematic.
    # For a large number of walkers, it's very slow.
    # A solution might be to not jit it directly and just let it
    # churn through, since it's a one-time cost.

    @jit
    def shuffle_and_swap_if_zero_single_spin(_key, _w_of_x, _spin):

        is_zero = numpy.equal(_w_of_x, numpy.zeros_like(_w_of_x))

        permuted_spin = random.permutation(_key, _spin, axis=0, independent=True)

        return numpy.where(is_zero, permuted_spin, _spin), is_zero

    shuffle_and_swap_if_zero = jit(vmap(shuffle_and_swap_if_zero_single_spin))

    @jit
    def shuffle_and_recompute_zero_spins(_keys, _spin):
        # Wavefunction value this time:
        log_w_of_x, sign = wavefunction(w_params, x, _spin, isospin)
        w_of_x = sign* numpy.exp(log_w_of_x)

        # How many walkers?
        n_walkers = _spin.shape[0]

        new_spin, zeros = shuffle_and_swap_if_zero(_keys, w_of_x, _spin)

        n_zeros = numpy.sum(zeros.astype("float"))
        # print("new_spin in function: ", new_spin)
        return new_spin, n_zeros



    # # Create a local-scope function here to use function-scope items.
    # # We JIT this and use it below:
    # @jit
    # def shuffle_and_recompute_zero_spins(_key, _spin):
    #     w_of_x = wavefunction(w_params, x, _spin, isospin)

    #     n_walkers = _spin.shape[0]

    #     # Where are the zeros?
    #     zero_configuration_indexes = numpy.equal(w_of_x, numpy.zeros_like(w_of_x))
    #     # zero_configuration_indexes = find_non_zero(w_of_x)
    #     # zero_configuration_indexes = numpy.zeros(n_walkers, dtype="bool")
    #     # How many zeros?
    #     n_zero = numpy.sum(zero_configuration_indexes.astype("float"))

    #     # This shuffles ALL the spins, but returns a copy.
    #     exchanged_zero_spins = shuffle_all_spins(random.split(_key, n_walkers), _spin)
    #     # exchanged_zero_spins = _spin
    #     #
    #     # # Compute the wavefunction again - don't actually need this!
    #     # w_of_x_exchanged = wavefunction(w_params, x, exchanged_zero_spins, isospin)

    #     # (Why not compute on the reduced set of spins?  JAX would re-JIT with a new shape)

    #     # Use where to select the new spins, with the assumption that some aren't zero:
    #     # (Edge case handling: when none are zero, this should select only the un-exchanged spins)
    #     _spin = numpy.where(zero_configuration_indexes.reshape((-1,1)), exchanged_zero_spins, _spin)

    #     return _spin, n_zero


    # This is just a loop until non-zero:
    subkeys, keys = multisplit(keys)
    spin, n_zero = shuffle_and_recompute_zero_spins(subkeys, spin)

    i = 0
    while n_zero > 0:
        subkeys, keys = multisplit(keys)
        spin, n_zero = shuffle_and_recompute_zero_spins(subkeys, spin)
        # print("New Spin: ", spin)
        # exit()
        if i % 10 == 0:
            logger.info(f"Spin swap finished {i} iterations with {n_zero} zeros left to solve.")
        i += 1
    #
    # # Insert a check here:
    # w_of_x = wavefunction(w_params, x, spin, isospin)
    # if (w_of_x == 0).any():
    #     raise Exception("Failed to initialize all spin walkers to non-zero states")

    end = time.time()
    logger.info(f"Spin swaps completed in {end - start:.4f} seconds.")
    return spin

@jit
def mean_subtract_walker(walker):
    # Calculate the center of mass over all particles
    mean = walker.mean(axis=0)

    # Remove the center of mass, reintroduce the particle dimension first:
    return walker - mean.reshape((1,-1))

mean_subtract_walkers = jit(vmap(mean_subtract_walker, in_axes=(0)))


# construct a rotation matrix:
@jit
def construct_rotation_matrix(alpha, beta, gamma):

    def yaw(alpha):
        ca = numpy.cos(alpha)
        sa = numpy.sin(alpha)
        return numpy.asarray([
            [ca, -sa, 0.],
            [sa,  ca, 0.],
            [0.,  0., 1.],
        ])

    def pitch(beta):
        cb = numpy.cos(beta)
        sb = numpy.sin(beta)
        return numpy.asarray([
            [cb,  0., sb],
            [0.,  1., 0.],
            [-sb, 0., cb],
        ])

    def roll(gamma):
        cg = numpy.cos(gamma)
        sg = numpy.sin(gamma)
        return numpy.asarray([
            [1., 0.,  0.],
            [0., cg, -sg],
            [0,  sg,  cg],
        ])


    matrix = yaw(alpha)@pitch(beta)@roll(gamma)
    return matrix

@jit
def generate_random_rotation_matrix(key):

    alpha, beta, gamma = random.uniform(key, shape=(3,), minval=-numpy.pi, maxval=numpy.pi)
    return construct_rotation_matrix(alpha, beta, gamma)

@jit
def random_rotate_walker(key, x):


    rotation_matrix = generate_random_rotation_matrix(key)

    original_walker_shape = x.shape
    x = x.reshape((-1,3,1))
    rotation_matrix = rotation_matrix.reshape((1,3,3))

    x_rotated = rotation_matrix@x

    return x_rotated.reshape(original_walker_shape)

random_rotate_all_walkers = vmap(random_rotate_walker, in_axes=[0, 0])

random_rotate_all_walkers_same_rotation = vmap(random_rotate_walker, in_axes=[None, 0])