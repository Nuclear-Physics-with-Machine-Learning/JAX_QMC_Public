import jax.numpy as numpy
from jax import random
from jax import vmap

from jax_qmc.config import Sampler

from jax.config import config; config.update("jax_enable_x64", True)


def initialize_single_walker(key, sampler_config : Sampler, dtype):
    """
    Initizlize a single walker configuration using a random key.

    Why not batch these into one big initialization?  The idea is to
    have exact reproducability with strong scaling, so one key per walker.
    """

    size = ( sampler_config.n_particles, sampler_config.n_dim)
    spin_size = (sampler_config.n_particles)

    # The key is CONSUMED here but we split it to get fresh values for each time.
    subkey, key = random.split(key)
    walkers = random.normal(subkey, shape=size, dtype=dtype)

    spin    = initialize_spin_vector(spin_size, sampler_config.n_spin_up, dtype)
    isospin = initialize_spin_vector(spin_size, sampler_config.n_protons, dtype)

    subkey, key = random.split(key)
    spin = random.permutation(subkey, spin, axis=(-1), independent=True)

    subkey, key = random.split(key)
    isospin = random.permutation(subkey, isospin, axis=(-1), independent=True)

    return walkers, spin, isospin

spatial_initialization = vmap(initialize_single_walker, in_axes=(0, None, None))



def initialize_spin_vector(shape, n_z, dtype):

    #  The initializer sets a random number of particles in each walker
    #  to the spin up state in order to create a total z spin as specified.

    _spin_walkers = numpy.zeros(shape=shape, dtype=dtype) - 1
    for i in range(n_z):
        _spin_walkers = _spin_walkers.at[i].add(2.)

    return _spin_walkers
