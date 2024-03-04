import jax
import jax.numpy as numpy
from jax import jit, vmap


from functools import partial


@jit
def compute_O_observables(flattened_jacobian, energy):

    # dspi_i is the reduction of the jacobian over all walkers
    # In other words, it's the mean gradient of 
    # the parameters with respect to inputs.
    # This is effectively the measurement of O^i in the paper.

    dpsi_i = numpy.sum(flattened_jacobian, axis=0)
    dpsi_i = numpy.reshape(dpsi_i, (-1,1))

    # Computing <O^m H>:
    e_reshaped = numpy.reshape(energy, (1, -1) )


    dpsi_i_EL = numpy.matmul(e_reshaped, flattened_jacobian)
    
    dpsi_i_EL = numpy.reshape(dpsi_i_EL, [-1, 1])

    return dpsi_i, dpsi_i_EL


@jit
def mean_r(x):
    """
    Reduce x to <R> values, only for metric purposes
    
    :param      x:    { parameter_description }
    :type       x:    { type_description }
    """
    original_shape = x.shape
    mean_sub_r = x - x.mean(axis=1).reshape(
        (original_shape[0], -1, original_shape[-1]))

    # Sum over the position and particle.
    # Summing over x/y/z:
    mean_sub_r = numpy.sum(mean_sub_r**2, axis=(2))
    # Convert to r for each particle:
    mean_sub_r = numpy.sqrt(mean_sub_r)
    # Average over all the particles and walkers:
    mean_sub_r = numpy.mean(mean_sub_r)

    # Sum over x/y/z:
    r = numpy.sum(x**2, axis=(2))
    # convert to r:
    r = numpy.sqrt(r)
    # Average over all walkers and particles:
    r = numpy.mean(r)

    return r, mean_sub_r
