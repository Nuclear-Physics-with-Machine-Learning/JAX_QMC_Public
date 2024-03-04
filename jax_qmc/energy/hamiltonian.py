import jax.numpy as numpy
from jax import random

from jax import jit, vmap

from .. config import HBAR

@jit
def kinetic_energy_jf_log_psi(dlogw_dx, M, HBAR):
    """Return Kinetic energy

    Calculate and return the KE JF

    Arguments:
        w_of_x {DeviceArray} -- Computed derivative of the wavefunction
        dw/dx {DeviceArray} -- Computed derivative of the wavefunction
        M {float} -- Mass

    Returns:
        Device Array - kinetic energy (JF) of shape [1]
    """
    # If w_of_x is the wavefunction, KE_JF is this:
    # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2

    #####################################################################################
    # This section is the (sign, log(psi)) wavefunction
    #####################################################################################

    ke_jf = (HBAR**2 / (2 * M)) * numpy.sum(dlogw_dx**2, axis=(1,2))

    return ke_jf

@jit
def kinetic_energy_log_psi(ke_jf, d2logw_dx2, M, HBAR):
    """Return Kinetic energy


    If all arguments are supplied, calculate and return the KE.

    Arguments:
        d2w_dx2 {tf.Tensor} -- Computed second derivative of the wavefunction
        KE_JF {tf.Tensor} -- JF computation of the kinetic energy

    Returns:
        tf.Tensor - potential energy of shape [1]
    """

    ke = -(HBAR**2 / (2 * M)) * numpy.sum(d2logw_dx2, axis=(1,2))

    return ke - ke_jf


###############################################################################
# Below are deprecated functions if the NN is representing phi not log(phi)
###############################################################################


@jit
def kinetic_energy_jf_psi(w_of_x, dw_dx, M, HBAR):
    """Return Kinetic energy

    Calculate and return the KE directly

    Otherwise, exception

    Arguments:
        w_of_x {DeviceArray} -- Computed derivative of the wavefunction
        dw/dx {DeviceArray} -- Computed derivative of the wavefunction
        M {float} -- Mass

    Returns:
        Device Array - kinetic energy (JF) of shape [1]
    """
    # If w_of_x is the wavefunction, KE_JF is this:
    # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2

    #####################################################################################
    # This section is the non-log wavefunction:
    #####################################################################################
    internal_arg = dw_dx / (w_of_x.reshape(-1,1,1))

    # Contract d2_w_dx over spatial dimensions and particles:
    ke_jf = (HBAR**2 / (2 * M)) * numpy.sum(internal_arg**2, axis=(1,2))

    return ke_jf


@jit
def kinetic_energy_psi(w_of_x, d2w_dx2, M, HBAR):
    """Return Kinetic energy


    If all arguments are supplied, calculate and return the KE.

    Arguments:
        d2w_dx2 {tf.Tensor} -- Computed second derivative of the wavefunction
        KE_JF {tf.Tensor} -- JF computation of the kinetic energy

    Returns:
        tf.Tensor - potential energy of shape [1]
    """

    # If w_of_x is the wavefunction, KE Is this:

    # Compute the inverse of the wavefunction
    inverse_w = numpy.reshape(1/(w_of_x), (-1,1) )

    # Only reduce over the spatial dimension here:
    summed_d2 = numpy.sum(d2w_dx2, axis=(2))

    ke = -(HBAR**2 / (2 * M)) * \
        numpy.sum(inverse_w * summed_d2, axis=1)

    return ke