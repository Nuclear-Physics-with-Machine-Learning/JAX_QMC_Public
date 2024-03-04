# This file contains implementations of the wavefunction as well as functions
# to compute it's derivatives, second derivatives and jacobians.

import jax.numpy as numpy
import jax.random as random

from jax import scipy

from jax import vmap, jit, grad, jacfwd
from jax import tree_util
import flax
import flax.linen as nn

from flax.linen.initializers import xavier_uniform, ones

from typing import Tuple

from ..config import Sampler, ManyBodyCfg

from jax import tree_util


@jit
def leaky_tanh(x, leakiness=0.05):
    return (1-leakiness)*numpy.tanh(x) + (leakiness) * x

from .. spatial import mean_subtract_walker


from . confinement  import Confinement
from . deepsets     import DeepSets, init_deep_sets
from . graph_net    import init_graph_net
from . antisymmetry import init_antisymmetry

from . utils import concat_inputs_single_walker


class ManyBodyWavefunction(nn.Module):
    """
    This class describes a many body wavefunction.

    Composed of several components:
     - a many-body, fully-symmetric correlator component based on the DeepSets wavefunction
     - Individual spatial wavefunctions based on single-particle neural networks

    The apply call computes the many-body correlated multipled by the slater determinant.
    """

    mean_subtract: bool
    backflow:      bool
    confinement:   Confinement
    correlator:    DeepSets
    antisymmetry:  nn.Module
    graph_net:     nn.Module
    time_reversal: bool
    mirror:        bool

    @nn.compact
    def __call__(self, x, spin, isospin):
        '''
        This returns the log of the wavefunction
        '''
        if self.mirror and self.time_reversal:
            logpsi1, s1 = self.wavefunction(  x,   spin,   isospin)
            logpsi2, s2 = self.wavefunction( -x, - spin,   isospin)
            logpsi3, s3 = self.wavefunction(  x,   spin,   isospin)
            logpsi4, s4 = self.wavefunction( -x, - spin,   isospin)


            logpsi, sign = scipy.special.logsumexp(
                a = numpy.asarray([logpsi1, logpsi2, logpsi3, logpsi4]),
                b = numpy.asarray([s1, s2, s3, s4]),
                return_sign = True
            )
        elif self.mirror and not self.time_reversal:
            logpsi1, s1 = self.wavefunction(  x,   spin,   isospin)
            logpsi2, s2 = self.wavefunction( -x,   spin,   isospin)


            logpsi, sign = scipy.special.logsumexp(
                a = numpy.asarray([logpsi1, logpsi2]),
                b = numpy.asarray([s1, s2,]),
                return_sign = True
            )

            # print("Returned logpsi: ", logpsi)
            # print("Returned sign: ", sign)
        elif self.time_reversal and not self.mirror:
            logpsi1, s1 = self.wavefunction(  x,   spin,   isospin)
            logpsi2, s2 = self.wavefunction(  x, - spin,   isospin)


            logpsi, sign = scipy.special.logsumexp(
                a = numpy.asarray([logpsi1, logpsi2]),
                b = numpy.asarray([s1, s2,]),
                return_sign = True
            )

        else:
            logpsi, sign = self.wavefunction(  x,   spin,   isospin)

        return logpsi, sign

    def wavefunction(self, x, spin, isospin):


        # First, do we mean subtract?
        if self.mean_subtract:
            inputs = mean_subtract_walker(x)
        else:
            inputs = x


        # Normalize the inputs by number of particles:
        n_particles = inputs.shape[0]
        norm = 1.2 * numpy.cbrt(n_particles)/numpy.sqrt(3.)
        inputs = inputs / norm

        # Apply confinement just on x:
        confinement = self.confinement(inputs)

        # Create inputs and node features from the EGNN:

        h_i = self.graph_net(inputs, spin, isospin)

        correlation = self.correlator(h_i)

        if self.backflow:
            antisymmetry_inputs = h_i
        else:
            antisymmetry_inputs = concat_inputs_single_walker(inputs,spin,isospin)

        sign, logdet = self.antisymmetry(antisymmetry_inputs)

        w_of_x = confinement + correlation + logdet

        return  w_of_x, sign


def init_wavefunction(wavefunction_cfg, sampler_config):


    conf = Confinement(wavefunction_cfg.confinement)

    if wavefunction_cfg.activation == "leaky_tanh":
        activation = leaky_tanh
    else:
        activation = getattr(nn, wavefunction_cfg.activation)

    corr = init_deep_sets(wavefunction_cfg.correlator_cfg, activation)

    # For the antisymmetry config:
    antisymmetry_net = init_antisymmetry(wavefunction_cfg.antisymmetry, activation)
    # For the graph net:
    graph_net        = init_graph_net(wavefunction_cfg.graph_net, activation)

    wf = ManyBodyWavefunction(
        mean_subtract = wavefunction_cfg.mean_subtract,
        backflow      = wavefunction_cfg.backflow,
        confinement   = conf,
        correlator    = corr,
        antisymmetry  = antisymmetry_net,
        graph_net     = graph_net,
        time_reversal = wavefunction_cfg.time_reversal,
        mirror        = wavefunction_cfg.mirror,
    )

    return wf
