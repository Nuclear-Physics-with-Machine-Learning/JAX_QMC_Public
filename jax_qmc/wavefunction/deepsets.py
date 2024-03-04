import jax
import jax.numpy as numpy
import jax.scipy as scipy
import flax.linen as nn

from typing import Tuple

from . mlp import MLP, init_mlp


# This is a module in flax that we'll use to build up the bigger modules:
class DeepSets(nn.Module):
    individual_module: MLP
    aggregate_module:  MLP
    active: bool

    @nn.compact
    def __call__(self, inputs):
        # Compute the correlator:
        if self.active:
            # Apply the individual network, sum over particles:
            individual_response = self.individual_module(inputs)
            
            # Sum the inputs directly over the particle dimension.
            # individual_response = scipy.special.logsumexp(individual_response, axis=0)
            individual_response = numpy.mean(individual_response, axis=0)
            
            # Apply the aggregate network over the latent space:
            aggregate_response = self.aggregate_module(individual_response)
            return aggregate_response.reshape(())
        else:
            return numpy.ones(())

def init_deep_sets(deepsets_cfg, activation):

    active = deepsets_cfg.active

    i_module = init_mlp(deepsets_cfg.individual_cfg, activation)
    a_module = init_mlp(deepsets_cfg.aggregate_cfg,  activation)

    deep_sets = DeepSets(i_module, a_module, active=active)
    # deep_sets = DeepSets(a_module, active=active)

    return deep_sets
