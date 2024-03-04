import jax
import jax.numpy as numpy
import jax.scipy as scipy

import flax.linen as nn

from typing import Tuple, Union

class StatePadder(nn.Module):
    """docstring for StatePreparer"""

    n_output: int

    @nn.compact
    def __call__(self, inputs):

        # First, determine the number of outputs needed:
        n_output_neurons = self.n_output - inputs.shape[-1]
        # n_output_neurons = self.n_output

        # Then, create and apply a single dense layer - no bias - and concat
        # the output with the input:
        padding = nn.Dense(n_output_neurons,
                kernel_init = nn.initializers.xavier_uniform(),
                )(inputs)

        output = numpy.concatenate([inputs, padding], axis=-1)
        return  output

from .. utils import concat_inputs_single_walker
class StatePreparer(nn.Module):

    padding: Union[int, None]
    edges:      bool

    @nn.compact
    def __call__(self, x, spin, isospin):

        # X we leave as it is

        # Spin and isospin get combined into the node "state":
        # print(x.shape)
        # print(spin.shape)
        # print(isospin.shape)
        nodes = numpy.concatenate([x, spin.reshape((-1,1)), isospin.reshape((-1,1))], axis=-1)


        if self.padding is not None:
            padded_nodes = StatePadder(self.padding)(nodes)
        else:
            padded_nodes = None

        if self.edges:
            # Edges gets the difference between different nucleons: same spin/isospin and distance between them:
            n_particles = x.shape[0]
            n_dim       = x.shape[1]

            # Reshape these to be easy to broadcast:
            r_i = x.reshape((1, n_particles, n_dim))
            r_j = x.reshape((n_particles, 1, n_dim))

            # This is the displacement vector between all particles:
            r_ij = r_i - r_j

            # This is the displacement magnitude:
            # r_mag = numpy.sqrt( numpy.sum(r_ij**2, axis=-1)).reshape((n_particles, 1, 1))


            # We use the distances between nodes as the edge features, which is invariant
            # Under many transforms.

            # r_ij now has shape [n_particles, n_particles, n_dim]
            # To add spin and isospin, we need to create s_i and s_j as part of this

            # We do this through tiling along different dimensions for i and j:

            s_i = numpy.tile(spin.reshape(n_particles, 1, 1), (1, n_particles, 1))
            s_j = numpy.tile(spin.reshape(1, n_particles, 1), (n_particles, 1, 1))

            i_i = numpy.tile(isospin.reshape(n_particles, 1, 1), (1, n_particles, 1))
            i_j = numpy.tile(isospin.reshape(1, n_particles, 1), (n_particles, 1, 1))

            edges = numpy.concatenate([r_ij, s_i, s_j, i_i, i_j], axis=-1)
            
            # Pad the edge info, if needed:
            if self.padding is not None:
                padded_edges = StatePadder(self.padding)(edges)
            else:
                padded_edges = None
        else:
            edges = None
            padded_edges = None

        return nodes, edges, padded_nodes, padded_edges

def init_state_prep(state_prep_cfg):

    state_prep = StatePreparer(state_prep_cfg.padding, state_prep_cfg.edges)

    return state_prep
