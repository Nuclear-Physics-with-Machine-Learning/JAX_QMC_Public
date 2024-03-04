import jax

import flax.linen as nn

from typing import Tuple


# This is a module in flax that we'll use to build up the bigger modules:
class MLP(nn.Module):
    n_outputs:       Tuple[int, ...]
    bias:            bool
    activation:      callable
    last_activation: bool

    def setup(self):

        self.layers = [
            nn.Dense(n_out,
                use_bias = self.bias,
                # kernel_init = nn.initializers.xavier_uniform(),
                )
            for n_out in self.n_outputs
        ]

    def __call__(self, x):
        layer_input = x

        # Loop over the layers
        for i, layer in enumerate(self.layers):
            # compute the application of the layer:
            layer_output = layer(layer_input)
            if layer_output.shape == layer_input.shape:
                layer_output = layer_output + layer_input
            # If it's the last layer, don't apply activation if not specified:

            if i != len(self.layers) - 1 or self.last_activation:
                layer_output = self.activation(layer_output)

            # Prepare for the next layer:
            layer_input = layer_output


        return layer_output

def init_mlp(mlp_cfg, activation):

    mlp = MLP(
        n_outputs       = mlp_cfg.layers,
        bias            = mlp_cfg.bias,
        activation      = activation,
        last_activation = mlp_cfg.last_activation
    )

    return mlp
