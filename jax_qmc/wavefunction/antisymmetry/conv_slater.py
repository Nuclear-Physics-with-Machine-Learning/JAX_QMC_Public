import jax
import jax.numpy as numpy

import flax.linen as nn

from typing import Tuple


class Slater(nn.Module):
    
    n_outputs:       Tuple[int, ...]
    bias:            bool
    active:          bool
    activation:      callable
    last_activation: bool
    
    def setup(self):
        
        self.layers = [
            nn.ConvLocal(
                features    = n_out,
                use_bias    = self.bias,
                kernel_size = (1,1),
                kernel_init = nn.initializers.xavier_uniform()
                )
            for n_out in self.n_outputs
        ]
        
    def __call__(self, x):
        '''
        The trick here is three fold:
        - First, a convolution of kernelsize 1 is an MLP applied to every pixel.
        - Second, the "ConvLocal" variety of convolutions uses a unique kernel at every input location
        - Third, by repeating and reshaping the inputs, we make exactly n_part unique input locations.
        All told, this gives n_part MLPs that are applied to the n_part*n_part locations.
        This vectorizes the calls but also prevents the use of python loops.
        Avoid loops significantly improves compilation and execution speed
        '''

        if self.active:
            # Get the parameter shapes:
            n_part = x.shape[0]
            n_dim  = x.shape[1]
            # Repeat x along the particle axis:
            x = numpy.repeat(x, n_part, axis=0)

            # Reshape it to an image of shape NHWC = (n_part, 1,n_part, n_dim)
            x = x.reshape((n_part,1,n_part,n_dim))
            
            # Apply the layers of the network:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i != len(self.layers) -1 or self.last_activation:
                    x = self.activation(x)
                    
            # Remove the final output layer dimension, and transpose:
            slater = x.reshape((n_part, n_part,)).T


            # sign, logdet = numpy.linalg.slogdet(slater)
            # det = numpy.linalg.det(slater)
            sign, logdet = numpy.linalg.slogdet(slater)
            # w = sign * numpy.exp(logdet)

            # Flatten it:
            return sign.reshape(()), logdet.reshape(())
        else:
            return numpy.ones((), dtype=x.dtype), numpy.zeros((), dtype=x.dtype)

def init_slater(slater_cfg, activation):


    conv_slater = Slater(
        n_outputs       = slater_cfg.mlp_cfg.layers,
        bias            = slater_cfg.mlp_cfg.bias,
        active          = slater_cfg.active,
        activation      = activation,
        last_activation = slater_cfg.mlp_cfg.last_activation
        
    )
    return conv_slater

