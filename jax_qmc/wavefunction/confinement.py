import jax
import jax.numpy as numpy

import flax.linen as nn

from typing import Tuple

from . mlp import MLP


# This is a module in flax that we'll use to build up the bigger modules:
class Confinement(nn.Module):
    alpha: float
    # alpha: numpy.float32

    @nn.compact
    def __call__(self, x):

        # Compute the confinement term:
        boundary = - self.alpha * (x**2).sum()

        # Return it flattened to a single value for a single walker:
        boundary = boundary.reshape(())

        return boundary
