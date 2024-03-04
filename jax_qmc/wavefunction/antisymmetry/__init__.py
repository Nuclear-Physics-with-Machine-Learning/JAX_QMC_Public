import jax
import jax.numpy as numpy
import jax.scipy as scipy
import flax.linen as nn

from typing import Tuple

from jax_qmc.config.wavefunction import AntisymmetryKind


def init_antisymmetry(antisymmetry_cfg, activation):


    # For the antisymmetry config:

    if antisymmetry_cfg.form == AntisymmetryKind.Slater:
        from . conv_slater import init_slater
        return init_slater(antisymmetry_cfg, activation)