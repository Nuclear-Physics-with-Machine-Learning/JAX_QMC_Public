import jax.numpy as numpy
import jax.random as random
import jax

from jax_qmc.config import Sampler, ManyBodyCfg

import pytest
import time

from jax_qmc.spatial      import spatial_initialization
from jax_qmc.spatial      import initialize_spin_until_non_zero
from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn

from jax_qmc.optimization import cholesky_solve

@pytest.mark.parametrize("size", [10,100])
def test_cholesky_solve(size, seed):

    # Here, we simply construct a hermitian positive matrix, a target vector,
    # run the solve, and verify the result is correct.

    key = random.PRNGKey(int(seed))

    key, subkey = random.split(key)
    diag = random.uniform(subkey, (size,1))
    # print(diag)
    M = numpy.eye(size) * diag
    # print(M)

    # Add it to it's transpose to ensure it's equal to it's transpose:
    S = M + M.T

    # print(S)

    assert (S == S.T).all()

    # Create a random target vector:
    key, subkey = random.split(key)
    X = random.uniform(subkey, (size,))

    # We want to be sure that the solution actually exist, so
    # Compute the solution:
    B = numpy.matmul(X, S)
    # print(B)
    regularization_diagonal = numpy.zeros((size))
    dp_i, residual = cholesky_solve(S, regularization_diagonal, B)
    # print(dp_i)
    # print(X)
    assert(numpy.allclose(dp_i,X).all())
