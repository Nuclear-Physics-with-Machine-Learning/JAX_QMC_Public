import jax.numpy as numpy
from jax import jit

from jax import scipy



@jit
def natural_gradients(dpsi_i, energy, dpsi_i_EL):
    return 2*(dpsi_i_EL - dpsi_i * energy)

# @jit
# def compute_S_ij(dpsi_ij, dpsi_i):

#     return dpsi_ij - dpsi_i * dpsi_i.T


@jit
def par_dist(dp_i, S_ij):
    dist = (dp_i*numpy.matmul(S_ij, dp_i)).sum()
    return dist


@jit
def regularize_S_ij(S_ij, regularization_diagonal):
    # Make the diagonal a 1d tensor:
    S_ij_d = S_ij + numpy.diag(regularization_diagonal.reshape(-1,))
    return S_ij_d


@jit
def cholesky_solve(S_ij, regularization_diagonal, f_i):
    """Summary

    Args:
        S_ij (TYPE): Description
        eps (TYPE): Description
        f_i (TYPE): Description

    Returns:
        TYPE: Description
    """
    # Regularize along the diagonal:
    S_ij_d = regularize_S_ij(S_ij, regularization_diagonal)


    # Next, we need S_ij to be positive definite.
    U_ij = scipy.linalg.cholesky(S_ij_d)

    U_and_lower = (U_ij, False)

    dp_i = scipy.linalg.cho_solve(U_and_lower, f_i)

    residual = numpy.matmul(S_ij_d, dp_i) - f_i

    return dp_i, residual

# @jit
def cg_solve(jacobian, regularization_diagonal, f_i, x_0, norm):
    '''
    Compute the solution to f = Sx (solve for x)
    when supplied S, f.

    We actually supply the jacobian, J, and
    compute S = J.J^T + eps I
    '''
    # regularization_diagonal = regularization_diagonal.reshape((-1,1))
    # f_i = f_i.reshape((-1,))

    @jit
    def A(x):

        # First, we compute L = matmul(jacobian, x)
        # Then, we can compute f = matmul(Jacobian.T, L)
        # This process saves on memory by avoiding construction of
        # the entire S_ij matrix.

        # The jacobian we have is the local o^tilde_n_k object.

        L = numpy.matmul(jacobian, x)

        dp_i_test = numpy.matmul(jacobian.T, L) / norm

        res = dp_i_test + x*regularization_diagonal
        return res

    dp_i, info = scipy.sparse.linalg.cg(A, f_i, x0=x_0, tol=1e-5, atol=0.0, maxiter=200)

    residual = A(dp_i) - f_i

    return dp_i, residual

try:
    import mpi4jax
    from mpi4py import MPI
except:
    pass

@jit
def cg_solve_parallel(jacobian, regularization_diagonal, f_i, x_0, norm):
    '''
    Compute the solution to f = Sx (solve for x)
    when supplied S, f.

    We actually supply the jacobian, J, and
    compute S = J.J^T + eps I
    '''

    # print(jacobian.shape)
    # print(f_i.shape)
    @jit
    def A(x):

        # First, we compute L = matmul(jacobian, x)
        # Then, we can compute f = matmul(Jacobian.T, L)
        # This process saves on memory by avoiding construction of
        # the entire S_ij matrix.

        # How to make it collective?
        # The final step, J.T @ L, just needs to be summed over all ranks.

        # The jacobian we have is the local o^tilde_n_k object.

        L = numpy.matmul(jacobian, x)

        dp_i_test = numpy.matmul(jacobian.T, L) / norm

        dp_i_test, token = mpi4jax.allreduce(
            dp_i_test,
            op    = MPI.SUM,
            comm  = MPI.COMM_WORLD,
            token = None
        )

        return dp_i_test + x*regularization_diagonal


    dp_i, info = scipy.sparse.linalg.cg(A, f_i, x0=x_0, tol=1e-5, atol=0.0, maxiter=200)
    residual = A(dp_i) - f_i

    return dp_i, residual
