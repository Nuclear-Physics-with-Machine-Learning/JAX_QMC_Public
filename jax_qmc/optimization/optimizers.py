import jax.numpy as numpy
from jax import jit, vmap
import jax

from jax import tree_util

from . gradients import par_dist, natural_gradients
from . gradients import cg_solve, cg_solve_parallel, cholesky_solve

try:
    import mpi4jax
    from mpi4py import MPI
except:
    pass

from jax_qmc.config import Optimizer, Solver
from jax_qmc.utils import unflatten_tensor_like_example, flatten_tree_into_tensor




# Function to recompute the energy based on predicted gradients:



def close_energy_recomputation(function_registry, n_walkers, predict_energy, MPI_AVAILABLE):
    '''
    This closure lets me jit compile the function easily with the other functions
    in local scope, technically
    '''


    @jit
    def recompute_energy_and_overlap(w_params,
            x, spin, isospin, sign, logpsi):

        """
        This function is recomputing the energy with a new set of parameters,
        using the same walkers to avoid a metropolis walk.
        """


        # First, compute the energy:
        new_logw_of_x, new_sign  = function_registry["wavefunction"](w_params, x, spin, isospin)

        # Wavefunction ratio:
        psi_norm = sign * new_sign * numpy.exp(new_logw_of_x - logpsi)
        # Probability ratio:
        psi2_norm = psi_norm**2


        # Average probability ratio:
        # psi2_norm_mean = numpy.mean(psi2_norm)
        # Summing, and manually dividing by the global n_walkers,
        # Which lets us sum with MPI for the correct final answer.
        psi2_norm_mean = numpy.sum(psi2_norm) / n_walkers

        # This is a single-device computation:
        # overlap = numpy.mean(psi_norm)**2 / psi2_norm_mean

        # As well, sum and divide by walkers:
        overlap = numpy.sum(psi_norm) / n_walkers


        if predict_energy:
            dlogw_dx   = function_registry["first_deriv"]( w_params, x, spin, isospin)
            d2logw_dx2 = function_registry["second_deriv"](w_params, x, spin, isospin)

            energy_dict = function_registry["energy"](
                x, spin, isospin,
                new_logw_of_x, sign, dlogw_dx, d2logw_dx2,
                w_params)

            # Normalize:
            energy_dict = jax.tree_util.tree_map(
                lambda x : x / n_walkers,
                energy_dict
            )




            energy_n = numpy.sum(energy_dict["energy"] * psi2_norm) / psi2_norm_mean

            # And, correct the energy:
            energy_n = numpy.sum(energy_dict["energy"] * psi2_norm)

        if MPI_AVAILABLE:

            if predict_energy:
                # Work this allreduce merge by hand:
                reduce_tensor = numpy.concatenate([
                    psi2_norm_mean.flatten(),
                    overlap.flatten(),
                    energy_n.flatten()
                ])
            else:
                reduce_tensor = numpy.concatenate([
                    psi2_norm_mean.flatten(),
                    overlap.flatten(),
                ])

            # Do the reduction:
            reduce_tensor, token =  mpi4jax.allreduce(
                reduce_tensor,
                op = MPI.SUM,
                comm = MPI.COMM_WORLD,
                token = None
            )


            # Unpack by hand too:
            psi2_norm_mean = reduce_tensor[0]
            overlap        = reduce_tensor[1]
            if predict_energy:
                energy_n       = reduce_tensor[2]


        # Correct the final normalizations:
        overlap  = overlap**2  / psi2_norm_mean

        if predict_energy:
            energy_n = energy_n / psi2_norm_mean




        return_dict = {
            "local_overlap" :  overlap,
        }
        if predict_energy:
            return_dict.update({"energy": energy_n})


        return return_dict


    return recompute_energy_and_overlap

@jit
def compute_new_params(parameters, gradients, delta):
    # Apply this learning rate:
    updated_w_params = jax.tree_util.tree_map(
        lambda x, y: x - delta*y,
        parameters, gradients
    )
    return updated_w_params

compute_params_array = jit(vmap(compute_new_params, in_axes=(None, None, 0)))

# Returns a CLOSURE over the actual optimizer function!
def close_over_optimizer(function_registry, config, delta_fn, epsilon_fn, MPI_AVAILABLE):

    # define a local function for the energy recomputation:
    recompute_energy_and_overlap = close_energy_recomputation(
        function_registry, config.sampler.n_walkers,
        config.optimizer.predict_energy, MPI_AVAILABLE)

    # # This pulls the specific initialization and application functions
    # # based on the algorithm specified.
    # opt_init, apply_update_and_update_opt_state = opt_algo_closure(config)

    def opt_init(w_params):

        # Initialize the optimizer state:
        opt_state = {}
        x_0 = tree_util.tree_map(
            lambda x : numpy.zeros(x.shape, dtype=x.dtype) ,
            w_params
        )
        x_0, shapes, treedef = flatten_tree_into_tensor(x_0)
        opt_state["x_0"] = x_0.reshape((-1,1))

        opt_state["g2_i"] = tree_util.tree_map(
            lambda x : numpy.zeros(x.shape, dtype=x.dtype) ,
            w_params
        )

        opt_state["m_i"] = tree_util.tree_map(
            lambda x : numpy.zeros(x.shape, dtype=x.dtype) ,
            w_params
        )

        opt_state['eps'] = numpy.asarray(1.0, dtype=x_0.dtype)

        return opt_state


    @jit
    def generate_opt_metrics(x, spin, isospin,
        sign, log_psi, updated_w_params):

        # Re-Compute the energy with these updated parameters:

        # This function comes back pre-reduced!
        overlap_dict = recompute_energy_and_overlap(
            updated_w_params, x, spin, isospin, sign, log_psi)

        # Get the overlap:
        overlap = overlap_dict["local_overlap"]
        # Get the energy:
        if config.optimizer.predict_energy:
            next_energy  = overlap_dict["energy"]
        else:
            next_energy = numpy.asarray(0.0, dtype=overlap.dtype)


        opt_metrics      = {
            "optimizer/overlap" : overlap,
            # "optimizer/acos" : acos,
        }

        return next_energy, opt_metrics

    # generate_all_opt_metrics = jit(vmap(generate_opt_metrics,
    #     in_axes=(None, None, None, None, 0)))

    @jit
    def second_order_gradients(first_order, regularization_diagonal, jacobian, x_0=None):
        """Compute the second order gradients via Stochastic Reconfiguration

        Args:
            first_order (ndarray): The 1st order gradients
            regularization_diagonal (ndarray): regularization term for the S matrix
            jacobian (ndarray): Jacobian matrix, local only

        Returns:
            ndarray, same shape as first_order, which is the 2nd order grads
        """

        if config.optimizer.solver == Solver.Cholesky:

            # In here, jacobian = jacobian - <jacobian>
            S_ij = numpy.matmul(jacobian.T, jacobian) / config.sampler.n_walkers

            if MPI_AVAILABLE:
                # We have to sum the matrix across ranks in distributed mode!
                S_ij, token = mpi4jax.allreduce(
                    S_ij,
                    op = MPI.SUM,
                    comm = MPI.COMM_WORLD,
                    token = None
                )

            dp_i, residual = cholesky_solve(
                S_ij                    = S_ij,
                regularization_diagonal = regularization_diagonal,
                f_i                     = first_order
            )

        elif config.optimizer.solver == Solver.ConjugateGradient:

            norm =  config.sampler.n_walkers
            if MPI_AVAILABLE:
                dp_i, residual = cg_solve_parallel(
                    jacobian                = jacobian ,
                    regularization_diagonal = regularization_diagonal,
                    f_i                     = first_order,
                    x_0                     = x_0,
                    norm                    = norm
                )
            else:
                dp_i, residual = cg_solve(
                    jacobian                = jacobian,
                    regularization_diagonal = regularization_diagonal,
                    f_i                     = first_order,
                    x_0                     = x_0,
                    norm                    = norm
                )
        return dp_i, residual




    @jit
    def optimization_fn(normalized_observable_tree, jacobian,
        x, spin, isospin, log_psi, sign,
        current_w_params, opt_state, global_step):

        # These are the natural gradients:
        simple_gradients =  natural_gradients(
                        normalized_observable_tree["dpsi_i"],
                        normalized_observable_tree["energy"],
                        normalized_observable_tree["dpsi_i_EL"],
                    )

        # We apply the optax optimizer to the simple gradients:
        gradients = unflatten_tensor_like_example(simple_gradients, current_w_params)
        b1      = numpy.asarray(config.optimizer.b1,  dtype=simple_gradients.dtype)
        b2      = numpy.asarray(config.optimizer.b2,  dtype=simple_gradients.dtype)
        delta   = numpy.asarray(delta_fn(global_step),   dtype=simple_gradients.dtype)
        epsilon = numpy.asarray(epsilon_fn(global_step), dtype=simple_gradients.dtype)

        # This updates the state for RMS Prop regularization:
        # Get g2_i, apply this update, re-store it to opt_state
        g2_i = opt_state['g2_i']
        g2_i = tree_util.tree_map(
            lambda x, y : b1* x + (1 - b1) * y**2,
            g2_i, gradients
        )


        # We create a candidate opt state to replace but only if the gradients
        # are accepted by the step


        candidate_opt_state = {}

        candidate_opt_state['g2_i'] = g2_i
        if config.optimizer.solver is not None:


            # Repack the tensors back into a flat shape:
            repacked_grads, shapes, treedef = flatten_tree_into_tensor(gradients)
            g2_i_flat, shapes, treedef      = flatten_tree_into_tensor(g2_i)

            # If we're applying a 2nd order transform, make sure the diagonal
            # is properly shaped:

            regularization_diagonal = epsilon * (numpy.sqrt(g2_i_flat) + 1e-4)
            regularization_diagonal = regularization_diagonal.reshape(simple_gradients.shape)


            # convert the gradients to second order gradients:
            x_0 = opt_state['x_0']

            repacked_grads, residual = second_order_gradients(
                first_order             = simple_gradients,
                regularization_diagonal = regularization_diagonal,
                jacobian                = jacobian,
                x_0                     = x_0
            )

            residual = numpy.max(numpy.abs(residual))

            candidate_opt_state['x_0'] = repacked_grads




            # Shape the gradients into params space:
            gradients = unflatten_tensor_like_example(
                repacked_grads,
                current_w_params
            )

            # Update and apply the momentum term:
            m_i  = opt_state['m_i']
            gradients  = tree_util.tree_map(
                lambda x, y : b2* x + (1 - b2) * y,
                m_i, gradients
            )
            candidate_opt_state['m_i'] = gradients
            candidate_opt_state['eps'] = epsilon


        else:
            # Just apply the adam optimizer:

            # This is the state update:
            m_i  = opt_state['m_i']
            m_i  = tree_util.tree_map(
                lambda x, y : b2* x + (1 - b2) * y,
                m_i, gradients
            )
            candidate_opt_state['m_i'] = m_i

            gradients = tree_util.tree_map(
                lambda x, y : x / (numpy.sqrt(y) + 1e-8),
                m_i, g2_i
            )
            # This is just a placeholder:
            candidate_opt_state['x_0'] = opt_state['x_0']
            candidate_opt_state['eps'] = epsilon

            residual = 0.0

        if config.optimizer.adaptive:

            # For the adaptive algorithm, we start at a small small learning rate.
            # We consecutively double it,

            raise Exception("Adaptive opt not yet on")
        else: # Non-adaptive algorithm:


            updated_w_params = jax.tree_util.tree_map(
                lambda x, y : x - delta*y,
                current_w_params,
                gradients
            )

            pred_energy, opt_metrics = generate_opt_metrics(
                x, spin, isospin, sign, log_psi, updated_w_params
            )

            null_opt_metrics = {
                "optimizer/overlap" :  opt_metrics["optimizer/overlap"],
                # "optimizer/overlap" :  numpy.asarray(1.0, dtype=epsilon.dtype),
                "optimizer/delta"   :  numpy.asarray(0.0, dtype=epsilon.dtype),
                "optimizer/epsilon" :  epsilon,
                "optimizer/residual": numpy.asarray(0.0, dtype=epsilon.dtype),
            }
            null_pred_energy = normalized_observable_tree["energy"]

            opt_metrics["optimizer/delta"]   = delta
            opt_metrics["optimizer/epsilon"] = epsilon
            opt_metrics["optimizer/residual"]= residual


            # print(opt_metrics["optimizer/overlap"], flush=True)
            # Reject updates that don't overlap:
            cond = opt_metrics["optimizer/overlap"] > 0.5
            return jax.lax.cond(cond,
                lambda : (updated_w_params, opt_metrics, pred_energy, candidate_opt_state),  # on_true
                lambda : (current_w_params, null_opt_metrics, null_pred_energy, opt_state),  # on_false
            )



    return optimization_fn, opt_init
