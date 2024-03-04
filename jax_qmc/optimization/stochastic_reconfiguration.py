import jax
import jax.numpy as numpy
import jax.random as random
from jax import jit, vmap
from jax import grad

try:
    import mpi4jax
    from mpi4py import MPI
except:
    pass

from collections import OrderedDict
import time

# from spatial import metropolis_walk

import logging
logger = logging.getLogger()

from jax_qmc.spatial import multisplit
from jax_qmc.utils   import allreduce_dict

from functools import partial


from . observables import compute_O_observables, mean_r


#
# recompute_energy_and_overlap = partial(jit, static_argnums=(0,1,2))(vmap(recompute_energy_and_overlap_single_obs,
#         in_axes=(None, None, None, None, 0,0,0,0)))


from functools import partial
from jax import tree_util
from jax import lax

from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start

@jit
def normalize_jacobian(original_jacobian, dpsi_i):
    # we compute Õ^i = O^i - <O^i>
    return original_jacobian - dpsi_i.reshape((1,-1))




def sr_step(function_registry, w_params, opt_state, global_step,
    keys, x, spin, isospin, sampler_config, world_size, MPI_AVAILABLE):
    """
    Perform one step of the stochastic reconfiguration algorithm.

    """
    metrics = {}

    with catchtime() as t:
        # We do a thermalization step again:
        acceptance, x, spin, isospin  = \
            function_registry["metropolis"](
                keys, w_params,
                numpy.asarray(sampler_config.kick_size),
                x, spin, isospin,
                sampler_config.n_void_steps)


    metrics["time/walk"] = t()

    with catchtime() as t:

        # Compute the observables:
        # Here, perhaps we can compute the d_i of obs_energy:
        logw_of_x, sign  = function_registry["wavefunction_comp"](w_params, x, spin, isospin)
        dlogw_dx   = function_registry["first_deriv_comp"]( w_params, x, spin, isospin)
        d2logw_dx2 = function_registry["second_deriv_comp"](w_params, x, spin, isospin)
        # with jax.disable_jit():
        observable_tree = \
            function_registry["energy"](
                x, spin, isospin,
                logw_of_x, sign, dlogw_dx, d2logw_dx2,
                w_params)


        # For the uncertainty, compute the energy**2 locally:
        observable_tree["energy2"] = observable_tree["energy"]**2
        observable_tree["energy_jf2"] = observable_tree["energy_jf"]**2

    metrics["time/observables"] = t()

    with catchtime() as t:
        # Compute the flattened jacobian too:
        jacobian = function_registry["jacobian"](w_params, x, spin, isospin)

        # Shape is (n_walkers, n_params) - here, meaning n_walkers_LOCAL

        # This is not needed with the NN returning log(psi)
        # Normalize the jacobian by the value of the wavefunction:
        # jacobian = jacobian / w_of_x.reshape((-1,1))

        # We have to normalize the observable_tree - global normalization!:
        observable_tree = jax.tree_util.tree_map(
            lambda x : x / sampler_config.n_walkers,
            observable_tree
        )

        # this is a LOCAL computation:
        dpsi_i, dpsi_i_EL = compute_O_observables(jacobian, observable_tree["energy"])

        # this is a GLOBAL normalization
        dpsi_i   = dpsi_i   / sampler_config.n_walkers

        # For tracking purposes:
        r, mean_sub_r = mean_r(x)

        observable_tree["r"] = r
        observable_tree["mean_sub_r"] = mean_sub_r

    metrics["time/jacobian"] = t()

    with catchtime() as t:

        # For everything in the normalized tree, sum it:
        normalized_tree = jax.tree_util.tree_map(
            lambda x : numpy.sum(x),
            observable_tree
        )

        normalized_tree["dpsi_i"]    = dpsi_i
        normalized_tree["dpsi_i_EL"] = dpsi_i_EL


        # Above, n_walkers is a global variable accounting for all walkers over all ranks
        # When we sum, below, it completes the averaging if needed

        # Here, if MPI is available, we can do a reduction (sum) over walker variables


        if MPI_AVAILABLE:
            normalized_tree = allreduce_dict(normalized_tree)
            # for key in normalized_tree:
            #     normalized_tree[key], mpi_token = mpi4jax.allreduce(
            #         normalized_tree[key],
            #         op = MPI.SUM,
            #         comm = MPI.COMM_WORLD,
            #         token = mpi_token
            #     )
            # # Note: because we apply a global norm (1/n_walkers), we SUM here and don't AVERAGE

    metrics["time/normalization"] = t()

    with catchtime() as t:
        # Pull off the metrics values and name them nicely for tensorboard:
        error = numpy.sqrt((normalized_tree["energy2"] - normalized_tree["energy"]**2) / \
            (sampler_config.n_walkers - 1) )
        error_jf = numpy.sqrt((normalized_tree["energy_jf2"] - normalized_tree["energy_jf"]**2 )/ \
            (sampler_config.n_walkers - 1) )

        metrics['energy/energy']         = normalized_tree["energy"]
        metrics['energy/error']          = error
        metrics['energy/energy_jf']      = normalized_tree["energy_jf"]
        metrics['energy/error_jf']       = error_jf
        metrics['metropolis/accept_x']   = acceptance['x']
        metrics['metropolis/accept_spin']= acceptance['spin']
        metrics['metropolis/r']          = normalized_tree['r']
        metrics['metropolis/mean_sub_r'] = normalized_tree['mean_sub_r']
        metrics['energy/ke_jf']          = normalized_tree["ke_jf"]
        metrics['energy/ke_direct']      = normalized_tree["ke_direct"]
        metrics['energy/pe']             = normalized_tree["pe"]

    metrics["time/metrics"] = t()


    with catchtime() as t:

        jacobian = normalize_jacobian(jacobian, normalized_tree["dpsi_i"])



        # Snapshot the inputs to the optimizer function for creation of a miniapp
        # If disabled, this is a no-op:
        function_registry["opt_snapshot"](
            global_step,
            dpsi_i = normalized_tree['dpsi_i'],
            energy = normalized_tree['energy'],
            dpsi_i_EL = normalized_tree['dpsi_i_EL'],
            jacobian = jacobian,
            w_params = w_params,
            opt_state = opt_state,
        )
       


        # Here, we pass the variables to the optimizer of choice:
        updated_w_params, opt_metrics, pred_energy, opt_state = function_registry["optimizer"](
            normalized_tree, jacobian,
            x, spin, isospin, 
            logw_of_x, sign,
            w_params, opt_state, global_step
        )

        metrics.update(opt_metrics)

    metrics["time/optimizer"] = t()

    return updated_w_params, opt_state, metrics, pred_energy, x, spin, isospin
