import jax.numpy as numpy
import jax.random as random
import jax

from jax_qmc.config import Sampler, ManyBodyCfg

import pytest
import time

from jax_qmc.spatial      import spatial_initialization, multisplit
from jax_qmc.spatial      import initialize_spin_until_non_zero
from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn



# from jax_qmc.wavefunction import init_many_body_wf_test_functions
from jax_qmc.spatial import select_and_swap_particles
from jax_qmc.wavefunction.deepsets import init_deep_sets
from jax_qmc.wavefunction.utils    import concat_inputs

import flax
from jax import jit, vmap



def test_derivatives(sampler_config, network_config, seed):

    nwalkers = sampler_config.n_walkers
    nparticles = sampler_config.n_particles
    ndim = sampler_config.n_dim

    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")

    # Create the wavefunction:
    key, subkey = random.split(key)


    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction_fn, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)


    multikey, spin_init_key = multisplit(multikey)

    spin = initialize_spin_until_non_zero(
        spin_init_key, x, spin, isospin, wavefunction_fn, w_params)



    # w_of_x, dw_dx, d2w_dx2 = compute_derivatives(
    #     w_params, x, spin, isospin)

    log_w_of_x, sign = wavefunction_fn(w_params, x, spin, isospin)
    dw_dx   = g_fn(w_params, x, spin, isospin)
    d2w_dx2 = d2_fn(w_params, x, spin, isospin)

    # Scale the derivatives by the sign:
    # dw_dx   = dw_dx*sign.reshape(-1,-1,1)
    # d2w_dx2 = d2w_dx2*sign

    assert dw_dx.shape   == (nwalkers, nparticles, ndim)
    assert d2w_dx2.shape == (nwalkers, nparticles, ndim)


    # We proceed to check the derivatives with finte difference methods.

    # Need a "difference" term:
    kick = numpy.zeros(shape = x.shape)
    kick_size = 1e-4

    # First and second order derivative.  Check it for each dimension.
    for i_dim in range(ndim):

        # select a random particle to kick per walker:
        key, subkey = random.split(key)
        i_kicked_particle = random.choice(subkey, nparticles, shape=(nwalkers,))
        # i_kicked_particle = 0
        this_kick = numpy.copy(kick)
        # Have to create an index for walkers to slice:
        walkers = numpy.arange(nwalkers)
        # Only applying to particle 0
        this_kick = this_kick.at[walkers,i_kicked_particle,i_dim].add(kick_size)
        kicked_up_input = x + this_kick

        kicked_double_up_input = x + 2*this_kick
        # # Mean subtract:
        # up_x = kicked_up_input - \
        #     numpy.reshape(numpy.mean(kicked_up_input, axis=1), (nwalkers, 1, ndim))

        kicked_down_input = x - this_kick

        kicked_double_down_input = x - 2*this_kick
        # down_x = kicked_down_input - \
        #     numpy.reshape(numpy.mean(kicked_down_input, axis=1), (nwalkers, 1, ndim))


        # In this case, *because* there is a mean subtraction,
        # we will calculate a derivate for the first particle only.
        # The derivatives for the other particles based on this kick will be
        # flipped sign.

        # The total kick will actually be kick_size / nparticles, because of
        # the effect of mean subtraction

        # Differences:
        w_up, sign_up = wavefunction_fn(w_params, kicked_up_input, spin, isospin)
        w_down, sign_down = wavefunction_fn(w_params, kicked_down_input, spin, isospin)
        w_up_up, sign_up_up = wavefunction_fn(w_params, kicked_double_up_input, spin, isospin)
        w_down_down, sign_down_down = wavefunction_fn(w_params, kicked_double_down_input, spin, isospin)



        # Use numpy to make slicing easier
        w_prime_fd = (w_up - w_down) / (2*kick_size)
        # What about the second derivative?

        # https://math.stackexchange.com/questions/3756717/finite-differences-second-derivative-as-successive-application-of-the-first-deri
        # This gives precision of O(kick**4)
        w_prime_prime_num = -w_down_down + 16*w_down - 30* log_w_of_x + 16 * w_up - w_up_up
        w_prime_prime_fd = w_prime_prime_num/ (12 * kick_size**2)



        # Now, verify everything is correct.
        # print("dw_dx: ", dw_dx)
         # slice to just the dimension we're moving, all walkers
        target = dw_dx[walkers,i_kicked_particle,i_dim]
        second_target = d2w_dx2[walkers,i_kicked_particle, i_dim]

        print("target: ", target)
        print("w_prime_fd: ", w_prime_fd)
        # The tolerance on the second order first derivative is net_kick**2
        print("First difference: ", (w_prime_fd - target) )
        print("Target tolerance: ", kick_size**2)
        assert( numpy.abs(w_prime_fd - target) < kick_size ).all()

        print("second_target: ", second_target)
        print("w_prime_prime_fd: ", w_prime_prime_fd)
        print("2nd difference: ", w_prime_prime_fd - second_target)
        assert (numpy.abs(w_prime_prime_fd - second_target) < kick_size ).all()

def test_jacobian(sampler_config, network_config, seed):

    nwalkers = sampler_config.n_walkers
    nparticles = sampler_config.n_particles
    ndim = sampler_config.n_dim

    print(network_config)


    key = random.PRNGKey(int(seed))

    # How many total walkers?
    n_walkers = sampler_config.n_walkers
    key, subkey = random.split(key)
    multikey = random.split(subkey, n_walkers)

    # Spin up some initial points:
    x, spin, isospin = spatial_initialization(multikey, sampler_config, "float64")

    # Create the wavefunction:
    key, subkey = random.split(key)

    wf = init_wavefunction(network_config, sampler_config)

    key, subkey = random.split(key)
    w_params, wavefunction_fn, g_fn, d2_fn, J_fn = init_jit_and_vmap_nn(key, x, spin, isospin, wf)

    multikey, spin_init_key = multisplit(multikey)

    spin = initialize_spin_until_non_zero(
        spin_init_key, x, spin, isospin, wavefunction_fn, w_params)



    # How many total parameters in the w_params tree?

    n_parameters = 0
    flat_params, tree_def = jax.tree_util.tree_flatten(w_params)
    for p in flat_params:
        n_parameters += p.size


    jacobian = J_fn(w_params, x, spin, isospin)


    assert jacobian.shape == (nwalkers, n_parameters)


    kick = 5e-4

    # We proceed to check the derivatives with finte difference methods.

    # This becomes a loop over the parameters, which is most easily implemented
    # as a loop over the flat tree and a subloop over that parameter index
    global_parameter_index = 0
    for i_weight_layer, layer in enumerate(flat_params):
        this_layer_index = 0
        n_parameters_this_layer = numpy.asarray(layer.shape)

        for i_layer_index in range(layer.size):

            # Update the weight in this layer:
            this_layer = layer.flatten().at[i_layer_index].add(kick)
            flat_params[i_weight_layer] = this_layer.reshape(layer.shape)

            # re-tree the weights:
            w_params_up = jax.tree_util.tree_unflatten(tree_def, flat_params)

            w_up, sign_up = wavefunction_fn(w_params_up, x, spin, isospin)
            # Update the weight in this layer:
            this_layer = layer.flatten().at[i_layer_index].add(-kick)
            flat_params[i_weight_layer] = this_layer.reshape(layer.shape)

            # re-tree the weights:
            w_params_down = jax.tree_util.tree_unflatten(tree_def, flat_params)

            w_down, sign_down = wavefunction_fn(w_params_down, x, spin, isospin)

            jac_column = (w_up - w_down)  / (2*kick)

            print("jac_column: ", jac_column)
            print("jacobian[:,global_parameter_index]: ", jacobian[:,global_parameter_index])

            diff = jac_column - jacobian[:,global_parameter_index]
            print("diff: ", diff)
            print(numpy.abs(jacobian[:, global_parameter_index])*numpy.sqrt(kick) )
            assert numpy.allclose(jac_column, jacobian[:, global_parameter_index], atol=kick, rtol=numpy.sqrt(kick))

            # diff = numpy.abs(jac_column - jacobian[:,global_parameter_index])
            # relative_diff = diff / (jacobian[:,global_parameter_index] + 1e-8)
            # print("diff / (jacobian + 1e-8)", relative_diff  )
            # # This isn't necessarily a great solution ...
            # # The accuracy here depends on machine precision.
            # compared_diff = numpy.minimum(diff, relative_diff)
            # assert (compared_diff  < 1e-3 ).all()

            this_layer_index += 1
            global_parameter_index += 1

        # Set the parameter back after updates:
        flat_params[i_weight_layer] = layer

    # Need a "difference" term:
    kick = numpy.zeros(shape = x.shape)
