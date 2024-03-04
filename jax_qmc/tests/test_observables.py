# import jax.numpy as numpy
# import jax.random as random
# from jax import grad, tree_util

# from config import Sampler, ManyBodyCfg

# import pytest

# from spatial import spatial_initialization
# from spatial import close_walk_over_wavefunction
# from spatial import multisplit, multi_normal, multi_uniform
# from spatial import initialize_spin_until_non_zero

# from energy.harmonic_oscillator_potential import compute_energy

# from optimization import compute_final_observables_single_obs, compute_f_i
# from optimization import unflatten_weights_or_gradients

# from wavefunction import init_many_body_wf

# from jax.config import config; config.update("jax_enable_x64", True)


# # For this test, we're creating a single function that handles multiple tests
# # The reason for this is the individual execution time of these tests 
# # could grow unreasonably.

# def test_observable_calcs(sampler_config, network_config, seed):

#     if sampler_config.n_dim > 1:
#         pytest.skip("Only validating observables in 1D.")

#     if sampler_config.n_particles > 1:
#         pytest.skip("Only validating observables in for 1 particle.")


#     # Right now, this test is not ready.  So skip it:
#     pytest.skip("Skipping test that is not finished.")

#     # For this test, TURN OFF spin:
#     network_config.spatial_cfg.n_layers = 0

#     n_total_walkers = sampler_config.n_walkers

#     key = random.PRNGKey(int(seed))
#     key, subkey = random.split(key)
#     walker_keys = random.split(subkey, n_total_walkers)

#     x, spin, isospin = spatial_initialization(
#         walker_keys, sampler_config,  "float64")



#     # Get the wavefunction:
#     w_params, wavefunction, compute_derivatives, compute_jacobian = \
#         init_many_body_wf(subkey, x, spin, isospin, 
#             sampler_config, network_config)

#     w_of_x = wavefunction(w_params, x, spin, isospin)

#     walker_keys, spin_swap_keys = multisplit(walker_keys)

#     new_spin = initialize_spin_until_non_zero(
#         spin_swap_keys, x, spin, isospin, wavefunction, w_params)

#     # Create the kick function:

#     metropolis_walk = close_walk_over_wavefunction(wavefunction)

#     walker_keys, kicker_keys = multisplit(walker_keys)

#     # Perform a number of kicks on this set:
#     acceptance, x, spin, isospin = \
#         metropolis_walk(kicker_keys, w_params,
#                         numpy.asarray(sampler_config.kick_size),
#                         x, spin, isospin, 5000)

#     w_of_x, dw_dx, d2w_dx2 = compute_derivatives(
#         w_params, x, spin, isospin)
#     jacobian = compute_jacobian(w_params, x, spin, isospin)

#     jacobian = jacobian / w_of_x.reshape((-1, 1))


#     # Now, we can assume that x_kicked, spin_kicked, etc are all distributed
#     # According to the distribution of the wavefunction.
#     h_params = {
#         "omega" : numpy.asarray(1.0, dtype="float64"),
#         "mass"  : numpy.asarray(1.0, dtype="float64"),
#         "hbar"  : numpy.asarray(1.0, dtype="float64")
#     }
#     # Since the hamiltonian here can be the harmonic oscillator in 1D
#     energy_dict = compute_energy(h_params, x, spin, isospin, 
#         w_of_x, dw_dx, d2w_dx2)

#     # print(energy_dict)

#     energy_dict = compute_final_observables_single_obs(n_total_walkers,
#         x, energy_dict, w_of_x, jacobian)

#     metropolis_simple_grads = compute_f_i(
#                     energy_dict["dpsi_i"],
#                     energy_dict["energy"],
#                     energy_dict["dpsi_i_EL"],
#                 )


#     # Part 2: Gradients by simple numerical integration:
#     x_numerical = numpy.arange(-5.,5,0.05).reshape((-1,1,1))
#     n_x = x_numerical.shape[0]
#     spin = numpy.zeros((n_x, 1))
#     isospin = numpy.zeros((n_x, 1))


#     def energy_numerical_int(_w_params, _x, _s, _i):
#         integral_width = numpy.prod(_x[1] - _x[0])
#         _w_of_x, _dw_dx, _d2w_dx2 = compute_derivatives(
#             _w_params, _x, _s, _i)

#         _energy_dict = compute_energy(h_params,
#             _x, _s, _i, _w_of_x, _dw_dx, _d2w_dx2)



#         return _energy_dict["energy"].sum()*integral_width

#     gradient_func = grad(energy_numerical_int)

#     dw_dtheta = gradient_func(w_params, x_numerical, spin, isospin)

#     print(metropolis_simple_grads.sum())
#     metropolis_simple_grads = unflatten_weights_or_gradients(
#         metropolis_simple_grads, dw_dtheta)


#     # print(dw_dtheta.sum())
#     print(metropolis_simple_grads.sum())

#     relative_diff = tree_util.tree_map(
#         lambda x, y:  x - y, dw_dtheta, metropolis_simple_grads)

#     # print(relative_diff)

#     #
#     #
#     # l = lambda _w_of_x : compute_energy(h_params, 
#     #   x_numerical, spin, isospin, _w_of_x, dw_dx, d2w_dx2)
#     #
#     # grad_fn = grad(compute_energy)
#     #
#     # energy_dict = compute_energy()
#     #
#     # grad_fn =  grad(numpy.sum(energy_dict["energy"]))
#     #
#     # # print(energy_dict)



#     assert False
