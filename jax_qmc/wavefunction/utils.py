from jax import jit, vmap
from jax import grad, jacfwd
from jax import tree_util

import jax.numpy as numpy

import flax


def create_grad_and_jac(x, spin, isospin, w_fn):


    # gradient function over the x positions:
    g_fn_raw = grad(w_fn, argnums=1, has_aux=True)

    # jacobian function:
    J_fn_raw = jacfwd(g_fn_raw, argnums=1, has_aux=True)


    n_particles = x.shape[1]
    n_dim       = x.shape[2]

    # This selects out the elements of the jacobian that we actually want:
    selection = numpy.eye(n_particles*n_dim, dtype=x.dtype).reshape(
        (n_particles, n_dim, n_particles, n_dim)
    )

    # To get the jacobian right, we use a lambda function here:

    # Index the output of the J_fn_raw here to ignore the sign
    J_fn = lambda _params, _x, _s, _is: \
        numpy.sum(J_fn_raw(_params, _x, _s,_is)[0]*selection, axis=(2,3))

    g_fn = lambda _params, _x, _s, _is: g_fn_raw(_params, _x, _s,_is)[0]

    return g_fn, J_fn
    # return jit(g_fn), jit(J_fn)

def create_jac_params(w_fn):

    # This is the gradient over the parameters only:
    grad_params = grad(w_fn, argnums=0, has_aux=True)

    # @jit
    def flatten_params(params):
        leaves, treedef = tree_util.tree_flatten(params)
        return numpy.hstack([l.flatten() for l in leaves])

    # Index the output of the grad_params here to ignore the sign
    flat_grad_fn = lambda _params, _x, _s, _is : flatten_params(grad_params(_params, _x, _s,_is)[0])

    return flat_grad_fn
    # return jit(flat_grad_fn)

def init_jit_and_vmap_nn(key, x, spin, isospin, _nn):

    w_params = _nn.init(key, x[0], spin[0], isospin[0])

    w_fn = flax.linen.apply(type(_nn).__call__, _nn)

    logpsi, sign = w_fn(w_params, x[0], spin[0], isospin[0])


    # Intercept here for the jacobian over parameters before vmap:
    J_fn = create_jac_params(w_fn)

    jacobian = J_fn(w_params, x[0], spin[0], isospin[0])


    # Get the gradiant and jacobian functions before vmap:
    g_fn, d2_fn = create_grad_and_jac(x, spin, isospin, w_fn)

    # Map over the walkers:
    w_fn  = jit(vmap(w_fn,  in_axes=(None, 0, 0, 0)))
    g_fn  = jit(vmap(g_fn,  in_axes=(None, 0, 0, 0)))
    d2_fn = jit(vmap(d2_fn, in_axes=(None, 0, 0, 0)))
    J_fn  = jit(vmap(J_fn,  in_axes=(None, 0, 0, 0)))


    return w_params, w_fn, g_fn, d2_fn, J_fn

def create_message(node_like_object, edge_like_object):

    n_particles = node_like_object.shape[0]
    node_features = node_like_object.shape[-1]

    # Create a message from h_i, h_j
    h_i = numpy.tile(node_like_object.reshape(n_particles, 1, node_features), (1, n_particles, 1))
    h_j = numpy.tile(node_like_object.reshape(1, n_particles, node_features), (n_particles, 1, 1))

    message = numpy.concatenate([h_i, h_j, edge_like_object], axis=-1)
    return message

@jit
def concat_inputs_single_walker(x, spin, isospin):
    # Stack up the spins and positions:
    inputs = numpy.concatenate((x,spin.reshape((-1,1)),isospin.reshape((-1,1))), axis=-1 )
    return inputs

concat_inputs = jit(vmap(concat_inputs_single_walker, in_axes=[0,0,0]))
