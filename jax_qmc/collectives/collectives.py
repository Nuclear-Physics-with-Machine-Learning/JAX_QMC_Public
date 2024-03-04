import jax

try:
    import mpi4jax
    from mpi4py import MPI
except:
    pass

def allreduce_dict(input_dict, fusion=False):

    if not fusion:
        token = None
        for key in summed_tree:
            summed_tree[key], token = mpi4jax.allreduce(
                summed_tree[key],
                op = MPI.SUM,
                comm = MPI.COMM_WORLD,
                token = token
            )
        return summed_tree
    else:
        # Here, we fuse all the tensors together
        # First, capture all the keys and shapes:
        shapes_tree = jax.tree_util.tree_map(
                lambda x : x.shape, input_dict
        )

        # Next, flatten all the tensors:
        flat_tree = jax.tree_util.tree_map(
            lambda x : numpy.flatten(x), input_dict
        )

        # Flatten out all the flat leaves into one long array:
        leaf_values, treedef = jax.tree_util.tree_flatten(flat_tree)

        # Concatenate the leaf_values together:
        
