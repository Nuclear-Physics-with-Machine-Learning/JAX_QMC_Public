import jax
import jax.numpy as numpy

import math

def unflatten_tensor_like_example(inputs, example_target):


    # Use this to keep track of where the flat index is:
    running_index = 0;
    # Flatten the example tree:
    leaf_values, treedef = jax.tree_util.tree_flatten(example_target)

    input_leaf_values = []

    for leaf_value in leaf_values:
        # How big is the leaf?
        input_size = leaf_value.size
        # Take a slice that size:
        input_slice = inputs[running_index:running_index+input_size]
        # Add it to the output tree values shaped properly:
        input_leaf_values.append(input_slice.reshape(leaf_value.shape))
        # Update the start point
        running_index += input_size

    return jax.tree_util.tree_unflatten(treedef, input_leaf_values)

def unflatten_tensor_into_tree(inputs, shapes, treedef):

    # Use this to keep track of where the flat index is:
    running_index = 0;

    input_leaf_values = []

    for shape in shapes:
        # How big is the leaf?
        input_size = math.prod(shape)
        # Take a slice that size:
        input_slice = inputs[running_index:running_index+input_size]
        # Add it to the output tree values shaped properly:
        input_leaf_values.append(input_slice.reshape(shape))
        # Update the start point
        running_index += input_size

    return jax.tree_util.tree_unflatten(treedef, input_leaf_values)

def flatten_tree_into_tensor(input_tree):

    # Flatten the tree structure into a flat structure:
    leaf_values, treedef = jax.tree_util.tree_flatten(input_tree)

    # Extract the shapes of the tensors:
    shapes = [ t.shape for t in leaf_values ]

    # Flatten every tensor:
    flattened = [t.flatten() for t in leaf_values ]

    # Combine the tensors:
    flat_tensor = numpy.concatenate(flattened, axis=0)

    return flat_tensor, shapes, treedef
