import jax
import jax.numpy as numpy
import jax.scipy as scipy

import flax.linen as nn

from typing import Tuple

from .. mlp       import MLP, init_mlp
from .. utils     import create_message
from . state_prep import StatePreparer


class AttentionMechanism(nn.Module):

    activation: callable

    @nn.compact
    def __call__(self, edges, features):

        N = features.shape[0]

        attention = self.activation(nn.Dense(1, use_bias=False)(edges))
        attention = nn.softmax(attention, axis=1).reshape((N,N))
        updated_features = numpy.matmul(attention, updated_features)
        
        return self.activation(updated_features)

class MultiHeadAttentionMechanism(nn.Module):
    
    n_heads:    int
    activation: callable

    @nn.compact
    def __call__(self, edges, features):

        N = features.shape[0]

        attention = self.activation(nn.Dense(self.n_heads, use_bias=False)(edges))

        # Apply the softmax function _per_ head:
        exp_attention = numpy.exp(attention)
        denominator   = numpy.sum(exp_attention, axis=1).reshape((N, 1, self.n_heads))
        attention = exp_attention / denominator

        # Transpose to make the matmul work properly:
        attention = attention.transpose((0,2,1))

        # Scale and sum the features by their attention with a matmul:
        updated_features = numpy.matmul(attention, features)

        # Take the average over the multiple heads:
        updated_features = numpy.mean(updated_features, axis=1)

        # Return:
        return self.activation(updated_features)

class GraphAttentionLayer(nn.Module):
    features:   int
    n_heads:    int 
    activation: callable

    @nn.compact
    def __call__(self, nodes, edge_info):

        # Take the input features and compute the new features:
        updated_features = nn.Dense(self.features, use_bias=False)(nodes)

        # Next, build the matrix of attention components:
        N = updated_features.shape[0]

        e_ij = create_message(updated_features, edge_info)


        return MultiHeadAttentionMechanism(self.n_heads, self.activation)(e_ij, updated_features)
        # # Computing the attention, here, is a matter of computing n_heads
        # # different outputs for each input vector e_ij.  We have to treat them all seperately later though.
        # print("message.shape: ", e_ij.shape)
        # attention = self.activation(nn.Dense(self.n_heads, use_bias=False)(e_ij))
        # print("attention.shape: ", attention.shape)
        # # We have to apply softmax to the attention output, but since it's multi-head attention
        # # we do it manually:
        # exp_attention = numpy.exp(attention)
        # denominator   = numpy.sum(exp_attention, axis=1).reshape((N, 1, self.n_heads))
        # attention = exp_attention / denominator
        # print("attention.shape: ", attention.shape)
        # # Move the multi-head feature up front:

        # attention = attention.transpose((0,2,1))
        # print("attention.shape: ", attention.shape)
        # print("updated_feeatures.shape (pre): ", updated_features.shape)
        # updated_features = numpy.matmul(attention, updated_features)
        # print("updated_feeatures.shape (post): ", updated_features.shape)
        # updated_features = numpy.mean(updated_features, axis=1)
        # print("updated_feeatures.shape (average): ", updated_features.shape)


        # # updated_features = nn.celu(numpy.mean(updated_features, axis=0))
        # return self.activation(updated_features)


class GraphAttentionNetwork(nn.Module):
    
    state_prep: StatePreparer
    layers: Tuple[GraphAttentionLayer, ...]

    @nn.compact
    def __call__(self, x, spin, isospin):

        # Then, concatenate them with the basic features:
        # original_features = concat_inputs_single_walker(x, spin, isospin)

        nodes, edges, padded_nodes, padded_edges = self.state_prep(x, spin, isospin)

        features = padded_nodes
        # original_features = nodes

        for layer in self.layers:
            features = features + layer(features, padded_edges)
            # features = numpy.concatenate([original_features, features], axis=-1)
        # return numpy.concatenate([features, original_features], axis=-1)

        return features
    
def init_gat_layer(features, heads, activation):

    return GraphAttentionLayer(features, heads, activation)

def init_graph_attention_net(gan_cfg, state_prep, activation=nn.celu):
    
    layers = [
        init_gat_layer(f, h, activation) 
        for f, h in zip(gan_cfg.features, gan_cfg.heads)
    ]
    return GraphAttentionNetwork(state_prep, layers)