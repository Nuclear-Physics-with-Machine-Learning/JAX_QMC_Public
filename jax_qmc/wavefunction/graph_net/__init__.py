import jax
import jax.numpy as numpy
import jax.scipy as scipy
import flax.linen as nn

from typing import Tuple

from jax_qmc.config.wavefunction import GraphKind 

from . state_prep import init_state_prep

def init_graph_net(graph_net_cfg, activation):

    state_prep = init_state_prep(graph_net_cfg.state_prep)

    if graph_net_cfg.form == GraphKind.Attention:
        from . attention import init_graph_attention_net
        return init_graph_attention_net(graph_net_cfg, state_prep, activation)

    else:
        from .. utils import concat_inputs_single_walker
        return concat_inputs_single_walker