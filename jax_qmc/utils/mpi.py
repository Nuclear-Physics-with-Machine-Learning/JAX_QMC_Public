import sys, os
import numpy

import logging
from logging import handlers

from . tensor_ops import flatten_tree_into_tensor, unflatten_tensor_into_tree

def discover_local_rank(verbose=False):

    local_rank_key_options = [
            'OMPI_COMM_WORLD_LOCAL_RANK',
            'MV2_COMM_WORLD_LOCAL_RANK',
            'MPI_LOCALRANKID',
            'PMI_LOCAL_RANK',
            'PALS_LOCAL_RANKID',
            ]
    # testable default value:
    local_rank = None
    for key in local_rank_key_options:
        if key in os.environ:
            local_rank = os.environ[key]
            logger = logging.getLogger()
            logger.info(f"Determined local rank through environment variable {key}")
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ[key]
            return int(local_rank)
            break
    if local_rank is None:
        # Try the last-ditch effort of home-brewed local rank deterimination
        from mpi4py import MPI
        import socket
        # Get the global communicator:
        COMM_WORLD = MPI.COMM_WORLD

        # The strategy here is to split into sub communicators
        # Each sub communicator will be just on a single host,
        # And that communicator will assign ranks that can be interpretted
        # as local ranks.

        # To subdivide, each host will need to use a unique key.
        # We'll rely on the hostname and order them all.

        hostname = socket.gethostname()
        # host_key = host_key %
        all_hostnames = COMM_WORLD.gather(hostname, root=0)

        if COMM_WORLD.Get_rank() == 0:
            # Order all the hostnames, and find unique ones
            unique_hosts = numpy.unique(all_hostnames)
            # Numpy automatically sorts them.
        else:
            unique_hosts = None

        # Broadcast the list of hostnames:
        unique_hosts = COMM_WORLD.bcast(unique_hosts, root=0)

        # Find the integer for this host in the list of hosts:
        i = int(numpy.where(unique_hosts == hostname)[0])
        # print(f"{hostname} found itself at index {i}")

        new_comm = COMM_WORLD.Split(color=i)
        if verbose:
            print(f"Global rank {COMM_WORLD.Get_rank()} of {COMM_WORLD.Get_size()} mapped to local rank {new_comm.Get_rank()} of {new_comm.Get_size()} on host {hostname}", flush=True)

        # The rank in the new communicator - which is host-local only - IS the local rank:
        return int(new_comm.Get_rank())


def init_mpi(distributed):
    # Firstly, we check if mpi is on the launch commands:
    try:
        assert distributed
        from mpi4py import MPI
        import mpi4jax
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        MPI_AVAILABLE = True
        return MPI_AVAILABLE, rank, size
    except:
        # print("MPI Configuration failed")
        return False, 0, 1

def allreduce_dict(dictionary):
    from mpi4py import MPI
    import mpi4jax

    # First, we flatten the dictionary:
    flat_tensor, shapes, treedef = flatten_tree_into_tensor(dictionary)

    # Call MPI and perform the allreduce:
    flat_tensor, mpi_token = mpi4jax.allreduce(
        flat_tensor,
        op = MPI.SUM,
        comm = MPI.COMM_WORLD,
        token = None
    )

    reduced_dict = unflatten_tensor_into_tree(flat_tensor, shapes, treedef)
    return reduced_dict
