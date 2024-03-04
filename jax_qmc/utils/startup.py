import jax
import logging
from logging import handlers
import pathlib

def should_do_io(_mpi_available, _rank):
    if not _mpi_available or _rank == 0:
        return True
    return False

def set_compute_parameters(local_rank):
    import socket
    try:
        devices = jax.local_devices()
    except:
        devices = []
    if len(devices) == 1:
        # Something external has set which devices are visible
        target_device = devices[0]
    elif len(devices) > 1:
        target_device = devices[local_rank]
    else:
        target_device = jax.devices("cpu")[0]
    # Not a pure function by any means ...
    from jax.config import config; config.update("jax_enable_x64", True)
    # config.update('jax_disable_jit', True)

    return target_device


def configure_logger(save_path, MPI_AVAILABLE, rank):

    logger = logging.getLogger()

    # Create a handler for STDOUT, but only on the root rank:
    if should_do_io(MPI_AVAILABLE, rank):
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        handler = handlers.MemoryHandler(capacity = 1, target=stream_handler)
        logger.addHandler(handler)
        # Add a file handler:

        # Add a file handler too:
        log_file = save_path / pathlib.Path("process.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler = handlers.MemoryHandler(capacity=1, target=file_handler)
        logger.addHandler(file_handler)


        logger.setLevel(logging.INFO)
        # fh = logging.FileHandler('run.log')
        # fh.setLevel(logging.DEBUG)
        # logger.addHandler(fh)
    else:
        # in this case, MPI is available but it's not rank 0
        # create a null handler
        handler = logging.NullHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
