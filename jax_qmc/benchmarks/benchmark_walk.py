import jax.numpy as numpy
import jax.random as random

import sys, os

# Add the local folder to the import path:
qmc_dir = os.path.dirname(os.path.abspath(__file__))
qmc_dir = os.path.dirname(qmc_dir)
sys.path.insert(0,qmc_dir)

import logging
from logging import handlers
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
handler = handlers.MemoryHandler(capacity = 1, target=stream_handler)
logger.addHandler(handler)


import argparse
parser = argparse.ArgumentParser(description="Benchmarking tool for JAX_QMC")
parser.add_argument("-b", "--benchmark",
    help="Name of benchmark",
    required=True,
    type=str,
    choices=["walk", "jacobian", "derivatives"])

args = vars(parser.parse_args())

from config import Sampler, ManyBodyCfg

import time

from spatial import spatial_initialization
from spatial import select_and_exchange_spins
from spatial import metropolis_walk
from spatial import initialize_spin_until_non_zero

from wavefunction import init_many_body_wf

# Create a sampler:
benchmark_sampler_config = Sampler()
benchmark_sampler_config.n_walkers   = 5000
benchmark_sampler_config.n_particles = 12
benchmark_sampler_config.n_dim       = 3
benchmark_sampler_config.n_spin_up   = 4
benchmark_sampler_config.n_protons   = 6

# Network config:
benchmark_network_config = ManyBodyCfg()

benchmark_seed = 1234

nparticles = benchmark_sampler_config.n_particles
k = random.PRNGKey(int(benchmark_seed))

k, s = random.split(k)
x, spin, isospin = spatial_initialization(s, benchmark_sampler_config,  "float64")

k, s = random.split(k)
w_params, wavefunction, compute_derivatives, compute_jacobian = \
    init_many_body_wf(s, x, spin, isospin, benchmark_sampler_config, benchmark_network_config)
logger.setLevel(logging.DEBUG)


logger.info("Initializing spin")
spin = initialize_spin_until_non_zero(
    s, x, spin, isospin, wavefunction, w_params)


k, s = random.split(k)

if args["benchmark"] == "walk":
    nkicks = 50
    # Run it once as a warm up:
    start = time.time()
    logger.info("Run warm up")
    accept_list, x, spin, isospin = \
        metropolis_walk(s, wavefunction, w_params, x, spin, isospin, nkicks)

    logger.info(f"Warm up completed in {time.time() - start:.4f} seconds")

    logger.info("Run benchmark")
    times = []
    for i in range(5):
        start = time.time()
        metropolis_walk(s, wavefunction, w_params, x, spin, isospin, nkicks)
        end = time.time()
        logger.info(f"Finished run {i}")
        times.append(end - start)

    times = numpy.asarray(times)

    logger.info(f"Mean metropolis walk time: {numpy.mean(times):.4f} +/- {numpy.std(times):.4f}")
    logger.info(f"  Min metropolis walk time: {numpy.min(times):.4f}")
    logger.info(f"  Max metropolis walk time: {numpy.max(times):.4f}")


if args["benchmark"] == "jacobian":

    logger.info("Run warm up")
    start = time.time()

    jac = compute_jacobian(w_params, x, spin, isospin)

    logger.info(f"Warm up completed in {time.time() - start:.4f} seconds")

    logger.info("Run benchmark")
    times = []
    for i in range(5):
        start = time.time()
        jac = compute_jacobian(w_params, x, spin, isospin)
        jac.block_until_ready()
        end = time.time()
        logger.info(f"Finished run {i}")
        times.append(end - start)

    times = numpy.asarray(times)

    logger.info(f"Mean jacobian time: {numpy.mean(times):.4f} +/- {numpy.std(times):.4f}")
    logger.info(f"  Min jacobian time: {numpy.min(times):.4f}")
    logger.info(f"  Max jacobian time: {numpy.max(times):.4f}")



if args["benchmark"] == "derivatives":

    logger.info("Run warm up")
    start = time.time()

    _ = compute_derivatives(w_params, x, spin, isospin)

    logger.info(f"Warm up completed in {time.time() - start:.4f} seconds")

    logger.info("Run benchmark")
    times = []
    for i in range(5):
        start = time.time()
        w_of_x, dw_dx, d2w_dx2 = compute_derivatives(w_params, x, spin, isospin)
        d2w_dx2.block_until_ready()
        end = time.time()
        logger.info(f"Finished run {i}")
        times.append(end - start)

    times = numpy.asarray(times)

    logger.info(f"Mean derivatives time: {numpy.mean(times):.4f} +/- {numpy.std(times):.4f}")
    logger.info(f"  Min derivatives time: {numpy.min(times):.4f}")
    logger.info(f"  Max derivatives time: {numpy.max(times):.4f}")
