import sys, os
import pathlib
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import signal
import pickle

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['PMI_LOCAL_RANK']

import jax
import jax.numpy as numpy
from jax import random
from jax import tree_util
import flax

import logging


# Add the local folder to the import path:
qmc_dir = os.path.dirname(os.path.abspath(__file__))
qmc_dir = os.path.dirname(qmc_dir) # + "/jax_qmc/"
sys.path.insert(0,qmc_dir)

from jax_qmc.config.wavefunction import GraphKind, AntisymmetryKind


from jax_qmc.spatial import spatial_initialization
from jax_qmc.spatial import multisplit
from jax_qmc.spatial import close_walk_over_wavefunction

from jax_qmc.wavefunction import init_wavefunction, init_jit_and_vmap_nn

from jax_qmc.optimization import close_over_optimizer
from jax_qmc.optimization import sr_step, build_lr_schedule

from jax_qmc.utils import init_mpi, discover_local_rank
from jax_qmc.utils import summary, model_summary, write_metrics
from jax_qmc.utils import init_checkpointer, save_walkers, create_state_snapshot_function
from jax_qmc.utils import set_compute_parameters, configure_logger, should_do_io

from tensorboardX import SummaryWriter


def interupt_handler( sig, frame):
    logger = logging.getLogger()

    logger.info("Finishing iteration and snapshoting weights...")
    global active
    active = False




@hydra.main(version_base = None, config_path="../jax_qmc/config/recipes")
def main(cfg : OmegaConf) -> None:

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")


    # Extend the save path:
    cfg.save_dir = cfg.save_dir + f"/{cfg.hamiltonian.form.name}/"
    cfg.save_dir = cfg.save_dir + f"/{cfg.sampler.n_particles}particles/"
    wavefunction = f"{cfg.wavefunction.antisymmetry.form.name}"
    if cfg.wavefunction.graph_net.form != GraphKind.NONE:
        wavefunction += f"-{cfg.wavefunction.graph_net.form.name}"
    cfg.save_dir = cfg.save_dir + f"/{wavefunction}/"
    cfg.save_dir = cfg.save_dir + f"/{cfg.run_id}/"


    # Prepare directories:
    work_dir = pathlib.Path(cfg.save_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(cfg.save_dir + "/log/")
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name = pathlib.Path(cfg["model_name"])


    MPI_AVAILABLE, rank, size = init_mpi(cfg.distributed)

    # Figure out the local rank if MPI is available:
    if MPI_AVAILABLE:
        local_rank = discover_local_rank()
    else:
        local_rank = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    configure_logger(log_dir, MPI_AVAILABLE, rank)

    logger = logging.getLogger()
    logger.info("")
    logger.info("\n" + OmegaConf.to_yaml(cfg))



    # Training state variables:
    global active
    active = True

    # Active is a global so that we can interrupt the train loop gracefully
    signal.signal(signal.SIGINT, interupt_handler)

    global_step = 0

    target_device = set_compute_parameters(local_rank)

    # Initialize the global random seed:
    if cfg.seed == -1:
        global_random_seed = int(time.time())
    else:
        global_random_seed = cfg.seed

    logger.info(f"Seed for this run is: {global_random_seed}.")

    # Handling randomness in a distributed world
    # There are a few points that spawn chains of random numbers
    # Most importantly, the walkers are an endless chain of random numbers
    # We want to ensure that each walker starts with the right random number,
    # and if that random number is on one rank or the other it shouldn't matter.
    # The total number of walkers will change things, but for 1000 walkers,
    # walker 501 will have the same starting point regardless if it's 1 GPU or 1000

    # First step: each rank needs the same _master_ key
    if MPI_AVAILABLE and size > 1:
        if rank == 0:
            # Create a single master key
            master_key = jax.device_put(random.PRNGKey(global_random_seed), target_device)
        else:
            # This isn't meaningful except as a placeholder:
            master_key = jax.device_put(random.PRNGKey(0), target_device)

        # Here, sync up all ranks to the same master key
        import mpi4jax
        from mpi4py import MPI
        master_key, token = mpi4jax.bcast(master_key, root=0, comm=MPI.COMM_WORLD)
    else:
        master_key = jax.device_put(random.PRNGKey(global_random_seed), target_device)

    # Now, everything has the same set of master keys.

    # We initialize the walker starting values on all nodes, and just throw away
    # most of it

    # Split the master key
    walker_key_seed, sub_master_key = random.split(master_key)

    # How many walkers on this device?
    assert cfg.sampler.n_walkers % size == 0, "Number of walkers must be evenly divisible."
    n_local_walkers = cfg.sampler.n_walkers / size

    # This gives the same keys on EVERY rank:
    global_walker_keys = random.split(walker_key_seed, cfg.sampler.n_walkers)

    # We don't actually need to initialize EVERY walker on every rank.
    # slice out the right keys:
    index_start = int(n_local_walkers * rank)
    index_end   = int(n_local_walkers * (rank + 1))

    walker_keys = global_walker_keys[index_start:index_end]

    TARGET_DTYPE = cfg.sampler.global_precision

    with jax.default_device(target_device):
        x, spin, isospin = spatial_initialization(walker_keys, cfg.sampler,  TARGET_DTYPE)
        
        # Normalize the inputs by number of particles:
        n_particles = cfg.sampler.n_particles
        norm = 1.2 * numpy.cbrt(n_particles)/numpy.sqrt(3.)
        x = x * norm

    # Here, we initialize the many body wave function.
    # Using exactly the same key, we get the same starting params on each device:
    network_key_seed, sub_master_key = random.split(sub_master_key)

    # This isn't the most elegant solution, but it kinda works.
    # Create a dictionary of functions to pass around and allow
    # semi-pure functional programing without a huge list of args

    wf = init_wavefunction(cfg.wavefunction, cfg.sampler)

    wf_str = wf.tabulate(network_key_seed, x[0], spin[0], isospin[0],
        console_kwargs={"width":120}, depth=1)

    logger.info(wf_str)

    w_params, wavefunction, g_fn, d2_fn, J_fn = \
        init_jit_and_vmap_nn(network_key_seed, x, spin, isospin, wf)

    # Ensure the wavefunction is in the right dtype
    w_params = tree_util.tree_map(
        lambda _x : _x.astype(TARGET_DTYPE),
        w_params
    )



    
    # n_parameters = 0
    # flat_params, tree_def = tree_util.tree_flatten(w_params)
    # logger.info(tree_def)
    
    # flat_params = numpy.concatenate([ p.reshape((-1,)) for p in flat_params])
    
    # for p in flat_params:
    #     n_parameters += numpy.prod(numpy.asarray(p.shape))
    
    # logger.info(f"Number of parameters in this network: {int(n_parameters)}")

    function_registry = {
        "wavefunction" : wavefunction,
        "first_deriv"  : g_fn,
        "second_deriv" : d2_fn,
        "jacobian"     : J_fn
    }

        
    logger.info("About to precompile wavefunction calls...")
    # All of the above functions have the same input signature, which is static now.
    # So, to save compilation time, AOT compile them and replace them in the dict:
    compiled_funcs = {}
    start = time.time()
    for func in function_registry.keys():
        # if func == "wavefunction": continue
        f = function_registry[func]
        f = jax.jit(f)
        f = f.lower(w_params, x, spin, isospin)
        f = f.compile()
        # print(f.cost_analysis())
        # print(f.memory_analysis())
        # .compile()
        compiled_funcs[func+"_comp"] = f
        logger.info(f"compiled {func} in {time.time() - start:.3f} s")
        start = time.time()

    function_registry.update(compiled_funcs)


    # Here, if necessary, Prepare the function to save the optimizer state:
    dirname = cfg.save_dir / pathlib.Path("opt_snapshots")
    function_registry["opt_snapshot"] = create_state_snapshot_function(
        dirname,
        rank,
        do_snapshotting = cfg.optimizer.make_snapshots
    )

    # Build the learning rate schedules for delta and epsilon:
    delta   = build_lr_schedule(cfg.optimizer.delta)
    epsilon = build_lr_schedule(cfg.optimizer.delta)

    optimizer, opt_init = close_over_optimizer(function_registry, cfg, delta, epsilon, MPI_AVAILABLE)

    function_registry["optimizer"] = optimizer

    # Initialize the base optimizer (with a FLAT list of parameters!):
    opt_state = opt_init(w_params)



    logger.info("Initializing spin.")
    spin_init_keys, next_seed_keys = multisplit(walker_keys)


    # Create a summary writer:
    if should_do_io(MPI_AVAILABLE, rank):
        writer = SummaryWriter(log_dir, flush_secs=20)
        metrics_file = log_dir / pathlib.Path("metrics.csv")
        metrics_file = open(metrics_file, 'a+')

    else:
        writer = None
        metrics_file = None

    # Load the hamiltonian:
    from jax_qmc.energy import build_energy_function
    compute_energy = build_energy_function(cfg, function_registry)

    # # Precompile the energy computation:
    # start = time.time()
    # logw_of_x, sign = function_registry["wavefunction_comp"](w_params, x, spin, isospin)
    # dlogw_dx   = function_registry["first_deriv_comp"]( w_params, x, spin, isospin)
    # d2logw_dx2 = function_registry["second_deriv_comp"](w_params, x, spin, isospin)



    # ce_f = ce_f.compile()
    # print(ce_f)
    # function_registry["energy"] = ce_f
    function_registry["energy"] = compute_energy
    # logger.info(f"compiled compute_energy in {time.time() - start:.3f} s")

    # We also snapshot the configuration into the log dir:
    if should_do_io(MPI_AVAILABLE, rank):
        with open(cfg.save_dir / pathlib.Path('config.snapshot.yaml'), 'w') as f_cfg:
            OmegaConf.save(config=cfg, f=f_cfg)

    # Here we initialize the checkpointer functions:
    target_dir = cfg.restore_path
    if target_dir == "": target_dir = None
    save_weights, restore_weights = init_checkpointer(cfg.save_dir, should_do_io=should_do_io(MPI_AVAILABLE, rank))
    # save_weights(w_params, opt_state, global_step)

    if should_do_io(MPI_AVAILABLE, rank):
        try:

            r_w_params, r_opt, r_global_step = restore_weights()
            # Catch the nothing returned case:
            assert r_global_step is not None
            assert r_w_params    is not None
            assert r_opt         is not None
            w_params    = r_w_params
            opt_state   = r_opt
            global_step = r_global_step
            logger.info("Loaded weights, optimizer and global step!")
        except Exception as excep:
            logger.info("Failed to load weights!")
            logger.info(excep)
            pass



    if MPI_AVAILABLE and size > 1:
        logger.info("Broadcasting initial model and opt state.")
        # We have to broadcast the wavefunction parameter here:
        token = None

        # First, flatten the parameter trees:
        w_params_flat, treedef = jax.tree_util.tree_flatten(w_params)

        # need to unfreeze to do this:
        for i, param in enumerate(w_params_flat):
            w_params_flat[i], token = mpi4jax.bcast(
                w_params_flat[i],
                root = 0,
                comm = MPI.COMM_WORLD,
                token = token
            )
        # And re-tree it:
        w_params = jax.tree_util.tree_unflatten(treedef, w_params_flat)
        # w_params = flax.core.frozen_dict.FrozenDict(r_w_params)

        # Now do the optimizer the same way:
        opt_state_flat, opt_treedef = jax.tree_util.tree_flatten(opt_state)

        # need to unfreeze to do this:
        for i, param in enumerate(opt_state_flat):
            opt_state_flat[i], token = mpi4jax.bcast(
                opt_state_flat[i],
                root  = 0,
                comm  = MPI.COMM_WORLD,
                token = token
            )
        # And re-tree it:
        opt_state = jax.tree_util.tree_unflatten(opt_treedef, opt_state_flat)


        # And the global step:
        global_step, token = mpi4jax.bcast(global_step,
                        root = 0,
                        comm = MPI.COMM_WORLD,
                        token = token)
        logger.info("Done broadcasting initial model and optimizer state.")



    precision = cfg.sampler.walk_precision
    mean_subtract = cfg.wavefunction.mean_subtract
    metropolis_walk = close_walk_over_wavefunction(wavefunction, precision, mean_subtract)

    # Here, do the training loop:
    #
    # First step - thermalize:
    logger.info("About to thermalize.")

    # Store the state of the walkers + their keys in a dict

    walker_keys, next_seed_keys = multisplit(next_seed_keys)

    acceptance, x, spin, isospin = metropolis_walk(
        walker_keys, w_params,
        numpy.asarray(cfg.sampler.kick_size),
        x, spin, isospin,
        cfg.sampler.n_thermalize)

    logger.info(f"Finished thermalization with acceptance {acceptance['x']:.4f} (x), {acceptance['spin']:.4f} (spin).")
    function_registry["metropolis"] = metropolis_walk


    if cfg.save_walkers:
        walker_dir = cfg.save_dir / pathlib.Path("walkers")
        save_walkers(x, spin, isospin, -1, walker_dir, rank )
        # Save the x/spin/isospin 

    checkpoint_iteration = 20

    # Before beginning the loop, manually flush the buffer:
    logger.handlers[0].flush()

    best_energy = 999
    predicted_energy = None




    while global_step < cfg["iterations"]:
        if not active: break
        start = time.time()

        if cfg.profile:
            if should_do_io(MPI_AVAILABLE, rank):
                jax.profiler.start_trace(str(cfg.save_dir) + "profile")
                # tf.profiler.experimental.start(str(cfg.save_dir))
                # tf.summary.trace_on(graph=True)


        # Split the keys:
        walker_keys, next_seed_keys = multisplit(next_seed_keys)

        w_params, opt_state, metrics, next_energy_pred, x, spin, isospin = sr_step(
            function_registry,
            w_params, opt_state, global_step,
            walker_keys, x, spin, isospin,
            cfg.sampler,
            world_size = size,
            MPI_AVAILABLE = MPI_AVAILABLE)
        
        if cfg.save_walkers:
                walker_dir = cfg.save_dir / pathlib.Path("walkers")
                save_walkers(x, spin, isospin, global_step, walker_dir, rank )
        
        if cfg.profile:
            if should_do_io(MPI_AVAILABLE, rank):
                x.block_until_ready()
                jax.profiler.save_device_memory_profile(str(cfg.save_dir) + f"memory{global_step}.prof")

        if cfg.optimizer.predict_energy and predicted_energy is not None:
            energy_diff = metrics['energy/energy'] - predicted_energy
        else:
            energy_diff = 0.0

        metrics["energy/energy_diff"] = energy_diff

        predicted_energy = next_energy_pred

        # Check if we've reached a better energy:
        if metrics['energy/energy'] < best_energy:
            best_energy = metrics['energy/energy']

        # # If below the target energy, snapshot the weights as the best-yet
        # if target_energy is None:
        #     pass
        # elif best_energy < target_energy:
        #     if should_do_io(MPI_AVAILABLE, rank):
        #         save_weights(cfg.save_dir, model_name, w_params, name="best_energy")

        metrics['time'] = time.time() - start
        if should_do_io(MPI_AVAILABLE, rank):
            summary(writer, metrics, global_step)
            write_metrics(metrics_file, metrics, global_step)

        # Add the gradients and model weights to the summary every 25 iterations:
        if global_step % 25 == 0:
            if should_do_io(MPI_AVAILABLE, rank):
                model_summary(writer, w_params, global_step)
                # self.wavefunction_summary(self.sr_worker.latest_psi, global_step)

        logger.info(f"step  = {global_step}, energy = {metrics['energy/energy']:.3f}, err = {metrics['energy/error']:.3f}")
        logger.info(f"step  = {global_step},     pe = {metrics['energy/pe']:.3f}")
        logger.info(f"step  = {global_step},     ke = {metrics['energy/ke_direct']:.3f}")
        if cfg.optimizer.predict_energy:
            logger.info(f"step  = {global_step}, e_diff = {metrics['energy/energy_diff']:.3f} (realized - predicted)")
        logger.info(f"step  = {global_step},      r = {metrics['metropolis/mean_sub_r']:.3f}")
        logger.info(f"step  = {global_step},  acc_x = {metrics['metropolis/accept_x']:.3f}")
        logger.info(f"step  = {global_step},  acc_s = {metrics['metropolis/accept_spin']:.3f}")
        logger.info(f"time = {metrics['time']:.3f}")

        # Iterate:
        global_step += 1

        if global_step % checkpoint_iteration  == 0:
            if should_do_io(MPI_AVAILABLE, rank):
                save_weights(w_params, opt_state, global_step)
                pass

        if cfg.profile:
            if should_do_io(MPI_AVAILABLE, rank):
                jax.profiler.stop_trace()
                # tf.profiler.experimental.stop()
                # tf.summary.trace_off()

    # Save the weights at the very end:
    if should_do_io(MPI_AVAILABLE, rank):
        try:
            save_weights(w_params, opt_state, global_step)
        except:
            pass




if __name__ == "__main__":
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += [
            'hydra/job_logging=disabled',
            'hydra.output_subdir=null',
            'hydra.job.chdir=False',
            'hydra.run.dir=.',
            'hydra/hydra_logging=disabled',
        ]


    main()
