# JAX_QMC_2

This repository is the public, scalable, JAX-based implementation of the software supporting the following papers:

- [Variational Monte Carlo Calculations of A≤4 Nuclei with an Artificial Neural-Network Correlator Ansatz](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.022502)
- [Nuclei with Up to A=6 Nucleons with Artificial Neural Network Wave Function](https://link.springer.com/article/10.1007/s00601-021-01706-0)

The results in these publications are possible to reproduce with the release v1.0 of the code.

## Requirements
This repository is built on [JAX](https://github.com/lanpa/tensorboardX), with several extensions:
- [Flax](https://github.com/google/flax), for the neural networks,
- [TensorboardX](https://github.com/lanpa/tensorboardX), for tracking metrics during training,
- [Hydra](https://hydra.cc/docs/intro/) for configuration,
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) and [mpi4jax](https://github.com/mpi4jax/mpi4jax) for distributed operations.

Because JAX is a GPU-enabled library, this code can be run on any system that supports these libraries.

## Installation

We recommend you follow the JAX instructions for installing an performant version of JAX on your system.  After that, you can install most of the remaining requirements with `pip install -r requirements.txt`.  The mpi requirements, if you want to use them, should be installed with some care to use the right MPI for your system, including GPU-aware MPI if you are using GPUs.

## Running the code:

The code is configured via hydra/omegaconf, and most of the core configuration is set in yaml files located at `jax_qmc/config/recipes`.  You can override on the command line to modify these configs as needed.  Once configured, you can run (Helium, for example), like so:

```bash
mpiexec -n 4 -ppn 4 --cpu-bind=numa \
python bin/sr.py --config-name=4He \
run_id=test-run \
iterations=750 \
distributed=True sampler.n_walkers=80000
```

Each run requires a `run_id` to set the log locations, and if you are not using multiple devices set `distributed=False` to skip collective operations.  When the code starts, it will print out the configuration and then you will see code that looks like this:
```bash

2023-04-07 21:59:34,307 - INFO - Number of parameters in this network: 9245
2023-04-07 21:59:34,307 - INFO - Initializing spin.
2023-04-07 21:59:35,658 - INFO - Spin swaps completed in 1.2809 seconds.
2023-04-07 21:59:35,660 - INFO - Done initializing spin.
2023-04-07 21:59:35,914 - INFO - Failed to load weights!
2023-04-07 21:59:35,914 - INFO - 
2023-04-07 21:59:35,914 - INFO - Broadcasting initial model.
2023-04-07 21:59:36,817 - INFO - Done broadcasting initial model and optimizer state.
2023-04-07 21:59:36,818 - INFO - About to thermalize.
2023-04-07 21:59:46,961 - INFO - Finished thermalization with acceptance 0.4312 (x), 0.3996 (spin).
2023-04-07 22:00:30,193 - INFO - step  = 0, energy = 41.464, err = 0.120
2023-04-07 22:00:30,194 - INFO - step  = 0,     pe = -26.585
2023-04-07 22:00:30,194 - INFO - step  = 0,     ke = 68.049
2023-04-07 22:00:30,194 - INFO - step  = 0, e_diff = 0.000 (realized - predicted)
2023-04-07 22:00:30,194 - INFO - step  = 0,      r = 3.646
2023-04-07 22:00:30,194 - INFO - step  = 0,  acc_x = 0.431
2023-04-07 22:00:30,195 - INFO - step  = 0,  acc_s = 0.399
2023-04-07 22:00:30,195 - INFO - time = 43.227
2023-04-07 22:00:52,218 - INFO - step  = 1, energy = 37.985, err = 0.113
2023-04-07 22:00:52,219 - INFO - step  = 1,     pe = -26.979
2023-04-07 22:00:52,219 - INFO - step  = 1,     ke = 64.964
2023-04-07 22:00:52,219 - INFO - step  = 1, e_diff = -0.115 (realized - predicted)
2023-04-07 22:00:52,219 - INFO - step  = 1,      r = 3.684
2023-04-07 22:00:52,219 - INFO - step  = 1,  acc_x = 0.439
2023-04-07 22:00:52,219 - INFO - step  = 1,  acc_s = 0.397
2023-04-07 22:00:52,219 - INFO - time = 22.020
2023-04-07 22:00:56,150 - INFO - step  = 2, energy = 34.872, err = 0.108
2023-04-07 22:00:56,150 - INFO - step  = 2,     pe = -27.279
2023-04-07 22:00:56,150 - INFO - step  = 2,     ke = 62.150
2023-04-07 22:00:56,150 - INFO - step  = 2, e_diff = -0.030 (realized - predicted)
2023-04-07 22:00:56,151 - INFO - step  = 2,      r = 3.729
2023-04-07 22:00:56,151 - INFO - step  = 2,  acc_x = 0.447
2023-04-07 22:00:56,151 - INFO - step  = 2,  acc_s = 0.395
2023-04-07 22:00:56,151 - INFO - time = 3.927
....
```

The time per step will vary based on the problem size configured and the hardware used.  More details of the run, optimizer, and other parts of the algorithm can be found in tensorboard, which will be automatically generated for you in the output directory (which you can find at `outputs`).
