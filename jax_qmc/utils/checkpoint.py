import sys, os
import pathlib

import jax.numpy as numpy

from flax.training import checkpoints

import orbax.checkpoint
from flax.training import orbax_utils
from flax.core import frozen_dict

def init_checkpointer(save_path, restore_path=None, should_do_io=False):

    if restore_path is None:
        restore_path = save_path

    restore_ckpt_path = pathlib.Path(restore_path) / pathlib.Path("checkpoint") / pathlib.Path("model")
    save_ckpt_path    = pathlib.Path(save_path)    / pathlib.Path("checkpoint") / pathlib.Path("model")
    
    restore_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_checkpointer    = orbax.checkpoint.PyTreeCheckpointer()

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    if should_do_io:
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            save_ckpt_path.resolve(),
            save_checkpointer,
            options
        )
        restore_manager = orbax.checkpoint.CheckpointManager(
            restore_ckpt_path.resolve(),
            restore_checkpointer,
            options
        )
    

        def save_weights(parameters, opt_state, global_step):

            ckpt = {
                'model' : parameters,
                'opt'   : opt_state,
            }
            save_args = orbax_utils.save_args_from_target(ckpt)

            checkpoint_manager.save(global_step, ckpt, save_kwargs={'save_args' : save_args})

            return


        def restore_weights():

            global_step = restore_manager.latest_step()

            checkpoint = restore_manager.restore(global_step)
            restored_model = checkpoint['model']
            # restored_model = target_type(checkpoint['model'])
            restored_opt   = checkpoint['opt']

            # This is a manual hack ...
            restored_opt["g2_i"] = restored_opt["g2_i"]
            restored_opt["m_i"]  = restored_opt["m_i"]
            # restored_opt["g2_i"] = frozen_dict.FrozenDict(restored_opt["g2_i"])
            # restored_opt["m_i"] = frozen_dict.FrozenDict(restored_opt["m_i"])

            return restored_model, restored_opt, global_step


        return save_weights, restore_weights
    
    else:

        return lambda * args, **kwargs : None, lambda * args, **kwargs : None


def save_walkers(x, spin, isospin, step, top_dir, rank):
    # This function is expected to write out one file per rank, per step.

    # First, decide the filename:
    dirname = top_dir / pathlib.Path(f"rank_{rank}")

    # Next, make the file directory if needed.
    # Doing rank / step so that each rank will be operating on fully unique posix paths.
    # step / rank will create a race condition to create the step directory.

    dirname.mkdir(parents=True, exist_ok=True)

    # Here is this rank's file:
    fname = dirname / pathlib.Path(f"step_{step}.npz")

    numpy.savez(str(fname), 
                x = x, 
                spin = spin, 
                isospin=isospin
                )

def create_state_snapshot_function(dirname, rank, do_snapshotting):
    

    if do_snapshotting:

        # First, decide the filename:
        import pathlib

        dirname = dirname / pathlib.Path(f"rank_{rank}")
        dirname.mkdir(parents=True, exist_ok=True)

        print("")
        def snapshop_function(step, **kwargs):
            # Next, make the file directory if needed.
            # Doing rank / step so that each rank will be operating on fully unique posix paths.
            # step / rank will create a race condition to create the step directory.


            # Here is this rank's file:
            fname = dirname / pathlib.Path(f"step_{step}.npz")
            numpy.savez(str(fname),
                **kwargs
            )
        
        return snapshop_function
    else:
        # return a no-op:
        no_op =  lambda *args, **kwargs: None

        return no_op
