from . stochastic_reconfiguration import sr_step

from . gradients import natural_gradients
from . gradients import par_dist
from . gradients import cholesky_solve, cg_solve

from . optimizers import close_over_optimizer
# from . optimizers import unflatten_weights_or_gradients
from . observables import mean_r
import optax

def build_lr_schedule(lr_cfg):

    schedules  = []
    boundaries = []
    running_total = 0

    for init_value, end_value, steps in zip(lr_cfg.init_values, lr_cfg.end_values, lr_cfg.steps):

        schedules.append(
            optax.linear_schedule(
                init_value = init_value,
                end_value  = end_value,
                transition_steps = steps,
                )
            )
        boundaries.append(steps + running_total)
        running_total += steps

    return optax.join_schedules(schedules=schedules, boundaries = boundaries[:1])