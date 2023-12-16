from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

from .linear import LinearLRScheduler
from .multistep import MultiStepLRScheduler

from src.registry import SCHEDULER


@SCHEDULER.register("cosine")
def cosine_lr_scheduler(
    optimizer, num_steps, warmup_steps, lr_min, warmup_lr_init, 
    cycle_limit=1, t_in_epochs=False, t_mul=1
):  
    return CosineLRScheduler(
        optimizer, lr_min=lr_min, t_initial=num_steps - warmup_steps,
        warmup_lr_init=warmup_lr_init, warmup_t=warmup_steps, cycle_limit=cycle_limit,
        t_mul=t_mul, t_in_epochs=t_in_epochs, warmup_prefix=True
    )


@SCHEDULER.register("linear")
def linear_lr_scheduler(
    optimizer, num_steps, warmup_lr_init, warmup_steps, lr_min_rate=0.01,  t_in_epochs=False   
):
    return LinearLRScheduler(
        optimizer,
        t_initial=num_steps,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_steps,
        lr_min_rate=lr_min_rate,
        t_in_epochs=t_in_epochs
    )


@SCHEDULER.register("step")
def step_lr_scheduler(
    optimizer, decay_steps, decay_rate, warmup_steps, warm_up_lr_init, t_in_epochs=False
):
    return StepLRScheduler(
        optimizer,
        decay_t=decay_steps,
        decay_rate=decay_rate,
        warmup_lr_init=warm_up_lr_init,
        warmup_t=warmup_steps,
        t_in_epochs=t_in_epochs,
    )