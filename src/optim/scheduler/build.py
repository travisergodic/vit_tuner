from timm.scheduler import cosine_lr, step_lr
from timm.scheduler.step_lr import StepLRScheduler

from src.registry import SCHEDULER


@SCHEDULER.register("cosine_lr")
def cosine_lr_scheduler(
    optimizer, n_iter_per_epoch, epochs, min_lr, warmup_lr_init,
    warmup_epochs, cycle_limit=1, warmup_prefix=True, t_in_epoch=False
):
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(warmup_epochs * n_iter_per_epoch)
    return cosine_lr.CosineLRScheduler(
        optimizer,
        t_initial=num_steps - warmup_steps,
        t_mul=1.,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr_init,
        warmup_prefix=warmup_prefix,
        warmup_t=warmup_steps,
        cycle_limit=cycle_limit,
        t_in_epochs=t_in_epoch,
    )