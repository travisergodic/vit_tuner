import torch.optim as optim

from .sam import SAM
from src.registry import OPTIMIZER


@OPTIMIZER.register
def Adam(**kwargs):
    return optim.Adam(**kwargs)


@OPTIMIZER.register
def AdamW(**kwargs):
    return optim.AdamW(**kwargs)


@OPTIMIZER.register
def SGD(**kwargs):
    return optim.SGD(**kwargs)


@OPTIMIZER.register
def SAM_Adam(**kwargs):
    return SAM(base_optimizer=optim.Adam, **kwargs)


@OPTIMIZER.register
def SAM_AdamW(**kwargs):
    return SAM(base_optimizer=optim.AdamW, **kwargs)


@OPTIMIZER.register
def SAM_SGD(**kwargs):
    return SAM(base_optimizer=optim.SGD, **kwargs)