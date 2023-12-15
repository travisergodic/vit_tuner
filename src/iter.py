import torch
from torch import inf

from src.registry import ITER


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


@ITER_HOOK.register
class NormalIterHook:
    def run_iter(self, trainer, X, y):
        pred = trainer.model(X)
        loss = trainer.loss_fn(pred, y)
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        return loss


@ITER_HOOK.register
class SamIterHook:
    def run_iter(self, trainer, X, y):
        loss = trainer.loss_fn(trainer.model(X), y)  # use this loss for any training statistics
        loss.backward()
        trainer.optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        trainer.loss(trainer.model(X), y).backward()  # make sure to do a full forward pass
        trainer.optimizer.second_step(zero_grad=True)
        return loss
    

@ITER_HOOK.register
class AccumulateIterHook:
    def __init__(self, clip_grad=None, accum_iter=1):
        self._scaler = torch.cuda.amp.GradScaler()
        self.clip_grad = clip_grad
        self.accum_iter = accum_iter

    def run_iter(self, trainer, X, y):
        loss = trainer.loss_fn(trainer.model(X), y)

        loss /= self.accum_iter
        self._scaler.scale(loss).backward()
        
        if (trainer.iter + 1) % self.accum_iter == 0:
            if self.clip_grad is not None:
                self._scaler.unscale_(trainer.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.clip_grad)
            else:
                self._scaler.unscale_(trainer.optimizer)
                norm = get_grad_norm_(trainer.model.parameters())
            self._scaler.step(trainer.optimizer)
            self._scaler.update()

        if (trainer.iter + 1) % self.accum_iter == 0:
            trainer.optimizer.zero_grad()
        return loss  * self.accum_iter