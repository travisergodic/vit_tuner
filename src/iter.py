import torch
from torch import inf

from src.registry import ITERATION


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = []
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        # total_norm += param_norm.item() ** norm_type
        total_norm.append(param_norm)
    # total_norm = total_norm ** (1. / norm_type)
    total_norm = torch.stack(total_norm).norm(norm_type).item()
    return total_norm



@ITERATION.register("normal")
class NormalIteration:
    def __init__(self, accumulate_steps, clip_grad, enable_amp=False):
        self._scaler = torch.cuda.amp.GradScaler()
        self.accumulate_steps=accumulate_steps
        self.clip_grad=clip_grad
        self.enable_amp=enable_amp

    def run_iter(self, trainer, X, y):
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            pred=trainer.model(X)
        
        if self.accumulate_steps > 1:
            loss = trainer.loss_fn(pred, y)
            loss = loss / self.accumulate_steps
            self._scaler.scale(loss).backward()
            if (trainer.idx + 1) % self.accumulate_steps == 0:
                if self.clip_grad: 
                    self._scaler.unscale_(trainer.optimizer)
                    grad_norm=torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.clip_grad)
                else:
                    grad_norm=get_grad_norm(trainer.model.parameter())
                self._scaler.step(trainer.optimizer)
                trainer.optimizer.zero_grad()
                self._scaler.update()
            else:
                grad_norm=get_grad_norm(trainer.model.parameter())

        else:
            loss = trainer.loss_fn(pred, y)
            trainer.optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            if self.clip_grad:
                self._scaler.unscale_(trainer.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.clip_grad)
            else:
                grad_norm = get_grad_norm(trainer.model.parameters())
            self._scaler.step(trainer.optimizer)
            self._scaler.update()
        return dict(
            loss=loss.item(), grad_norm=grad_norm, target=y, 
            output=pred, loss_scale=self._scaler.get_scale()
        )