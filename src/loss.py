import torch.nn as nn

from src.registry import LOSS


@LOSS.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true) * self.weight


@LOSS.register("multi_task")
class MultiTaskLoss(nn.Module):
    def __init__(self, task_to_loss_cfg, task_to_weight=None):
        super().__init__()
        self.task_to_loss_fn={
            task: LOSS.build(cfg) for task, cfg in task_to_loss_cfg.items()
        }
        self.task_to_weight={task: 1. for task in self.task_to_loss_fn} if task_to_weight is None else task_to_weight

    def forward(self, y_pred, y_true):
        total_loss = 0 
        for task, loss_fn in self.task_to_loss_fn.items():
            total_loss += loss_fn(y_pred[task], y_true[task])
        return total_loss