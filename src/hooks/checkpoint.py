import os
from pathlib import Path

import torch
import numpy as np

from src.registry import HOOKS


@HOOKS.register("checkpoint")
class CheckpointHook:
    rule_pick = {"greater": np.argmin, "less": np.argmax}
    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    best_save_fmt="best_epoch={epoch}-{value:.4f}.pt"

    def __init__(self, top_k, checkpoint_dir, monitor, rule, save_begin=10, interval=5, save_last=True):
        self.top_k=top_k
        self.checkpoint_dir=checkpoint_dir
        self.monitor=monitor
        self.iterval=interval
        self.rule=rule
        self.save_begin=save_begin
        self.save_last=save_last
        if self.top_k > 0:
            self._top_k_checkpoint=[]
            self._top_k_value=[]

    def save_top_k_checkpoint(self, trainer):
        if (trainer.epoch + 1 < self.save_begin) or (self.top_k <= 0):
            return 

        Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        if len(self._top_k_checkpoint) >= self.top_k:
            idx = self.rule_pick[self.rule](self._top_k_value)
            if self.rule_map(trainer.epoch_train_records[-1][self.monitor], self._top_k_value[idx]):
                # remove
                checkpoint_file = self._top_k_checkpoint.pop(idx)
                os.remove(os.path.join(self.checkpoint_dir, checkpoint_file))
                del self._top_k_value[idx]
            else:
                return
        # save new
        self._top_k_value.append(trainer.epoch_train_records[-1][self.monitor])
        checkpoint_file=self.best_save_fmt.format(epoch=trainer.epoch, value=self._top_k_value[-1])
        self._top_k_checkpoint.append(checkpoint_file)
        torch.save(trainer.model.state_dict(), os.path.join(self.checkpoint_dir, checkpoint_file))

        if self.save_last and (trainer.epoch + 1) % self.interval == 0:
            torch.save(trainer.model.state_dict(), os.path.join(self.checkpoint_dir, "last.pt"))