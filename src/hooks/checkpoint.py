import os
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from src.registry import HOOKS
from src.utils import get_pretty_table


logger=logging.getLogger(__name__)


@HOOKS.register("checkpoint")
class CheckpointHook:
    rule_pick = {"greater": np.argmin, "less": np.argmax}
    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    best_save_fmt="best_e{epoch}-{value:.4f}.pt"
    last_save_fmt="last_e{epoch}-{value:.4f}.pt"

    def __init__(self, top_k, monitor, rule, save_begin=10, interval=5, save_last=True):
        self.top_k=top_k
        self.monitor=monitor
        self.iterval=interval
        self.rule=rule
        self.save_begin=save_begin
        self.save_last=save_last
        if self.top_k > 0:
            self._top_k_epoch=[]
            self._top_k_checkpoint=[]
            self._top_k_value=[]

    def after_test_epoch(self, trainer):
        if (trainer.epoch + 1 < self.save_begin) or (self.top_k <= 0):
            return 
        
        if (trainer.epoch + 1) % trainer.eval_freq != 0:
            return 

        Path(trainer.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        if len(self._top_k_checkpoint) >= self.top_k:
            idx = self.rule_pick[self.rule](self._top_k_value)
            if self.rule_map(trainer.epoch_train_records[-1][self.monitor], self._top_k_value[idx]):
                # remove
                checkpoint_file = self._top_k_checkpoint.pop(idx)
                os.remove(os.path.join(trainer.checkpoint_dir, checkpoint_file))
                logger.info(f"remove checkpoint file: {checkpoint_file}")
                del self._top_k_value[idx]
                del self._top_k_epoch[idx]
            else:
                return
        # save new
        self._top_k_value.append(trainer.epoch_train_records[-1][self.monitor])
        checkpoint_file=self.best_save_fmt.format(epoch=trainer.epoch, value=self._top_k_value[-1])
        self._top_k_checkpoint.append(checkpoint_file)
        self._top_k_epoch.append(trainer.epoch)
        torch.save(trainer.model.state_dict(), os.path.join(trainer.checkpoint_dir, checkpoint_file))
        logger.info(f"save checkpoint file: {checkpoint_file}")

        if self.save_last and (trainer.epoch + 1) % self.interval == 0:
            torch.save(trainer.model.state_dict(), os.path.join(trainer.checkpoint_dir, "last.pt"))

    def after_run(self, trainer):
        epoch_test_df = pd.DataFrame.from_records(trainer.epoch_test_records)
        print_df=epoch_test_df.loc[epoch_test_df["epoch"].isin(self._top_k_epoch), :].sort_values(by=self.monitor, ascending=False)
        logger.info(get_pretty_table(print_df, title="top_k ckpt"))