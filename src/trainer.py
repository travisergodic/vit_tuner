import logging
import itertools
from tqdm import tqdm
from collections import defaultdict

import torch
from tensordict import TensorDict
from timm.utils import AverageMeter


from src.hooks import HOOKS


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, iteration, optimizer, scheduler, loss_fn, evaluator_dict, device, n_epochs, checkpoint_dir, eval_freq=1):
        self.device = device
        self.model = model.to(self.device)
        self.evaluator_dict = evaluator_dict
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iteration = iteration
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.checkpoint_dir=checkpoint_dir
        self.eval_freq = eval_freq
        self.epoch_train_records = []
        self.epoch_test_records = []
        self._hooks = []

    def train(self, train_loader, test_loader=None):
        self.call_hooks("before_run")
        num_steps=len(train_loader)
        for self.epoch in range(self.n_epochs): 
            self.model.train()
            pbar = tqdm(train_loader)
            pbar.set_description(f"Epoch {self.epoch}/{self.n_epochs}") 
            iter_train_records = defaultdict(list)
            
            for self.idx, batch in enumerate(pbar):
                X, y = batch["data"], batch["target"]
                if isinstance(y, dict):
                    y = TensorDict(y, batch_size=X.size(0)) 
                X, y = X.to(self.device), y.to(self.device)
                self.call_hooks("before_train_iter")
                iter_info = self.iteration.run_iter(self, X, y)
                self.call_hooks("after_train_iter")

                pbar.set_postfix(loss=iter_info["loss"])

                for k in ("target", "output"):
                    iter_train_records[k].append(iter_info[k].detach().to("cpu"))
                iter_train_records["loss"].append(iter_info["loss"])

                if self.scheduler is not None:
                    self.scheduler.step_update(self.epoch * num_steps + self.idx)

            train_metric_dict=self.evaluator_dict["train"].calculate(iter_train_records)
            train_metric_dict["epoch"] = self.epoch
            self.epoch_train_records.append(train_metric_dict)
            self.call_hooks("after_train_epoch")

            # evaluate
            if (test_loader is not None) and ((self.epoch + 1) % self.eval_freq == 0):
                self.test(test_loader)
            self.call_hooks("after_test_epoch")
        self.call_hooks("after_run")

    @torch.no_grad()
    def test(self, test_loader):
        self.model.to(self.device)
        self.model.eval()

        iter_test_records = defaultdict(list)
        for batch in tqdm(test_loader):
            X, y = batch["data"].to(self.device), batch["target"].to(self.device)
            self.call_hooks("before_test_iter")
            pred = self.model(X)
            self.call_hooks("after_test_iter")

            iter_test_records["target"].append(y.to("cpu"))
            iter_test_records["output"].append(pred.to("cpu"))
            iter_test_records["loss"].append(self.loss_fn(pred, y).item())

        test_metric_dict=self.evaluator_dict["test"].calculate(iter_test_records)
        test_metric_dict["epoch"] = self.epoch 
        self.epoch_test_records.append(test_metric_dict)

    def call_hooks(self, name):
        for hook in self._hooks:
            getattr(hook, name)(self)

    def register_hooks(self, hook_cfg_list):
        default_hook_cfg_list=[
            dict(type='CheckpointHook', top_k=1, monitor="loss", rule="less", save_begin=1),
        ]

        for i, default_hook_cfg in enumerate(default_hook_cfg_list):
            if default_hook_cfg["type"] in (cfg["type"] for cfg in hook_cfg_list):
                del default_hook_cfg[i]

        for hook_cfg in itertools.chain(default_hook_cfg_list, hook_cfg_list):
            self._hooks.append(HOOKS.build(**hook_cfg))
        self._hooks.sort(key=lambda x: x.priority, reverse=True)