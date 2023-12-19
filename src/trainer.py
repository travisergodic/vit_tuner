import logging
import itertools
from tqdm import tqdm
from collections import defaultdict

import torch

from src.hooks import HOOKS

logger = logging.getLogger(__name__)

    
def to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    else:
        return x.to(device)

def to_numpy(x):
    if isinstance(x, dict):
        return {k: v.to("cpu").detach().numpy() for k, v in x.items()}
    else:
        return x.to("cpu").detach().numpy()


class Trainer:
    def __init__(self, model, iteration, optimizer, scheduler, loss_fn, evaluator, device, n_epochs, checkpoint_dir):
        self.device = device
        self.model = model.to(self.device)
        self.evalators = evaluator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iteration = iteration
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.checkpoint_dir=checkpoint_dir
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
                X, y = to_device(batch["data"], self.device), to_device(batch["targets"], self.device)
                self.call_hooks("before_train_iter")
                iter_info = self.iteration.run_iter(self, X, y)
                self.call_hooks("after_train_iter")

                pbar.set_postfix(loss=iter_info["loss"])

                for k in ("targets", "outputs"):
                    iter_train_records[k].append(to_numpy(iter_info[k])) 
                     
                # for k, v in iter_info.items():
                #     iter_train_records[k].append(to_device(v, "cpu").numpy())

                if self.scheduler is not None:
                    self.scheduler.step_update(self.epoch * num_steps + self.idx)

            train_metric_dict=self.evaluator.calculate(iter_train_records)
            train_metric_dict["epoch"]=self.epoch
            self.epoch_train_records.append(train_metric_dict)
            self.call_hooks("after_train_epoch")

            # evaluate
            if test_loader is not None:
                self.test(test_loader)
            self.call_hooks("after_test_epoch")
        self.call_hooks("after_run")

    @torch.no_grad()
    def test(self, test_loader):
        self.model.to(self.device)
        self.model.eval()

        iter_test_records = defaultdict(list)
        for batch in tqdm(test_loader):
            X, y = to_device(batch["data"], self.device), to_device(batch["targets"], self.device)
            self.call_hooks("before_test_iter")
            pred = self.model(X)
            self.call_hooks("after_test_iter")

            iter_test_records["targets"].append(to_numpy(y, "cpu"))
            iter_test_records["outputs"].append(to_numpy(pred, "cpu"))
            iter_test_records["loss"].append([self.loss_fn(pred, y).item()])

        test_metric_dict=self.evaluator.calculate(iter_test_records)
        test_metric_dict["epoch"]=self.epoch
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