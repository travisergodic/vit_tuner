import logging
from tqdm import tqdm
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, iteration, optimizer, scheduler, loss_fn, evaluators, device, n_epochs):
        self.device = device
        self.model = model.to(self.device)
        self.evalators = evaluators
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iteration = iteration
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.epoch_train_records = defaultdict(list)
        self.epoch_test_records = defaultdict(list)

    def train(self, train_loader, test_loader=None):
        num_steps=len(train_loader)
        for self.epoch in range(self.n_epochs): 
            self.model.train()
            pbar = tqdm(train_loader)
            pbar.set_description(f"Epoch {self.epoch}/{self.n_epochs}") 
            iter_train_records = defaultdict(list)
            
            for self.idx, batch in enumerate(pbar):
                X, y = batch["data"].to(self.device), batch["targets"].to(self.device)
                iter_info = self.iter_hook.run_iter(self, X, y).item()

                pbar.set_postfix(loss=iter_info["loss"])
                for k, v in iter_info.items():
                    iter_train_records[k].append(v)

                if self.scheduler is not None:
                    self.scheduler.step_update(self.epoch * num_steps + self.idx)

            self.train_metric_dict=self.evaluators["train"].calculate(self, iter_train_records)
            self.call_hooks("after_train")

            # evaluate
            if test_loader is not None:
                self.test(test_loader)

            self.call_hooks("after_test")

    @torch.no_grad()
    def test(self, test_loader):
        self.model.to(self.device)
        self.model.eval()

        for batch in tqdm(test_loader):
            X, y, names = batch["data"].to(self.device), batch["targets"].to(self.device), batch["name"]
            pred = self.model(X)
            iter_test_records = defaultdict(list)

            iter_test_records["targets"].append(y.cpu().numpy())
            iter_test_records["outputs"].append(pred.cpu().numpy())
            iter_test_records["loss"].append([self.loss_fn(pred, y).item()])
        self.test_metric_dict=self.evalators["test"].calculate(self, iter_test_records)

    def call_hooks(self, name):
        for hook in self._hooks:
            getattr(hook, name)()

    def register_hooks(self, hook_cfg_list):
        pass 