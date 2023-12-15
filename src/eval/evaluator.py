import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.registry import EVALUATOR
from src.eval.metric import METRIC


@EVALUATOR.register
class MultiTaskEvaluator:
    def __init__(self, model, device, metric_cfg_list, task_names, loss_fn=None):
        self.model = model
        self.device = device
        self.metrics = [METRIC.build(**metric_cfg) for metric_cfg in metric_cfg_list]
        self.task_names=task_names
        self.loss_fn = loss_fn

    @torch.no_grad()
    def get_eval_dataframe(self, loader):
        self.model.to(self.device)
        self.model.eval()

        prob_list_dict = {task: [] for task in self.task_names}
        target_list_dict = {task: [] for task in self.task_names}
        loss_list, name_list = [], []
        data_dict = {}

        # predict
        for batch in tqdm(loader):
            X, y, names = batch["data"].to(self.device), batch["label"].to(self.device), batch["name"]
            pred = self.model(X)
            for i, task in enumerate(self.task_names): 
                prob_list_dict[task].append(pred[i])
                target_list_dict[task].append(y[:, i])
                
            if self.loss_fn: 
                loss_list.extend([self.loss_fn(pred, y).item()] * len(names))
            
            name_list.extend(names)

        use_cols = ["name"] + [f"gt_{task}" for task in self.task_names] + [f"pred_{task}" for task in self.task_names]
        data_dict["name"] = np.array(name_list)

        for task in self.task_names:
            pred_arr = torch.cat(prob_list_dict[task], dim=0).softmax(dim=1).cpu().numpy()

            data_dict[f"pred_{task}"] = pred_arr.argmax(axis=1)
            data_dict[f"gt_{task}"] = torch.cat(target_list_dict[task], dim=0).cpu().numpy()
            
            if len(pred_arr.shape) == 2:
                data_dict[f"prob_vec_{task}"] = pred_arr.tolist()
                use_cols.append(f"prob_vec_{task}")     

        if self.loss_fn:    
            data_dict["loss"] = np.array(loss_list)
            use_cols.append(f"loss")

        return pd.DataFrame(data_dict)[use_cols]

    def get_eval_metric(self, loader):
        if self.metrics is None:
            raise ValueError()
        
        performance_df = self.get_eval_dataframe(loader)
        res_dict = {}
        
        for metric in self.metrics:
            res_dict[str(metric)] = metric(performance_df)
        
        if self.loss_fn:
            res_dict["loss"] = performance_df["loss"].values.mean()
        return res_dict
            

@EVALUATOR.register
class SingleTaskEvaluator:
    def __init__(self, model, device, metric_cfg_list=None, loss_fn=None):
        self.model = model
        self.device = device
        self.metrics = [METRIC.build(**metric_cfg) for metric_cfg in metric_cfg_list]
        self.loss_fn = loss_fn

    @torch.no_grad()
    def get_eval_dataframe(self, loader):
        self.model.to(self.device)
        self.model.eval()

        prob_list, target_list, name_list, loss_list = [], [], [], []
        data_dict = {}

        # create pair sample loader
        for batch in tqdm(loader):
            X, y, names = batch["data"].to(self.device), batch["label"].to(self.device), batch["name"]
            pred = self.model(X)
  
            prob_list.append(pred)
            target_list.append(y)
            name_list.extend(names)

            if self.loss_fn: 
                loss_list.extend([self.loss_fn(pred, y).item()] * len(names))

        use_cols = ["name", "gt", "pred"]
        data_dict["name"] = np.array(name_list)

        pred_arr = torch.cat(prob_list, dim=0).softmax(dim=1).cpu().numpy()
        data_dict["pred"] = pred_arr.argmax(axis=1)
        data_dict["gt"] = torch.cat(target_list, dim=0).cpu().numpy()

        if len(pred_arr.shape) == 2:
            data_dict[f"prob_vec"] = pred_arr.tolist()
            use_cols.append(f"prob_vec")     

        if self.loss_fn:
            data_dict["loss"] = np.array(loss_list)
            use_cols.append("loss")
        return pd.DataFrame(data_dict)[use_cols]

    def get_eval_metric(self, loader):
        if self.metrics is None:
            raise ValueError()
        
        eval_df = self.get_eval_dataframe(loader)
        res_dict = {}
        for metric in self.metrics:
            res_dict[str(metric)] = metric(eval_df)

        if self.loss_fn:
            res_dict["loss"] = eval_df["loss"].values.mean()
        return res_dict