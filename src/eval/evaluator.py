import logging

import torch
import numpy as np
import pandas as pd

from src.registry import EVALUATOR
from src.eval.metric import METRIC


logger=logging.getLogger(__name__)


@EVALUATOR.register("multi_task")
class SingleTaskEvaluator:
    def __init__(self, metric_cfg_list):
        self.metrics = [METRIC.build(**metric_cfg) for metric_cfg in metric_cfg_list]

    def calculate(self, iter_records):
        eval_df=pd.DataFrame(
            {
                "outputs": torch.concat(iter_records["outputs"], dim=0).numpy(),
                "targets": torch.concat(iter_records["targets"], dim=0).numpy(),
                "losses": np.array(iter_records["loss"])
            }
        )
        metric_dict={str(metric): metric(eval_df) for metric in self.metrics}
        logger.info(", ".join([f"{k}: {v}" for k, v in metric_dict.items()]))
        return metric_dict
    

@EVALUATOR.register("single_task")
class MultiTaskEvaluator:
    def __init__(self, metric_cfg_list):
        self.metrics = [METRIC.build(**metric_cfg) for metric_cfg in metric_cfg_list]

    def calculate(self, iter_records):
        task_names=self._get_task_names(iter_records)
        eval_df=pd.DataFrame(
            {
                **{f"prob_{name}": torch.concat([ele[name] for ele in iter_records["outputs"]], dim=0).numpy() for name in task_names},
                **{f"label_{name}": torch.concat([ele[name] for ele in iter_records["targets"]], dim=0).numpy() for name in task_names},
                **{f"pred_{name}": torch.concat([ele[name] for ele in iter_records["outputs"]], dim=0).numpy().argmax(axis=1) for name in task_names},
                "losses": np.array(iter_records["loss"])
            }
        )
        metric_dict={str(metric): metric(eval_df) for metric in self.metrics}
        logger.info(", ".join([f"{k}: {v}" for k, v in metric_dict.items()]))
        return metric_dict
    
    def _get_task_names(self, iter_records):
        return iter_records["output"][0].keys()
