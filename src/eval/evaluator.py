import logging

import numpy as np
import pandas as pd

from src.registry import EVALUATOR
from src.eval.metric import METRIC


logger=logging.getLogger(__name__)


@EVALUATOR.register("multi_task")
class SingleTaskEvaluator:
    def __init__(self, metric_cfg_list):
        self.metrics = [METRIC.build(**metric_cfg) for metric_cfg in metric_cfg_list]

    def calculate(self, trainer, iter_records):
        eval_df=pd.DataFrame(
            {
                "outputs": np.concatenate(iter_records["outputs"], axis=0),
                "targets": np.concatenate(iter_records["targets"], axis=0),
                "losses": np.array(iter_records["loss"])
            }
        )
        metric_dict={str(metric): metric(eval_df) for metric in self.metrics}
        logger.info(", ".join([f"{k}: {v}" for k, v in metric_dict.items()]))
        return metric_dict
    

@EVALUATOR.register("single_task")
class MultiTaskEvaluator:
    def __init__(self, metric_cfg_list, task_names):
        self.metrics = [METRIC.build(**metric_cfg) for metric_cfg in metric_cfg_list]
        self.task_names=task_names

    def calculate(self, trainer, iter_records):
        eval_df=pd.DataFrame(
            {
                **{f"output_{name}": np.concatenate([ele[i] for ele in iter_records["output"]]) for i, name in enumerate(self.task_names)},
                **{f"targets_{name}": np.concatenate([ele[i] for ele in iter_records["output"]]) for i, name in enumerate(self.task_names)},
                "losses": np.array(iter_records["loss"])
            }
        )
        metric_dict={str(metric): metric(eval_df) for metric in self.metrics}
        logger.info(", ".join([f"{k}: {v}" for k, v in metric_dict.items()]))
        return metric_dict