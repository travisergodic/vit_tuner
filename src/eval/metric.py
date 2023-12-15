import numpy as np 
from sklearn import metrics

from src.registry import METRIC


class BaseMetric:
    def __init__(self, **kwargs):
        self.tag=kwargs.pop("tag", None)
        self.metric_cfg=kwargs

    def __repr__(self) -> str:
        if self.tag:
            return self.tag
        return self.__class__.__name__ + "(" + ", ".join([f"{k}={v}" for k, v in self.cfg.items()]) + ")"


@METRIC.register
class Accuracy(BaseMetric):
    def __init__(self, gt_col, pred_col, **kwargs):
        super().__init__(**kwargs)
        self.gt_col=gt_col
        self.pred_col=pred_col
        
    def __call__(self, df):
        return metrics.accuracy_score(df[self.gt_col].values, df[self.pred_col].values, **self.metric_cfg)


@METRIC.register
class Recall(BaseMetric):
    def __init__(self, gt_col, pred_col, **kwargs):
        super().__init__(**kwargs)
        self.gt_col=gt_col
        self.pred_col=pred_col
    
    def __call__(self, df):
        return metrics.recall_score(df[self.gt_col].values, df[self.pred_col].values, **self.metric_cfg)


@METRIC.register
class Precision(BaseMetric):
    def __init__(self, tag, gt_col, pred_col, **kwargs):
        super().__init__(**kwargs)
        self.gt_col=gt_col
        self.pred_col=pred_col

    def __call__(self, df):
        return metrics.precision_score(df[self.gt_col].values, df[self.pred_col].values, **self.metric_cfg)


@METRIC.register("AUC")
class AUC(BaseMetric):
    def __init__(self, gt_col, prob_col, **kwargs):
        super().__init__(**kwargs)
        self.gt_col=gt_col
        self.prob_col=prob_col
    
    def __call__(self, df):
        prob = np.stack(df[self.prob_col].values, axis=0)
        return metrics.roc_auc_score(df[self.gt_col].values, prob, **self.metric_cfg)