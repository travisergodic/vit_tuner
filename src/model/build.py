import torch
import torch.nn as nn 

from .backbone import BACKBONE
from .head import HEAD


def load_model(model, checkpoint_path):
    return model.load_state_dict(torch.load(checkpoint_path))


def build_model(model_cfg):
    backbone_cfg, head_cfg = model_cfg["backbone"], model_cfg["head"]
    backbone = BACKBONE.build(**backbone_cfg)
    head = HEAD.build(**head_cfg)
    model = ClassificationModel(backbone, head)
    if model_cfg.get("checkpoint_path") is not None:
        load_model(model, model_cfg["checkpoint_path"])
    return model


class ClassificationModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, X):
        X = self.backbone(X)
        X = self.head(X)
        print(type(X["expansion"]))
        return X