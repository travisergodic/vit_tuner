import torchvision
import torch.nn as nn

from src.registry import BACKBONE


@BACKBONE.register("resnet50")
def resnet50(pretrained=True):
    backbone = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    )
    backbone.fc = nn.Identity()
    return backbone