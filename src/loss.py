import torch.nn as nn

from src.registry import LOSS


@LOSS.register
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        total_loss = 0 
        for i, weight in enumerate(self.weights): 
            total_loss += self.loss_fn(y_pred[i], y_true[:, i]) * weight
        return total_loss