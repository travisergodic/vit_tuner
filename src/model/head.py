import torch.nn as nn

from src.registry import HEAD


@HEAD.register
class SingleOutputHead(nn.Module):
    def __init__(self, in_features=768, out_features=512, dropout=0.0):
        super(SingleOutputHead, self).__init__()  
        self.in_features = in_features
        self.out_feature = out_features
        self.linear_layer = nn.Linear(in_features, out_features)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, X):
        return self.linear_layer(self.dropout_layer(X))


@HEAD.register
class MultiOutputHead(nn.Module):
    def __init__(self, in_features=2048, out_features_list=[512, 512], dropout=0.0):
        super(MultiOutputHead, self).__init__() 
        self.in_features = in_features
        self.out_features_list = out_features_list
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.in_features, out_features) for out_features in self.out_features_list
        ])
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def forward(self, X):
        X = self.dropout_layer(X)
        outputs = []
        for linear_layer in self.linear_layers:
            outputs.append(linear_layer(X))
        return outputs