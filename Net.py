import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, mid_features * 1))
        self.layers.append(nn.Linear(mid_features * 1, mid_features * 2))
        self.layers.append(nn.Linear(mid_features * 2, mid_features * 2))
        self.layers.append(nn.Linear(mid_features * 2, out_features))
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        x = inputs
        for layer in self.layers[:-1]:  # Проходим через все, кроме последнего слоя
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x