import torch.nn as nn

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


@register('res-mlp')
class ResMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        assert in_dim == out_dim

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1])) + x
        return x.view(*shape, -1)


@register('linear')
class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(MyLinear, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, inp):
        return self.layer(inp)
