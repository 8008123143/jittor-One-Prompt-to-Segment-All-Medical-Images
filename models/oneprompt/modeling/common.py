"""
Common basic modules, migrated from PyTorch to Jittor.
"""

import jittor as jt
from jittor import nn


class Adapter(nn.Module):
    """Adapter module for feature adaptation."""
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def execute(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    """MLP block with GELU activation."""
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def execute(self, x: jt.Var) -> jt.Var:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """2D Layer Normalization (channel-wise normalization)."""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = jt.ones(num_channels)
        self.bias = jt.zeros(num_channels)
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        u = x.mean(1, keepdims=True)
        s = (x - u).pow(2).mean(1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
