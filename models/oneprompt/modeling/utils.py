"""
Modeling-specific utilities (Jittor version).
"""

import numpy as np
import jittor as jt
from jittor import nn


softmax_helper = lambda x: jt.nn.softmax(x, 1)
sigmoid_helper = lambda x: jt.nn.sigmoid(x)


class InitWeights_He(object):
    """He (Kaiming) weight initialization."""
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = jt.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = jt.init.constant_(module.bias, 0)


def maybe_to_torch(d):
    """Convert numpy array to jt.Var if needed."""
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, jt.Var) else i for i in d]
    elif not isinstance(d, jt.Var):
        d = jt.array(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    """Jittor manages devices automatically — this is a pass-through."""
    return data


class no_op(object):
    """No-operation context manager (for mixed precision compatibility)."""
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
