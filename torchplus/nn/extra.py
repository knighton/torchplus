"""
Miscellaneous "batteries included" PyTorch layers.

These layers make life easier in an environment where you lean on sequences more
for defining models (because of the "+" syntax).
"""

from torch import nn


class Flatten(nn.Module):
    """
    Flatten batches.
    """

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    """
    Reshape batches.
    """

    def __init__(self, *batch_shape):
        nn.Module.__init__(self)
        self.batch_shape = batch_shape

    def forward(self, x):
        shape = (x.shape[0],) + self.batch_shape
        return x.view(shape)
