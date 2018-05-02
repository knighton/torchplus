"""
The corresponding TorchPlus layers to PyTorch's `torch.nn`.

They look identical but perform various magic behind the scenes to enable "+"
sequences, layers without parentheses, etc.
"""

from inspect import isclass
import sys
import torch
from torch.nn.modules import Module

from . import extra
from .internal import Keyword


def wrap_layer_classes(from_module, to_module):
    """
    Create a corresponding wrapper in to_module for each from_module nn.Module.
    """
    for class_name in dir(from_module):
        if class_name == 'Module':
            continue
        klass = getattr(from_module, class_name)
        if not isclass(klass):
            continue
        if not issubclass(klass, Module):
            continue
        assert not hasattr(to_module, class_name)
        wrapped = Keyword(klass)
        setattr(to_module, class_name, wrapped)


this_module = sys.modules[__name__]
wrap_layer_classes(torch.nn.modules, this_module)
wrap_layer_classes(extra, this_module)
