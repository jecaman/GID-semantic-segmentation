import torch
from torch.autograd import Variable
import numpy as np
from collections.abc import Mapping
from collections.abc import Sequence

__all__ = ['as_variable', 'as_numpy', 'mark_volatile']

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, Sequence):
        return [as_variable(v) for v in obj]
    elif isinstance(obj, Mapping):
        return {k: as_variable(v) for k, v in obj.items()}
    else:
        return Variable(obj)

def as_numpy(obj):
    if isinstance(obj, Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def mark_volatile(obj):
    if torch.is_tensor(obj):
        obj = Variable(obj)
    if isinstance(obj, Variable):
        obj.no_grad = True
        return obj
    elif isinstance(obj, Mapping):
        return {k: mark_volatile(o) for k, o in obj.items()}
    elif isinstance(obj, Sequence):
        return [mark_volatile(o) for o in obj]
    else:
        return obj
