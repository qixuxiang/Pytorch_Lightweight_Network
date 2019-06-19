import math
from collections.abc import Sequence, Mapping

import torch


class Args(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, tuple(args))

    def __repr__(self):
        return "Args" + super().__repr__()


def one_hot(tensor, C=None, dtype=torch.float):
    d = tensor.dim()
    C = C or tensor.max() + 1
    t = tensor.new_zeros(*tensor.size(), C, dtype=dtype)
    return t.scatter_(d, tensor.unsqueeze(d), 1)


CUDA = torch.cuda.is_available()


def detach(t, clone=True):
    if torch.is_tensor(t):
        if clone:
            return t.clone().detach()
        else:
            return t.detach()
    elif isinstance(t, Args):
        return t
    elif isinstance(t, Sequence):
        return t.__class__(detach(x, clone) for x in t)
    elif isinstance(t, Mapping):
        return t.__class__((k, detach(v, clone)) for k, v in t.items())
    else:
        return t


def cuda(t):
    if torch.is_tensor(t):
        return t.cuda() if CUDA else t
    elif isinstance(t, Sequence):
        return t.__class__(cuda(x) for x in t)
    elif isinstance(t, Mapping):
        return t.__class__((k, cuda(v)) for k, v in t.items())
    else:
        return t


def cpu(t):
    if torch.is_tensor(t):
        return t.cpu()
    elif isinstance(t, Sequence):
        return t.__class__(cpu(x) for x in t)
    elif isinstance(t, Mapping):
        return t.__class__((k, cpu(v)) for k, v in t.items())
    else:
        return t


def _tuple(x, n=-1):
    if x is None:
        return ()
    elif torch.is_tensor(x):
        return (x,)
    elif not isinstance(x, Sequence):
        assert n > 0, "Length must be positive, but got %d" % n
        return (x,) * n
    else:
        if n == -1:
            n = len(x)
        else:
            assert len(x) == n, "The length of x is %d, not equal to the expected length %d" % (len(x), n)
        return tuple(x)


def select0(t, indices):
    arange = torch.arange(t.size(1), device=t.device)
    return t[indices, arange]


def select1(t, indices):
    arange = torch.arange(t.size(0), device=t.device)
    return t[arange, indices]


def select(t, dim, indices):
    if dim == 0:
        return select0(t, indices)
    elif dim == 1:
        return select1(t, indices)
    else:
        raise ValueError("dim could be only 0 or 1, not %d" % dim)


def sample(t, n):
    if len(t) >= n:
        indices = torch.randperm(len(t), device=t.device)[:n]
    else:
        indices = torch.randint(len(t), size=(n,), device=t.device)
    return t[indices]


def _concat(xs, dim=1):
    if torch.is_tensor(xs):
        return xs
    elif len(xs) == 1:
        return xs[0]
    else:
        return torch.cat(xs, dim=dim)


def inverse_sigmoid(x, eps=1e-6, inplace=False):
    if not torch.is_tensor(x):
        if eps != 0:
            x = min(max(x, eps), 1-eps)
        return math.log(x / (1 - x))
    if inplace:
        return inverse_sigmoid_(x, eps)
    if eps != 0:
        x = torch.clamp(x, eps, 1-eps)
    return (x / (1 - x)).log()


def inverse_sigmoid_(x, eps=1e-6):
    if eps != 0:
        x = torch.clamp_(x, eps, 1 - eps)
    return x.div_(1 - x).log_()


def expand_last_dim(t, *size):
    return t.view(*t.size()[:-1], *size)