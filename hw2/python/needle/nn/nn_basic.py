"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), 
                                device=device, dtype=dtype) # (in, out)
        if bias:
            # 在处理一些bug时注意到：目前的写法很容易忽视没有broadcast的问题
            # numpy作为array_api在forward时会隐式broadcast
            # 导致缺少broadcast的问题只会在backward时暴露出来
             
            # 需要先reshape再创建Parameter，因为对Parameter reshape会返回Tensor而不是Parameter
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape((1,out_features)),
                                  device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X (b, in)
        ret = X @ self.weight # (b, out)
        if self.bias is not None:
            bias = ops.broadcast_to(self.bias, ret.shape)
            ret += bias
        return ret
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # note: do not flatten batch dimention
        if len(X.shape) <= 1:
            return X
        
        size = 1
        for dim in X.shape[1:]:
            size *= dim
        return ops.reshape(X, (X.shape[0], size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # logits (batch, out_dim)
        y_one_hot = init.one_hot(logits.shape[1], y) # (batch, out_dim)
        losses = ops.logsumexp(logits, (-1,)) - ops.summation(logits * y_one_hot, axes=(-1,)) #(batch, )
        # mean loss
        return ops.summation(losses) / losses.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        
        self.running_mean = init.zeros(dim, requires_grad=False, device=device, dtype=dtype)
        self.running_var = init.ones(dim, requires_grad=False, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            ex = x.sum(axes=(0, )) / x.shape[0] # (dim, )
            diff = x - ex.broadcast_to(x.shape) # (batch, dim)
            vx = (diff ** 2).sum(axes=(0,)) / x.shape[0] # (dim, )
            
            # running
            m = self.momentum
            self.running_mean = ((1 - m) * self.running_mean + m * ex).detach()
            self.running_var = ((1 - m) * self.running_var + m * vx).detach()
        
            std = ((vx + self.eps) ** 0.5).broadcast_to(x.shape) # (batch, dim)
            w_b = self.weight.broadcast_to(x.shape)
            b_b = self.bias.broadcast_to(x.shape)
            x_norm = diff / std
            return x_norm * w_b + b_b
        else:
            # running
            std = ops.broadcast_to((self.running_var + self.eps) ** 0.5, x.shape) # (batch, dim)
            w_b = ops.broadcast_to(self.weight, x.shape)
            b_b = ops.broadcast_to(self.bias, x.shape)
            x_norm = (x - ops.broadcast_to(self.running_mean, x.shape)) / std
            return w_b * x_norm + b_b
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (batch, dim)
        ex = ops.summation(x, axes=(-1,)) / x.shape[-1] # (batch, )
        ex_b = ops.broadcast_to(ops.reshape(ex, ex.shape + (1,)), x.shape) # (batch, dim)
        diff = x - ex_b
        vx = ops.summation(ops.power_scalar(diff, 2), axes=(-1,)) / x.shape[-1] # (batch, )
        under_b = ops.broadcast_to(ops.reshape(ops.power_scalar(vx + self.eps, 0.5), vx.shape + (1,)), x.shape) # (batch, dim)
        w_b = ops.broadcast_to(self.weight, x.shape)
        b_b = ops.broadcast_to(self.bias, x.shape)
        return w_b * (diff / under_b) + b_b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # p is probability of masked neuron, not not-masked
            mask = init.randb(*(x.shape), p=(1-self.p), dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
