from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=True) # keep dim so maxz broadcast currectly
        sumz = array_api.sum(array_api.exp(Z - maxz), axis=self.axes, keepdims=False) # sumz should have same shape as non-keep-dim maxz
        return array_api.log(sumz) + array_api.reshape(maxz, sumz.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # we consider max Z as a scaling factor, not a value that the gradient should flow to
        # grad = exp(Z - maxZ) / exp(node - maxZ) shape same as Z
        # out_grad shape may be different, should be same as keep dim max Z to align axis
        Z = node.inputs[0]
        # print(f'Z with shape {Z.shape}')
        maxZ = Tensor(array_api.max(Z.cached_data, axis=self.axes, keepdims=True), requires_grad=False)
        # print(f'max Z with shape {maxZ.shape}')
        grad = exp(Z - broadcast_to(maxZ, Z.shape)) / broadcast_to(exp(reshape(node, maxZ.shape) - maxZ), Z.shape)
        # print(f'node with shape {node.shape}')
        # print(f'grad with shape {grad.shape}')
        return broadcast_to(reshape(out_grad, maxZ.shape), grad.shape) * grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

