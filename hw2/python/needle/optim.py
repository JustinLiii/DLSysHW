"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # momentum
        beta = self.momentum
        wd = self.weight_decay
        if len(self.u) == 0:
            self.u = {p: ((1 - beta) * (p.grad + wd * p)).data for p in self.params}
            # self.u = {p: 0 for p in self.params}
        else:
            self.u = {p: (beta * self.u[p] + (1 - beta) * (p.grad + wd * p)).data
                      for p in self.params}
        # print("updated momentum")
        # for p, m in zip(self.params, self.u):
        #     print(f'p: {p.shape}, m: {m.shape}')
        # step
        for p in self.params:
            p.data = (p - self.lr * self.u[p]).data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        wd = self.weight_decay
        # 1st degree momentum
        beta1 = self.beta1
        if len(self.m) == 0:
            self.m = {p: ((1 - beta1) * (p.grad + wd * p)).data for p in self.params}
        else:
            self.m = {p: (beta1 * self.m[p] + (1 - beta1) * (p.grad + wd * p)).data
                      for p in self.params}
        
        # 2nd
        beta2 = self.beta2
        if len(self.v) == 0:
            self.v = {p: ((1 - beta2) * (p.grad+ wd * p) ** 2 ).data for p in self.params}
        else:
            self.v = {p: (beta2 * self.v[p] + (1 - beta2) * (p.grad + wd * p) ** 2 ).data
                      for p in self.params}
            
        # bias correction
        self.t += 1
        m_hat = {p: self.m[p] / (1 - beta1 ** self.t) for p in self.params}
        v_hat = {p: self.v[p] / (1 - beta2 ** self.t) for p in self.params}
        
        # update
        for p in self.params:
            p.data = (p - self.lr * m_hat[p] / (v_hat[p] ** 0.5 + self.eps)).data
        ### END YOUR SOLUTION
