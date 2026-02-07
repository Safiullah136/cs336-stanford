import torch
import numpy as np
from torch import nn, Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import math


def cross_entropy(logits: Tensor, targets: Tensor):
    stable_logits = logits - logits.max(dim=-1, keepdim=True).values
    sum_exp_logit = torch.sum(stable_logits.exp(), dim=-1)

    return (-stable_logits[torch.arange(targets.size(0)), targets] + torch.log(sum_exp_logit)).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            b1 = betas[0]
            b2 = betas[1]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data 
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * (grad ** 2)

                current_lr = lr * math.sqrt( 1 - (b2) ** t ) / ( 1 - (b1) ** t )
                p.data -= current_lr * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
    

def lr_cosine_schedule(it, lr_max, lr_min, warmup_iters, cosine_iters):
    if it < warmup_iters: return (it / warmup_iters) * lr_max 
    elif it > cosine_iters: return lr_min
    else: return lr_min + 0.5 * (1 + math.cos( math.pi * (it - warmup_iters) / (cosine_iters - warmup_iters) ) ) * (lr_max - lr_min)


def gradient_clipping(params, max_l2_norm):
    eps = 1e-6
    total_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params if p.grad is not None)) 
    
    clip_coeff = max_l2_norm / (total_norm + eps)
    
    if clip_coeff < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coeff)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data 
                p.data -= lr / math.sqrt(t + 1) * grad 
                state["t"] = t + 1
        return loss
    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=math.exp(3))
    for t in range(10):
        opt.zero_grad() 
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() 
        opt.step() 