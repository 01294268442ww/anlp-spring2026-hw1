from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                total_norm_sq = 0.0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    param_norm = p.grad.norm(2)
                    total_norm_sq += param_norm ** 2
                total_norm = total_norm_sq ** 0.5
                clip_coef = min(group["max_grad_norm"] / (total_norm + 1e-6), 1.0)
            else:
                clip_coef = 1.0
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                if clip_coef < 1:
                    with torch.no_grad():
                        p.grad.mul_(clip_coef)

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                beta1, beta2 = group["betas"]

                eps = group["eps"]

                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # TODO: Update first and second moments of the gradients
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2 
                # https://arxiv.org/pdf/1711.05101

                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    step_size = alpha / bias_correction1
                    denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
                else:
                    step_size = alpha
                    denom = exp_avg_sq.sqrt().add_(eps)

                # TODO: Update parameters and Add weight decay after the main gradient-based updates.
                with torch.no_grad():
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if weight_decay > 0:
                        p.add_(p, alpha=-alpha * weight_decay)
                
                # Please note that the learning rate should be incorporated into this update.

        return loss