# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model. <https://arxiv.org/abs/1810.04805>"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging

logger = logging.getLogger(__name__)

def warmup_cosine(x, warmup=0.002):#在预热期内，学习率线性增加；预热期之后，学习率按照余弦函数衰减，最终趋近于0。
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002): #在预热期内，学习率线性增加；预热期之后，学习率保持不变。
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002): #在预热期内，学习率线性增加；预热期之后，学习率线性衰减，最终降为0
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

SCHEDULES = {
    'warmup_cosine':   warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear':   warmup_linear,
}


class BertAdam(Optimizer): #BERT版本的Adam优化算法，并修复了权重衰减问题。
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm) #首先检查参数的有效性，然后将参数保存到defaults字典中
        super(BertAdam, self).__init__(params, defaults)
 
    def get_lr(self):#获取当前的学习率
        lr = []
        for group in self.param_groups: #遍历参数组中的每个参数
            for p in group['params']:
                if p.grad is None:#如果参数的梯度为None，则跳过该参数。
                    continue
                #获取参数的状态，并根据状态、调度策略和其他参数计算学习率，
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled) #将学习率添加到lr列表
        return lr

    def step(self, closure=None): #执行单步优化
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """#接受一个可选的闭包参数closure，该闭包重新评估模型并返回损失。
        loss = None
        if closure is not None:
            loss = closure() #如果提供了closure，则调用它并将返回的损失赋值给loss。

        for group in self.param_groups: #参数组中的每个参数进行迭代
            for p in group['params']:
                if p.grad is None: #参数的梯度为None，则跳过该参数
                    continue
                grad = p.grad.data#获取参数的梯度数据。
                if grad.is_sparse:#如果梯度是稀疏的，抛出一个运行时错误，因为Adam不支持稀疏梯度
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]#获取参数的状态

                # State initialization
                if len(state) == 0: #如果状态为空，初始化状态
                    state['step'] = 0 #当前的优化步数，初始化为0。
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data) #梯度值的指数移动平均，初始化为与参数p具有相同形状和类型的零张量。
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data) #平方梯度值的指数移动平均，初始化为与参数p具有相同形状和类型的零张量

                next_m, next_v = state['next_m'], state['next_v'] #从状态中获取next_m和next_v。
                beta1, beta2 = group['b1'], group['b2'] #从参数组中获取Adam的超参数beta1和beta2。

                # Add grad clipping
                if group['max_grad_norm'] > 0: #如果设置了最大梯度范数，对梯度进行裁剪。
                    clip_grad_norm_(p, group['max_grad_norm']) #通过调用clip_grad_norm_函数实现的 #接受参数p和最大梯度范数作为参数

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # next_m.mul_(beta1).add_(1 - beta1, grad) --> pytorch 1.7
                #更新梯度的一阶矩（动量）和二阶矩（RMS) #过对next_m和next_v进行原地操作实现的
                #
                #add_()是原地操作，所以它会直接修改张量的值 #add()不会修改原始张量，而是返回一个新的张量。
                #
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1) #，next_m乘以beta1，然后加上(1 - beta1)乘以梯度
                # next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad) --> pytorch 1.7
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) #next_v乘以beta2，然后加上(1 - beta2)乘以梯度的平方。
                update = next_m / (next_v.sqrt() + group['e']) #计算参数的更新。 将next_m除以next_v的平方根加上epsilon 

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0: #设置了权重衰减，将权重衰减添加到更新中
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1: #设置了总训练步数，计算学习率调度。否则，使用基础学习率。
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step']/group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                else:
                    lr_scheduled = group['lr']
                #如果设置了总训练步数，计算学习率调度。否则，使用基础学习率
                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss