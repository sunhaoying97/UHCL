# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. <https://arxiv.org/abs/1810.04805>"""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.until_config import PretrainedConfig

logger = logging.getLogger(__name__)

def gelu(x):#激活函数 高斯误差线性单元
    #是一种平滑的激活函数，它可以在负输入值时产生接近于0的输出，而在正输入值时产生接近于输入值的输出。
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
 
def swish(x):#激活函数 swish函数是一种自门控的激活函数，它可以在负输入值时产生接近于0的输出，而在正输入值时产生接近于输入值的输出。
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        #hidden_size是隐藏层的大小，eps是一个很小的数，用于防止除以零的错误。
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        #
    def forward(self, x):
        u = x.mean(-1, keepdim=True) #x沿着最后一个维度的均值 并保持维度不变
        #x减去均值后的平方的均值，即方差。
        s = (x - u).pow(2).mean(-1, keepdim=True)
        #x减去均值后的结果除以标准差，得到了归一化后的x。
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        #将归一化后的x乘以权重，然后加上偏置，得到了最终的输出。
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):#PreTrainedModel类的权重初始化函数，它接受一个模块module作为参数。
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):#
            #检查module是否是nn.Linear或nn.Embedding类的实例。如果是，就使用正态分布来初始化其权重
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            #检查module是否是LayerNorm类的实例。如果是，就将其偏置初始化为0，将其权重初始化为1。
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        #检查module是否是nn.Linear类的实例，并且其偏置不为None。如果是，就将其偏置初始化为0
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):#用于调整令牌嵌入的大小。
        raise NotImplementedError

    @classmethod #一个装饰器，表示下面的方法是一个类方法。类方法是绑定到类的方法，而不是类的实例。类方法的第一个参数总是类本身，通常被命名为cls
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        #model是要初始化的模型，state_dict是包含预训练权重的字典，prefix是一个可选的字符串，用于在权重的键名前添加前缀，task_config是一个可选的配置对象。
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            #遍历state_dict的所有键。如果键中包含'gamma'或'beta'，则将其替换为'weight'或'bias'，并将新的键和旧的键添加到new_keys和old_keys列表中。
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        #遍历old_keys和new_keys列表。对于每一对旧键和新键，它将state_dict中对应旧键的值移动到新键。
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        #
        if prefix is not None: #如果prefix不为None，
            old_keys = []
            new_keys = []
            for key in state_dict.keys():#对于每一个键，将其添加到old_keys列表中，并将prefix加上该键后的字符串添加到new_keys列表中。
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):#对于每一对旧键和新键，将state_dict中对应旧键的值移动到新键。
                state_dict[new_key] = state_dict.pop(old_key)
        #用于存储在加载权重时可能出现的问题。
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)#获取state_dict的_metadata属性
        state_dict = state_dict.copy() #创建state_dict的一个副本。这样在修改state_dict时，不会影响原始的state_dict。
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''): #调用load函数来加载模型的权重。
            #一个内部函数，用于递归地加载模块的权重。它首先调用module._load_from_state_dict方法来加载权重，然后对模块的所有子模块递归地调用load函数。
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')#调用load函数来加载模型的权重。
        #这个条件检查是否需要打印日志信息。如果需要，它将打印出未从预训练模型中初始化的权重，未在模型中使用的预训练权重，以及在模型中引起错误的预训练权重。
        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property #装饰器，表示下面的方法是一个属性。 属性是一种特殊的方法，可以像访问数据属性一样访问它。
    def dtype(self): #它假设所有模块参数都有相同的数据类型，并返回这个数据类型。
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try: #尝试返回模块参数的数据类型。
            return next(self.parameters()).dtype #会获取模块的第一个参数的数据类型。
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                #这是一个内部函数，它接受一个模块作为参数，返回一个元组列表，每个元组包含一个属性名和一个张量
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples
            #这行代码创建了一个生成器，用于生成模块的所有张量属性。
            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen) #获取生成器的第一个元素，即模块的第一个张量属性。
            return first_tuple[1].dtype #返回第一个张量属性的数据类型

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs) #创建了一个新的模型实例。
        if state_dict is None:
            return model
        #如果state_dict不是None，那么就使用init_preweight方法初始化模型的权重。
        model = cls.init_preweight(model, state_dict)

        return model

##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    """
    Implementation of cross entropy loss over similarity score matrix, used for calculating
        symmetric cross entropy loss <https://arxiv.org/abs/1908.06112>.
    """
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1) #对相似性矩阵应用对数softmax函数，得到对数概率。
        logpt = torch.diag(logpt) #获取对数概率的对角线元素。
        nce_loss = -logpt #计算负对数概率，得到NCE损失。
        sim_loss = nce_loss.mean() #计算NCE损失的平均值，得到相似性损失。
        return sim_loss

class MILNCELoss(nn.Module):
    """
    Implementation of MIL-NCE Loss <https://arxiv.org/abs/1912.06430>
    """
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix): #接受一个相似性矩阵作为参数。
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair))) #创建一个掩码矩阵，它的对角线元素为1，其他元素为0。
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12 #计算从文本矩阵和从视频矩阵。
        from_video_matrix = sim_matrix.transpose(1, 0)

        #将从视频矩阵和从文本矩阵沿最后一个维度拼接起来，形成新的相似性矩阵。
        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)#对新的相似性矩阵应用对数softmax函数，得到对数概率。
        #使用torch.cat函数将mm_mask和一个与mm_mask形状相同的全零张量沿最后一个维度拼接起来，形成新的掩码mm_mask_logpt。
        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        #将logpt中mm_mask_logpt为0的位置的值设置为一个非常小的负数（-1e12）。
        # 这是通过将mm_mask_logpt从logpt中减去并乘以-1e12来实现的。
        # 这样做的目的是为了在后续的计算中，使得这些位置的值在应用softmax函数后接近于0，从而在计算损失时忽略这些位置。
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12
        #计算掩码后的对数概率的对数和的负值。
        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)
        #：创建一个选择器，用于选择需要计算损失的对数概率。
        logpt_choice = torch.zeros_like(new_logpt) #首先创建了一个从0到self.batch_size-1的整数序列，
        #然后将其转移到sim_matrix所在的设备（CPU或GPU）。
        # 接着，每个元素都乘以self.n_pair，然后再加上self.n_pair除以2的商。结果保存在mark_ind中。
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1 #将logpt_choice张量中mark_ind指定的位置的元素设置为
        #选择需要计算损失的对数概率，计算其平均值作为损失。
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):#实现最大边际排序损失
    """
    Implementation of max margin ranking loss
    """
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):#接受边际、负权重、批量大小、对数和硬负率作为参数。
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        #"""
        #“易负率”（easy_negative_rate）：这通常指的是在训练过程中，模型容易正确分类的负样本的比例。这些样本对于模型来说是“容易”的，因为模型可以很容易地识别出它们。
        #“难负率”（hard_negative_rate）：相反，这通常指的是在训练过程中，模型难以正确分类的负样本的比例。这些样本对于模型来说是“困难”的，因为模型在尝试识别它们时可能会犯错误。
        #“负权重”（negative_weighting）：这通常指的是在计算损失函数时，对负样本的权重。通过调整负样本的权重，可以影响模型对正样本和负样本的关注度，从而影响模型的性能。
        #"""
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:#如果对数大于1且批量大小大于1，那么计算并设置mm_mask属性。
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            #创建一个形状为self.batch_size x self.batch_size的单位矩阵，然后乘以1 - alpha，再加上alpha。结果保存在mm_mask中
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            #使用克罗内克积将mm_mask和一个形状为n_pair x n_pair的全1矩阵相乘。结果保存在mm_mask中。
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            #：将mm_mask转换为张量，然后乘以batch_size * (1 - easy_negative_rate)。
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float() #将mm_mask转换为浮点数，然后保存为类的一个属性

    def forward(self, x):
        d = torch.diag(x)#获取输入x的对角线元素。
        #计算最大边际。这里使用了ReLU激活函数来确保结果非负。
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        #如果启用了负权重，并且对数大于1且批量大小大于1，那么更新最大边际。
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean() #返回最大边际的平均值。
