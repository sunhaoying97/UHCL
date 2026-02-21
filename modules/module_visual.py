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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import pdb
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'visual_config.json'
WEIGHTS_NAME = 'visual_pytorch_model.bin'


class VisualConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `VisualModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file=4096,
                 hidden_size=768,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512, #
                 initializer_range=0.02):
        """Constructs VisualConfig.

        Args:
            vocab_size_or_config_json_file: Size of the encoder layers and the pooler layer.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str): 
            #检查vocab_size_or_config_json_file的类型。如果它是一个字符串，那么它被认为是配置文件的路径，函数会从该路径加载配置
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):#如果它是一个整数，那么它被认为是词汇表的大小，函数会将其以及其他参数保存为对象的属性。
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class VisualEmbeddings(nn.Module): #构建词嵌入、位置嵌入和标记类型嵌入
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(VisualEmbeddings, self).__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        #对隐藏层大小config.hidden_size进行层归一化
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embeddings):
        ## 获取输入嵌入的序列长度

        seq_length = input_embeddings.size(1)
        #  创建位置id，范围从0到seq_length
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
        # 扩展位置id以匹配输入嵌入的大小
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)
        #通过词嵌入层传递 输入嵌入
        words_embeddings = self.word_embeddings(input_embeddings)
        # words_embeddings = self.transform_act_fn(words_embeddings)
        #通过位置嵌入层传递位置id
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        #对嵌入进行层归一化
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VisualSelfAttention(nn.Module):
    def __init__(self, config):
        super(VisualSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0: ## 检查隐藏层大小是否可以被注意力头数整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads ## 注意力头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) ## 注意力头大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size #所有头的总大小

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x): ## 调整张量的形状以适应注意力分数的计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        ## 对隐藏状态进行线性变换以得到查询、键和值
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        ### 转置查询、键和值以适应注意力分数的计算
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # # 计算查询和键的点积以得到原始的注意力分数
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in VisualModel forward() function)
        # # 应用注意力掩码
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) ## 将注意力分数归一化为概率

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        ## 计算上下文
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class VisualSelfOutput(nn.Module): #自注意力输出
    def __init__(self, config):
        super(VisualSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualAttention(nn.Module): #自注意力机制
    def __init__(self, config):
        super(VisualAttention, self).__init__()
        self.self = VisualSelfAttention(config)
        self.output = VisualSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class VisualIntermediate(nn.Module): #自注意力机制的中间层
    def __init__(self, config):
        super(VisualIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        #中间激活函数：如果config.hidden_act是字符串，则从ACT2FN字典中获取对应的激活函数；
        # # 否则，直接使用config.hidden_act作为激活函数
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class VisualOutput(nn.Module):
    def __init__(self, config):
        super(VisualOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualLayer(nn.Module): # #实现自注意力机制的整个层
    def __init__(self, config):
        super(VisualLayer, self).__init__()
        self.attention = VisualAttention(config)
        self.intermediate = VisualIntermediate(config)
        self.output = VisualOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super(VisualEncoder, self).__init__()
        layer = VisualLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = [] ## 创建一个空列表来存储所有编码器层的输出
        for layer_module in self.layer: #
            hidden_states = layer_module(hidden_states, attention_mask)## 将隐藏状态和注意力掩码传递给当前层，并获取输出
            if output_all_encoded_layers:#将当前层的输出添加到all_encoder_layers列表中
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers: #如果output_all_encoded_layers为False，则只将最后一层的输出添加到all_encoder_layers列表中
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class VisualPooler(nn.Module): #自注意力机制的池化 
    def __init__(self, config):
        super(VisualPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()#双曲正切函数

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]## 我们通过简单地取第一个令牌对应的隐藏状态来"池化"模型
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VisualPredictionHeadTransform(nn.Module):# transformer预测头
    def __init__(self, config):
        super(VisualPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        ##如果config.hidden_act是字符串，则从ACT2FN字典中获取对应的激活函数；
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):#语言模型预测头
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualLMPredictionHead, self).__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.weight = visual_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(visual_model_embedding_weights.size(1)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(self.weight) + self.bias
        return hidden_states


class VisualOnlyMLMHead(nn.Module):#实现仅视觉的Masked Language Model（MLM）预测头
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualOnlyMLMHead, self).__init__()
        self.predictions = VisualLMPredictionHead(config, visual_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores## 返回预测分数

#“序列关系分数"可能是指在自然语言处理或机器学习任务中，模型试图预测两个输入序列之间的某种关系的得分。
# 例如，在下一句预测（Next Sentence Prediction，NSP）任务中，
# 模型需要预测第二个句子是否在原文中紧接着第一个句子出现。
# 模型会为"是"和"否"这两种可能性分别赋予一个得分，这些得分就可以被看作是"序列关系分数”。

class VisualOnlyNSPHead(nn.Module):#仅视觉的Next Sentence Prediction（NSP）预测头
    def __init__(self, config):
        super(VisualOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score # # 返回序列关系分数


class VisualPreTrainingHeads(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualPreTrainingHeads, self).__init__()
        self.predictions = VisualLMPredictionHead(config, visual_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output) #获得预测分数
        seq_relationship_score = self.seq_relationship(pooled_output) #获得序列关系分数
        return prediction_scores, seq_relationship_score


def mean_distance(a, b, weight=None, training=True):
    dis = ((a - b) ** 2).sum(-1)

    if weight is not None:
        dis *= weight

    if not training:
        return dis
    else:
        return dis.mean().unsqueeze(0)


def distance(a, b):
    return ((a - b) ** 2).sum(-1)
class VisualModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
video
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config,train):
        super(VisualModel, self).__init__(config)
        self.embeddings = VisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)
        self.Mheads = nn.Linear(config.max_position_embeddings, config.num_prototype, bias=False)
        self.config = config


        # self.spatial_prototype_weight = nn.Sequential(
        #     nn.Linear(config.max_position_embeddings, config.max_position_embeddings), nn.ReLU(inplace=True),
        #     nn.Linear(config.max_position_embeddings, config.max_position_embeddings), nn.ReLU(inplace=True))
        #
        # self.temporal_prototype_weight = nn.Sequential(
        #     nn.Linear(config.max_position_embeddings, config.max_position_embeddings), nn.ReLU(inplace=True),
        #     nn.Linear(config.max_position_embeddings, 1), nn.ReLU(inplace=True))


        self.proto_size = config.num_prototype
        self.training = train
        if self.config.video_feat_type == 'Swin' or self.config.video_feat_type == 'Swin+Clip':
            self.reshape_Swin = nn.Sequential(
                nn.Linear(1024, 512, bias=True),
                LayerNorm((512,), eps=1e-12),
                nn.Dropout(p=0.5, inplace=False))


    def forward(self, video, attention_mask=None, output_all_encoded_layers=True,**kwargs):


        if self.config.video_feat_type == 'Swin':
            video = self.reshape_Swin(video)
        elif self.config.video_feat_type == 'Swin+Clip':
            video_swin = self.reshape_Swin(kwargs['video_swin'])
            video = torch.cat((video,video_swin),dim=1)
        #if attention_mask is None:## 如果没有提供注意力掩码，则创建一个全为1的掩码
        attention_mask = torch.ones(video.size(0), video.size(1)).to(video.device)

        if self.config.prototype:
            multi_heads_weights = self.Mheads(video)
            # softmax on weights
            multi_heads_weights = F.softmax(multi_heads_weights, dim=1)
            protos = multi_heads_weights * video
            if self.training:
                updated_video, fea_loss, cst_loss, dis_loss = self.query_loss(video, protos)
            else:
                updated_video, fea_loss, video = self.query_loss(video, protos)

            # spatial_prototype_weight = self.spatial_prototype_weight(video)
            # updated_video = video * spatial_prototype_weight
            #
            # temporal_prototype_weight = self.temporal_prototype_weight(updated_video)
            # video = video + video * temporal_prototype_weight
            # sparse_temporal_loss = self.get_loss_sparsity(temporal_prototype_weight)
            # sparse_spatial_loss = self.get_loss_sparsity(spatial_prototype_weight)
            # sparse_loss = sparse_temporal_loss + sparse_spatial_loss
            # fea_loss = sparse_loss

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(video)


        ## 通过编码器传递嵌入输出和扩展的注意力掩码
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]  #sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) ## 通过池化层传递序列输出以获得池化输出
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1] ## # 如果不输出所有编码层，则只获取最后一层的编码层
        #return encoded_layers, pooled_output

        if not self.config.prototype:
            return encoded_layers, pooled_output
        else:
            if self.training:
                return encoded_layers, pooled_output, fea_loss, cst_loss, dis_loss #fea_loss, cst_loss, dis_loss
            else:
                return encoded_layers, pooled_output, fea_loss # fea_loss

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss
    def query_loss(self, query, keys):
        batch_size, n, dims = query.size()  # b X n X d, n=w*h
        if self.training:
            # Distinction constrain

            keys_ = F.normalize(keys, dim=-1)

            dis = 1 - distance(keys_.unsqueeze(1), keys_.unsqueeze(2))

            mask = dis > 0
            dis *= mask.float()
            dis = torch.triu(dis, diagonal=1)
            dis_loss = dis.sum(1).sum(1) * 2 / (self.proto_size * (self.proto_size - 1))
            dis_loss = dis_loss.mean()

            # maintain the consistance of same attribute vector
            cst_loss = mean_distance(keys_[1:], keys_[:-1])

            # Normal constrain
            loss_mse = torch.nn.MSELoss()

            keys = F.normalize(keys, dim=-1)
            _, softmax_score_proto = self.get_score(keys, query)

            new_query = softmax_score_proto.unsqueeze(-1) * keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)

            # maintain the distinction among attribute vectors
            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)

            # 1st closest memories
            pos = torch.gather(keys, 1, gathering_indices[:, :, :1].repeat((1, 1, dims)))

            fea_loss = loss_mse(query, pos)

            return new_query, fea_loss, cst_loss, dis_loss

        else:
            loss_mse = torch.nn.MSELoss(reduction='none')
            keys = F.normalize(keys, dim=-1)
            softmax_score_query, softmax_score_proto = self.get_score(keys, query)

            new_query = softmax_score_proto.unsqueeze(-1) * keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)

            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)

            # 1st closest memories
            pos = torch.gather(keys, 1, gathering_indices[:, :, :1].repeat((1, 1, dims)))

            fea_loss = loss_mse(query, pos)

            return new_query, fea_loss, query

    def get_score(self, pro, query):

        bs, n, d = query.size()  # n=w*h
        bs, m, d = pro.size()

        score = torch.bmm(query, pro.permute(0, 2, 1))  # b X h X w X m
        score = score.view(bs, n, m)  # b X n X m

        score_query = F.softmax(score, dim=1)
        score_proto = F.softmax(score, dim=2)

        return score_query, score_proto