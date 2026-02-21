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

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn
from modules.module_bert import BertModel, BertConfig
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_decoder import DecoderModel, DecoderConfig

from functools import partial

logger = logging.getLogger(__name__)
import copy

class CaptionGeneratorPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, decoder_config, *inputs, **kwargs):#接受BERT配置、视觉模型配置、解码器配置以及其他参数
        # utilize bert config as base config
        super(CaptionGeneratorPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.decoder_config = decoder_config


        self.visual = None
        self.decoder = None

        self.lp = None

    @classmethod
    #用于从预训练的模型中创建一个新的CaptionGeneratorPreTrainedModel实例。
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        #获取BERT配置、视觉模型配置和解码器配置。
        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        #创建一个新的CaptionGeneratorPreTrainedModel实例。
        model = cls(bert_config, visual_config, decoder_config,*inputs, **kwargs)
        # assert model.bert is not None
        assert model.visual is not None #检查模型是否有视觉属性。
        if state_dict is not None: #如果提供了状态字典，那么使用它来初始化模型的权重
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

class NormalizeVideo(nn.Module):
    #对视频进行归一化处理
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        #创建一个LayerNorm对象，用于对视频进行二维归一化。LayerNorm是一种常用的归一化技术，它可以使神经网络的训练更加稳定。
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float() #将视频转换为张量，并确保其数据类型为浮点数。
        #改变视频张量的形状。-1表示该维度的大小会自动计算，以保持张量中元素的总数不变。
        video = video.view(-1, video.shape[-2], video.shape[-1])
        #对视频进行二维归一化。
        video = self.visual_norm2d(video)
        return video
class NormalizeVideo_512(nn.Module):
    #对视频进行归一化处理
    def __init__(self, task_config):
        super(NormalizeVideo_512, self).__init__()
        #创建一个LayerNorm对象，用于对视频进行二维归一化。LayerNorm是一种常用的归一化技术，它可以使神经网络的训练更加稳定。
        self.visual_norm2d = LayerNorm(512)

    def forward(self, video):
        video = torch.as_tensor(video).float() #将视频转换为张量，并确保其数据类型为浮点数。
        #改变视频张量的形状。-1表示该维度的大小会自动计算，以保持张量中元素的总数不变。
        video = video.view(-1, video.shape[-2], video.shape[-1])
        #对视频进行二维归一化。
        video = self.visual_norm2d(video)
        return video

def show_log(task_config, info): #用于显示日志信息
    #如果task_config为None或者task_config.local_rank等于0，那么就会打印警告信息。
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None): 
    #用于更新属性
    #先检查source_config是否有source_attr_name属性，
    if hasattr(source_config, source_attr_name):
        #如果有，并且该属性的值不等于默认值，
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            #那么就将该属性的值设置到target_config的target_attr_name属性上，并显示日志信息。
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    #，返回更新后的target_config。
    return target_config

def check_attr(target_name, task_config):
    #target_name想要检查的属性的名称  task_config是包含属性的对象 #检查task_config对象是否有一个特定的属性，并返回该属性的值
    #hasattr(task_config, target_name)：检查task_config对象是否有一个名为target_name的属性。
    #task_config.__dict__[target_name]：如果task_config对象有这个属性，这会获取该属性的值
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CaptionGenerator(CaptionGeneratorPreTrainedModel):
    #用于生成视频字幕。
    def __init__(self, bert_config, visual_config, decoder_config,task_config):
        super(CaptionGenerator, self).__init__(bert_config, visual_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        #这些断言语句用于确保任务配置中的最大单词数和最大帧数不超过BERT配置和视觉模型配置中的最大位置嵌入数。
        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        #创建BERT模型、视觉模型和解码器模型，并获取它们的词嵌入权重和位置嵌入权重。

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        bert = BertModel(bert_config)
        bert_word_embeddings_weight = bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config,train = self.training)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

        # Decoder ===>
        decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                   self.task_config, "decoder_num_hidden_layers")
        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
        # <=== End of Decoder

        self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.normalize_video = NormalizeVideo(task_config) #对视频进行归一化处理。
        self.normalize_video_512 = NormalizeVideo_512(task_config)
        self.apply(self.init_weights) #应用权重初始化函数

    def forward(self, video, video_mask=None,
                input_caption_ids=None,
                decoder_mask=None,schedule_sampler=None, **kwargs):
        #video_mask = video_mask.view(-1, video_mask.shape[-1])

        if kwargs['video_swin']!=None:
            video = self.normalize_video_512(video)
            video_swin = self.normalize_video(kwargs['video_swin'])
            kwargs['video_swin']=video_swin
        else:
            video = self.normalize_video(video)

        #这段代码检查是否提供了input_caption_ids。如果提供了，那么就改变input_caption_ids和decoder_mask的形状
        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])
        if self.training:
            if not self.visual_config.prototype:
                visual_output, = self.get_visual_output(video, video_mask, shaped=True,**kwargs) # 获取视觉输出。
            else:
                visual_output, fea_loss, cst_loss, dis_loss = self.get_visual_output(video, video_mask, shaped=True,**kwargs) # 获取视觉输出。
        if not self.training:
            if not self.visual_config.prototype:
                visual_output, = self.get_visual_output(video, video_mask, shaped=True,**kwargs) # 获取视觉输出。
            else:
                visual_output, fea_loss  = self.get_visual_output(video, video_mask, shaped=True,**kwargs) # 获取视觉输出。

        video_mask = torch.ones(visual_output.size(0), visual_output.size(1)).to(video.device)
        if self.training:
            loss = 0.

            if (input_caption_ids is not None):
                #使用_get_decoder_score方法计算解码器的分数
                if self.decoder_config.cluster:
                    decoder_scores, res_tuples, sequence_output , cls_scores_entity, cls_scores_action,sequence_output_entity,sequence_output_action= self._get_decoder_score(visual_output, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    sequence_output_mask = torch.from_numpy(np.zeros((sequence_output.shape[-3], sequence_output.shape[-2]), dtype=np.long)).cuda()
                else:
                    decoder_scores, res_tuples, sequence_output= self._get_decoder_score(visual_output, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    sequence_output_mask = torch.from_numpy(np.zeros((sequence_output.shape[-3], sequence_output.shape[-2]), dtype=np.long)).cuda()

                if self.decoder_config.cluster:
                    if self.visual_config.prototype:
                        return decoder_scores, sequence_output, cls_scores_entity, cls_scores_action,sequence_output_entity,sequence_output_action,fea_loss, cst_loss, dis_loss
                    else:
                        return decoder_scores, sequence_output, cls_scores_entity, cls_scores_action,sequence_output_entity,sequence_output_action
                else:
                    if self.visual_config.prototype:
                        return decoder_scores, sequence_output,fea_loss, cst_loss, dis_loss
                    else:
                        return decoder_scores, sequence_output


        else:
            return None

    def get_visual_output(self, video, video_mask, shaped=False,**kwargs): #获取视觉模型的输出
        if shaped is False: #检查shaped参数是否为False。

            video_mask = video_mask.view(-1, video_mask.shape[-1])
            #video = self.normalize_video(video) #对视频进行归一化处理。
            video_swin = kwargs.get('video_swin')
            if video_swin is not None:
                video = self.normalize_video_512(video)
                video_swin = self.normalize_video(kwargs['video_swin'])
                kwargs['video_swin'] = video_swin
            else:
                video = self.normalize_video(video)
        #使用视觉模型处理视频，并获取所有编码层的输出。
        if not self.visual_config.prototype:
            visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True,**kwargs)
            visual_output = visual_layers[-1]  # 获取最后一层的输出作为视觉输出。
            return visual_output,

        else:
            if self.training:

                visual_layers, _, fea_loss, cst_loss, dis_loss = self.visual(video, video_mask, output_all_encoded_layers=True,**kwargs)
                visual_output = visual_layers[-1] #获取最后一层的输出作为视觉输出。
                return visual_output,fea_loss, cst_loss, dis_loss
            else:
                visual_layers, _, fea_loss = self.visual(video, video_mask, output_all_encoded_layers=True,**kwargs)
                visual_output = visual_layers[-1] #获取最后一层的输出作为视觉输出。
                return visual_output, fea_loss

    def _get_decoder_score(self, visual_output, video_mask, input_caption_ids, decoder_mask, shaped=False): #获取解码器分数
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        if self.training:
            if  self.decoder_config.cluster:

                decoder_scores, sequence_output, cls_scores_entity, cls_scores_action,sequence_output_entity,sequence_output_action= self.decoder(input_caption_ids, encoder_outs=visual_output,
                                                               answer_mask=decoder_mask, encoder_mask=video_mask)
                sequence_output_mask = torch.from_numpy(
                    np.zeros((sequence_output.shape[-3], sequence_output.shape[-2]), dtype=np.long)).cuda()

                return decoder_scores, res_tuples, sequence_output, cls_scores_entity, cls_scores_action,sequence_output_entity,sequence_output_action
            else:

                decoder_scores, sequence_output = self.decoder(input_caption_ids, encoder_outs=visual_output,
                                                               answer_mask=decoder_mask, encoder_mask=video_mask)
                sequence_output_mask = torch.from_numpy(
                    np.zeros((sequence_output.shape[-3], sequence_output.shape[-2]), dtype=np.long)).cuda()
                return decoder_scores, res_tuples, sequence_output
        else:
            decoder_scores, sequence_output = self.decoder(input_caption_ids, encoder_outs=visual_output,
                                                           answer_mask=decoder_mask, encoder_mask=video_mask)
            sequence_output_mask = torch.from_numpy(
                np.zeros((sequence_output.shape[-3], sequence_output.shape[-2]), dtype=np.long)).cuda()
            return decoder_scores, res_tuples, sequence_output

    def decoder_caption(self, visual_output, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores, _, _, = self._get_decoder_score(visual_output,
                                                    video_mask,
                                                    input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores  # 返回解码器的分数

        _, decoder_scores_result = torch.max(decoder_scores, -1)  # ，：否则，获取解码器分数的最大值。
        return decoder_scores_result
