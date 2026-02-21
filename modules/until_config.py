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
import logging
import tarfile
import tempfile
import shutil
import torch
from .file_utils import cached_path

logger = logging.getLogger(__name__)

class PretrainedConfig(object):

    pretrained_model_archive_map = {} #存储预训练模型的名称和对应的文件路径。
    #分别用于存储配置文件和权重文件的名称。
    config_name = ""
    weights_name = ""

    @classmethod
    def get_config(cls, pretrained_model_name, cache_dir, type_vocab_size, state_dict, task_config=None):
        archive_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_model_name)
        #用于获取预训练模型的配置。它接受五个参数：预训练模型的名称，缓存目录的路径，类型词汇的大小，状态字典，以及一个可选的任务配置。
        if os.path.exists(archive_file) is False:#检查archive_file（预训练模型的文件）是否存在。如果不存在
            if pretrained_model_name in cls.pretrained_model_archive_map:
                #检查预训练模型的名称是否在pretrained_model_archive_map（预训练模型的名称和对应的文件路径的映射）中。
                # 如果在，它会将archive_file设置为映射中对应的文件路径
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name]
            else:
                #如果预训练模型的名称不在映射中，它会假设pretrained_model_name是一个路径或URL，
                # 并将archive_file设置为pretrained_model_name。
                archive_file = pretrained_model_name

        # redirect to the cache, if necessary
        try: #它会尝试从缓存中加载模型。如果模型不在缓存中，cached_path函数会下载模型并将其保存到缓存中。
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            #文件未找到错误），它会记录一条错误信息，指出预训练模型的名称未在模型名称列表中找到，且假设的路径或URL没有关联的文件#
            if task_config is None or task_config.local_rank == 0:
                logger.error(
                    "Model name '{}' was not found in model name list. "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name,
                        archive_file))
            return None
        #加载预训练模型的存档文件，并在必要时将其解压缩到临时目录。具体步骤如下：
        #首先，它检查resolved_archive_file（从缓存中加载的模型文件）是否与archive_file（预训练模型的文件）相同。
        #如果相同，它会记录一条信息，指出正在加载archive_file。
        if resolved_archive_file == archive_file:
            if task_config is None or task_config.local_rank == 0:
                logger.info("loading archive file {}".format(archive_file))
        else:        # 如果不同，它会记录一条信息，指出正在从缓存中的resolved_archive_file加载archive_file。
            if task_config is None or task_config.local_rank == 0:
                logger.info("loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file): 
            #resolved_archive_file是否是一个目录。
            # 如果是，它将serialization_dir（序列化目录）设置为resolved_archive_file。
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir #如果模型是一个压缩文件，它会将其解压到一个临时目录中。
            #并将tempdir（临时目录）设置为新创建的目录。然后，它会记录一条信息，指出正在将resolved_archive_file解压缩到临时目录。
            tempdir = tempfile.mkdtemp()
            if task_config is None or task_config.local_rank == 0:
                logger.info("extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir))
            #使用tarfile.open函数打开resolved_archive_file
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)#使用extractall方法将其解压缩到临时目录
            serialization_dir = tempdir #将serialization_dir设置为临时目录。
        # Load config #会从解压后的目录中加载配置文件，并将type_vocab_size设置为配置的属性。
        config_file = os.path.join(serialization_dir, cls.config_name) #通过将serialization_dir（序列化目录）和cls.config_name（配置文件的名称）连接起来，得到配置文件的路径。
        config = cls.from_json_file(config_file) #从配置文件中加载配置，
        config.type_vocab_size = type_vocab_size # 将type_vocab_size（类型词汇的大小）设置为配置的属性。
        if task_config is None or task_config.local_rank == 0: 
            logger.info("Model config {}".format(config))

        if state_dict is None:#state_dict（状态字典）为None 尝试从序列化目录中加载权重文件
            weights_path = os.path.join(serialization_dir, cls.weights_name) #serialization_dir和cls.weights_name（权重文件的名称），得到权重文件的路径
            if os.path.exists(weights_path): #如果权重文件存在，它使用torch.load函数加载权重文件
                state_dict = torch.load(weights_path, map_location='cpu')
            else:
                if task_config is None or task_config.local_rank == 0: #记录一条信息，指出权重文件不存在。
                    logger.info("Weight doesn't exsits. {}".format(weights_path))

        if tempdir: #如果使用了临时目录，它会使用shutil.rmtree函数清理临时目录。
            # Clean up temp dir
            shutil.rmtree(tempdir)

        return config, state_dict

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1) #创建一个新的BertConfig对象config，并将vocab_size_or_config_json_file参数设置为-1
        for key, value in json_object.items(): #遍历json_object中的每一对键值对，将每个键值对添加到config对象的__dict__属性中
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader: #打开并读取json_file，将文件内容保存到变量text中
            text = reader.read()
        return cls.from_dict(json.loads(text)) #使用json.loads函数将text转换为Python字典，并将该字典传递给from_dict方法 创建并返回一个BertConfig对象。

    def __repr__(self):#当你打印一个对象或使用str()函数将其转换为字符串时，Python会调用这个方法
        return str(self.to_json_string()) #调用to_json_string方法将对象实例转换为JSON字符串，并返回这个字符串。

    def to_dict(self): #将对象实例序列化为Python字典。
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__) #copy.deepcopy函数创建self.__dict__（对象实例的属性字典）的一个深拷贝，然后返回这个拷贝
        #这样返回的字典不会影响原始对象实例的属性。
        return output

    def to_json_string(self):#将对象实例序列化为JSON字符串。
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n" #to_dict方法将对象实例转换为Python字典
        #json.dumps函数将这个字典转换为JSON字符串
        #json.dumps函数的indent参数设置为2，表示在输出的JSON字符串中，每一级缩进使用两个空格
        #sort_keys参数设置为True，表示在输出的JSON字符串中，字典的键按照字母顺序排序。
        #最后，它在JSON字符串的末尾添加一个换行符，