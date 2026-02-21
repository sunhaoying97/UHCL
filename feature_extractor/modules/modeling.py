from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32" 
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)#获取预训练 CLIP 模型的配置，并将其添加到状态字典中。
        for key, val in clip_state_dict.items(): #遍历 clip_state_dict 中的每个键值对
            new_key = "clip." + key 
            if new_key not in state_dict: #如果状态字典中不存在以 “clip.” 开头的键，就将其添加到状态字典中。
                state_dict[new_key] = val.clone()
        #获取交叉模型的配置 cross_config
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        #创建一个 CLIP4ClipPreTrainedModel 实例
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)
        """
        linear_patch 是一个字符串参数，它可以是 ‘2d’ 或 ‘3d’。这个参数用于指定模型应该使用哪种类型的卷积层。
        如果 linear_patch 的值为 ‘3d’，那么模型会使用一个三维卷积层 conv2。这个卷积层在模型的 forward 方法中被用来处理输入张量 x。
        具体来说，它会将 x 重塑为一个五维张量，然后通过 conv2 层进行卷积操作，最后再将结果重塑回四维。
        如果 linear_patch 的值为 ‘2d’，那么模型会使用一个二维卷积层 conv1 来处理输入张量 x。
        总的来说，linear_patch 参数允许你选择模型应该在二维空间还是三维空间中进行卷积操作，这可能会影响模型的性能和结果。
        """
        ## ===> Initialization trick [HARD CODE]
        #在状态字典中添加一个新的键值对，其键是 “clip.visual.conv2.weight”，值是 cp_weight。
        #这个新的键值对可以用于初始化模型的 conv2 层的权重。这是一种常见的权重初始化技巧，可以帮助提高模型的性能。
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1: #如果找到了以 “visual.conv2.weight” 结尾的键，就将 contain_conv2 设置为 True
                    contain_conv2 = True
                    break
            #如果 contain_conv2 为 False，并且模型的 clip.visual 有 conv2 属性
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone() #复制状态字典中 “clip.visual.conv1.weight” 的权重 ，并保存为 cp_weight
                #获取 conv2 层的核大小和权重大小，并将权重大小转换为列表。
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)
                #复制 conv2 层的权重大小，并分别保存为 left_conv2_size 和 right_conv2_size
                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                #计算 left_conv2_size 和 right_conv2_size 的第三个元素。
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]
            
                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device) #创建一个全零张量
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                #将 left_zeros、cp_weight 和 right_zeros 添加到列表中
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2) #将列表中的所有张量沿第三个维度拼接起来，并保存为 cp_weight
                #将 cp_weight 添加到状态字典中，键为 “clip.visual.conv2.weight”。
                state_dict["clip.visual.conv2.weight"] = cp_weight
        #在状态字典中添加一些新的键值对，这些键值对用于初始化模型的 cross 属性。权重初始化技巧，可以帮助提高模型的性能。
        if model.sim_header == 'tightTransf':
            #遍历状态字典的所有键，如果找到了以 “cross.transformer” 结尾的键，就将 contain_cross 设置为 True，并跳出循环。
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            #它遍历 clip_state_dict 中的每个键值对，
            #如果键等于 “positional_embedding”，就将其值复制并添加到状态字典中，键为 “cross.embeddings.position_embeddings.weight”
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    #如果键以 “transformer.resblocks” 开头，就获取键中的第三个元素，并将其转换为整数，保存为 num_layer。
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        #如果 num_layer 小于 task_config.cross_num_hidden_layers，就将其值复制并添加到状态字典中，键为 “cross.” 加上原来的键。
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue
        #在状态字典中添加一些新的键值对，这些键值对用于初始化模型的特定属性
        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                #如果找到以 “frame_position_embeddings” 结尾的键， contain_frame_position 设置为 True，并跳出循环。
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                #遍历 clip_state_dict 中的每个键值对，
                # 如果键等于 “positional_embedding”，就将其值复制并添加到状态字典中，键为 “frame_position_embeddings.weigh
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    #如果模型的 sim_header 属性为 “seqTransf”，并且键以 “transformer.resblocks” 开头，
                    # 就获取键中的第三个元素，并将其转换为整数，保存为 num_layer
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        #如果 num_layer 小于 task_config.cross_num_hidden_layers，
                        # 就将其值复制并添加到状态字典中，键为 “transformerClip.” 加上原来的键。
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
                            #这个新的键值对可以用于初始化模型的 transformerClip 层的权重
        ## <=== End of initialization trick
        #使用预训练权重初始化模型
        if state_dict is not None:
            #调用类方法 init_preweight 来初始化模型
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info) #使用 logger.warning 来打印 info

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        #如果 source_config 有 source_attr_name 属性，并且其值不等于 default_value
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            #将 source_config 的 source_attr_name 属性的值设置为 target_config 的 target_attr_name 属性的值，并打印一条日志
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    #检查 task_config 是否有 target_name 属性并且该属性的值是否为真。如果条件满足，它返回 True，否则返回 False。
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        #可能表示模型的训练阶段。
        self._stage_one = True
        self._stage_two = False
        #显示当前的训练阶段
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))
        #初始化 loose_type 为 False，并检查是否在第一阶段并且 task_config 有 ‘loose_type’ 属性。
        # 如果条件满足，就将 loose_type 设置为 True，并打印一条日志
        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        #从预训练的CLIP模型的状态字典 clip_state_dict 中提取一些关键的配置信息，用于初始化 CLIP4Clip 模型。
        vit = "visual.proj" in clip_state_dict #检查 clip_state_dict 是否包含键 “visual.proj”
        assert vit
        if vit:#包含键“visual.proj”，则提取关键的配置信息，如 vision_width、vision_layers、vision_patch_size、grid_size 和 image_resolution。
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:#不包含，则会提取一些不同的配置信息，
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts) #
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
        
        embed_dim = clip_state_dict["text_projection"].shape[1] ##文本投影的维度 
        context_length = clip_state_dict["positional_embedding"].shape[0] #上下文长度
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0] #词汇表的大小
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64 #Transformer的头的数量
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))#Transformer的层数

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        #cut_top_layer 是一个变量，通常在迁移学习中使用，用于指定要从预训练模型中移除的顶层数量。
        # 在迁移学习中，我们通常会使用一个预训练的模型（如VGG16、ResNet等）作为基础模型，
        # 然后根据特定任务的需要，对这个模型进行一些修改。这些修改可能包括添加新的层，或者移除一些现有的层。
        # 在这个上下文中，cut_top_layer 的值为 0，意味着不从预训练模型中移除任何层。然而
        # 如果你设置 cut_top_layer 为一个大于 0 的值，那么就会从预训练模型的顶部移除指定数量的层。让模型能够更好地适应新的任务。
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float() #创建一个 CLIP 实例，并将其赋给 self.clip #调用 float 方法将模型的权重转换为浮点数 #避免溢出或下溢。
        #从 clip_state_dict 中删除了 “input_resolution”、“context_length” 和 “vocab_size” 这三个键（如果它们存在的话）
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]
        #调用 convert_weights 函数对 self.clip 进行权重转换。
        convert_weights(self.clip)
        # <=== End of CLIP Encoders


        #用于指定模型应该使用哪种类型的相似性计算方法。
        # 在 `CLIP4Clip` 模型中，`sim_header` 可以设置为 'meanP'、'seqLSTM'、'seqTransf' 或 'tightTransf'。
        #当 `sim_header` 设置为 'meanP' 时，模型将使用一种被称为 "parameter-free type" 的相似性计算方法。这种方法不需要额外的参数，因此被称为 "parameter-free"。具体的实现细节可能会根据模型的其他部分而变化，
        # 但通常，这种方法会计算视频帧和文本之间的平均相似性，然后使用这个平均值来进行视频检索。
        self.sim_header = 'meanP' 
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:#布尔变量，#通常，当 self.loose_type 为 True 时，模型可能会在处理某些任务时采取一种更宽松的策略
            # Cross Encoder ===>
            #更新 cross_config 的 num_hidden_layers 属性
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            #创建一个 CrossModel 实例和一个线性层。
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            #创建一个嵌入层
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            #创建一个 TransformerClip 实例
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            #创建一个 LSTM 层。
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn() #设置损失函数

        self.apply(self.init_weights) #初始化模型的权重

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        #将这些参数重塑为适合模型处理的形状
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float() #视频输入转换为浮点数，
        #重塑形状
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        #获取序列输出和视觉输出
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:  #处于训练模式，
            loss = 0.
            #计算相似性矩阵，
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                    shaped=True, loose_type=self.loose_type)
            #并使用交叉熵损失函数计算损失
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss

            return loss
        else:
            return None
    #获取模型的序列输出
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)#获取 input_ids 的第一个维度的大小，并将其保存为 bs_pair。
        #对 input_ids 进行编码，并将结果转换为浮点数，保存为 sequence_hidden
        sequence_hidden = self.clip.encode_text(input_ids).float() 
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden
    #获取模型的视觉输出。
    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            #将 video_mask 和 video 重塑为适合模型处理的形状，并计算视频帧数。
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        #对 video 进行编码
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts #计算视频帧数
        #获取模型的序列输出和视觉输出
        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output
    #获取cross模型的输出
    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        #创建了两个与 attention_mask 和 video_mask 形状相同的张量 text_type_ 和 video_type_，
        # 分别填充零和一，然后将它们沿第一维度连接起来，得到 concat_type。
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)
        #为了在后续的处理中，能够根据 concat_type 中的值来区分输入序列中的文本部分和视频部分。
        #
        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1] #取 cross_layers 的最后一层作为 cross_output。

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1) #转换为浮点数并增加一个维度。
        attention_mask_un[:, 0, :] = 0. #第一列设置为0，这可能是为了忽略某些特定的输入标记。
        sequence_output = sequence_output * attention_mask_un#只有那些被掩码标记为关注的标记才会被保留。
        #平均池化后的文本输出
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un #只有那些被掩码标记为关注的视频帧才会被保留。
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)#求和
        video_mask_un_sum[video_mask_un_sum == 0.] = 1. #如果和为0，则将其设置为1，这可能是为了避免除以0的情况
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum #平均池化后的视频输出
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask) #文本输出 text_out
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask) #视频输出 video_out

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):

        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()# 转换为连续的内存布局。

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training: 
            #allgather 函数用于将来自不同 GPU 的 visual_output 聚合到一起。
            #每个 GPU 计算一部分数据，然后需要将它们合并以更新模型参数
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            #同步操作，它会阻塞代码执行，直到所有 GPU 都完成了前面的聚合操作。
            # 这确保了在继续之前，所有 GPU 都已经完成了数据的交换
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True) #归一化
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask) #平均池化
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)#归一化

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t()) #点积，得到检索 logits。
        return retrieve_logits
    #对每个小批次的数据进行处理，计算序列和视觉输出之间的交叉相似性
    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = [] #用于存储检索 logits

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)  #是一个列表，其元素都是 step_size
        release_size = b_text - sum(split_size) #b_text 不能被 step_size 整除，那么 release_size 就是余数。
        if release_size > 0: #如果 release_size 大于0，那么将 release_size 添加到 split_size 列表的末尾。
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)
        #根据步长将 sequence_output 和 attention_mask 分割成多个部分。
        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            #获取当前行 sequence_output_row 和 attention_mask_row。
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0) #sequence_output_row 的大小
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)
             
            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual) #使用 similarity_dense 函数处理池化输出，得到检索 logits

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    #计算序列和视觉输出之间的相似性
    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            #将 attention_mask 和 video_mask 的形状重塑为 (-1, shape[-1])，其中 shape[-1] 是各自数组的最后一个维度的大小。
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type: #True 则 调用 _loose_similarity 函数来计算相似性。
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:#否则，调用 _cross_similarity 函数来计算相似性
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
