from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
from nlgeval import NLGEval
import time
import argparse
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CaptionGenerator
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_feats import MSRVTT_Feats_DataLoader
from feature_extractor.util import get_logger
from tqdm import tqdm
from dataloaders.dataloader_msvd_feats import MSVD_Feats_DataLoader
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


torch.distributed.init_process_group(backend="nccl")

global logger
#ema
def get_args(description='CaptionGenerator'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--data_path', type=str, default='data/MSRVTT_data.json',
                        help='caption and transcription file path')
    parser.add_argument('--features_path', type=str, default='data/msrvtt_videos_feature.pickle',
                        help='feature path for CLIP features')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-model", type=str, required=True, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset `msrvtt` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")
    parser.add_argument('--d_model', type=int, default=512, help="dim of gcn model.")

    parser.add_argument('--patience', type=int, default=50, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--patience_metric', type=str, default="CIDEr", help="Metric which is used for early stopping.")
    parser.add_argument('--target_metric', type=str, default="CIDEr", help="Target metric which is used to select the best model.")


    ##
    parser.add_argument("--contrast", default=True, type=bool, help="Use contrast loss or not.")
    parser.add_argument("--cluster", default=True, type=bool, help="Use cluster loss or not.")
    parser.add_argument("--prototype", default=False, type=bool, help="Use prototype loss or not.")
    parser.add_argument("--video_feat_type", default='Clip', choices=['Swin', 'Swin+Clip','Clip'], type=str, help="Use 'Swin', 'Swin+Clip','Clip'.") # TODO: Swin+Clip

    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False #禁用CuDNN的基准测试模式，这将使得计算结果更加确定。
    torch.backends.cudnn.deterministic = True #启用CuDNN的确定性模式，这将使得计算结果更加确定。

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    # pip install scikit_learn -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
    if not os.path.exists(args.output_dir): #如果输出目录不存在
        os.makedirs(args.output_dir, exist_ok=True) #使用get_logger函数创建一个新的日志记录器，它将把日志信息写入到输出目录下的"log.txt"文件中。

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__): #对args中的所有字段按照字段名排序，然后遍历每一个字段。
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
            #记录一条信息，表示当前字段的名字和值。
    return args

def init_device(args, local_rank):
    #定义一个名为init_device的函数，它接受两个参数：args（模型和任务的配置参数）和local_rank（用于分布式训练的本地排名）
    global logger #：声明logger为全局变量，这样就可以在函数内部修改它。

    #创建一个torch.device对象，表示将要在其上执行Tensor操作的设备。
    # 如果CUDA可用，设备将设置为"CUDA"，否则将设置为"CPU"。local_rank参数指定了在多GPU设置中使用哪个设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    # n_gpu = torch.cuda.device_count()
    n_gpu = torch.distributed.get_world_size() #获取分布式训练的大小，即参与训练的进程总数。这个值也被视为可用的GPU数量。
    logger.info("device: {} n_gpu: {}".format(device, n_gpu)) #使用logger记录设备和GPU数量的信息。
    args.n_gpu = n_gpu #将args中的n_gpu字段设置为当前的GPU数量。

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:#如果批处理大小或验证批处理大小不能被GPU数量整除，那么…
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu)) #…抛出一个ValueError异常，说明批处理大小和GPU数量的参数无效。

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    # args（模型和任务的配置参数），
    # device（用于训练的设备，如CPU或GPU），n_gpu（用于训练的GPU数量），local_rank（用于分布式训练的本地排名
    """
        weight initialization
    """

    if args.init_model: #如果args中有预训练模型的路径，加载预训练模型的状态字典（state_dict），并将所有张量加载到CPU上。
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None #否则将model_state_dict设置为None。

    # Prepare model
    #设置缓存目录。如果args中有缓存目录的路径，就使用该路径，否则使用默认的缓存目录。
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    #从预训练模型创建一个新的模型实例。 #args.bert_model BERT模型的名称或路径
    #args.visual_model（视觉模型的名称或路径），args.decoder_model（解码器模型的名称或路径），
    # cache_dir（缓存目录的路径），state_dict（模型的状态字典），task_config（任务的配置参数）
    model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device) #将模型的所有参数和缓冲区移动到给定的设备上

    return model #返回初始化后的模型


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'): #
        model = model.module


    param_optimizer = list(model.named_parameters()) #使用 model.named_parameters 方法获取模型的所有参数，并将它们放入一个列表中
    #定义一个名为 no_decay 的列表，包含 ‘bias’、‘LayerNorm.bias’ 和 ‘LayerNorm.weight’。这些参数在优化过程中不会衰减。
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    #将模型参数分为四类不衰减的 BERT 参数、不衰减的非 BERT 参数、衰减的 BERT 参数和衰减的非 BERT 参数。
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    #根据这四类参数创建一个名为 optimizer_grouped_parameters 的列表。每个元素是一个字典，包含一类参数、它们的权重衰减系数和学习率。
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    #创建一个 BertAdam 优化器，传入 optimizer_grouped_parameters、学习率、预热比例、调度策略、优化步骤总数、权重衰减系数和最大梯度范数
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Feats_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        video_feat_type=args.video_feat_type
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,#(train_sampler is None),
        #sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler

def dataloader_msrvtt_val_test(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Feats_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        video_feat_type=args.video_feat_type
    )

    #test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        #sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_msvd_train(args, tokenizer, split_type="train",):
    msvd = MSVD_Feats_DataLoader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        feature_framerate=args.feature_framerate,
        video_feat_type=args.video_feat_type
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        #sampler=train_sampler,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,#(train_sampler is None),
        drop_last=True,)
    return dataloader_msvd, len(msvd)#, train_sampler

def dataloader_msvd_val_test(args, tokenizer, split_type="val",):
    msvd = MSVD_Feats_DataLoader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        video_feat_type=args.video_feat_type
    )

    #sampler = SequentialSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        #sampler=sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msvd, len(msvd)
def score(ref, hypo): #计算指标
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    #这个函数接受一个状态字典和一个目标类型，然后将状态字典中的所有张量转换为目标类型。
    # 如果状态字典是一个字典，那么它会递归地对每个值进行转换。
    # 如果状态字典是一个列表，那么它会对列表中的每个元素进行转换。如果状态字典是一个张量，那么它会直接进行转换。
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dicdec_outputt):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    #这个函数用于保存模型。它首先获取模型的状态字典，然后将其保存到指定的文件中。文件的路径由输出目录、类型名和周期数共同决定。
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "VIT-B-32.pt111.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    #它首先确定模型文件的路径，然后从文件中加载模型的状态字典。
    # 然后，它使用状态字典和其他参数创建一个新的模型实例，并将其移动到指定的设备上。
    # 如果模型文件不存在，那么它会返回None。
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "VIT-B-32.pt111.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores
import copy
def train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0,schedule_sampler=None):
    """
        For training model simultaneously.
    """
    global logger
    torch.cuda.empty_cache() #清空CUDA的缓存，以释放GPU内存。
    model.train() #将模型设置为训练模式



    log_step = args.n_display #
    start_time = time.time()
    total_loss = 0
    offset = []

    for step, batch in enumerate(train_dataloader): #对数据加载器中的每个批次进行遍历。
        # if n_gpu == 1:
        #     # multi-gpu does scattering it-self
        #     batch = tuple(t.to(device) for t in batch)
        ##将批次中的每个张量移动到指定的设备上。

        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        if args.video_feat_type=='Swin+Clip':
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            video_swin, video_mask_swin, masked_video_swin, video_labels_index_swin, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
            pairs_decoder_mask_pos, pairs_decoder_mask_neg, pairs_input_caption_ids_pos, pairs_output_caption_ids_pos, pairs_input_caption_ids_neg, pairs_output_caption_ids_neg, \
            cap_verbs_ids, cap_nouns_ids, cap_pos_verbs_ids, cap_pos_nouns_ids, cap_neg_verbs_ids, cap_neg_nouns_ids = batch
        else:
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
            pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
            cap_verbs_ids, cap_nouns_ids,cap_pos_verbs_ids, cap_pos_nouns_ids,cap_neg_verbs_ids, cap_neg_nouns_ids= batch


        if not args.contrast:

            decoder_scores, sequence_output = model(video, video_mask,
                                                    input_caption_ids=pairs_input_caption_ids,
                                                    decoder_mask=pairs_decoder_mask,
                                                    schedule_sampler=schedule_sampler,
                                                    video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)
            # decoder_scores.shape [256, 48, 30522]) [batchsize, 48, vocabsize])

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            loss = model.decoder_loss_fct(decoder_scores.view(-1, model.bert_config.vocab_size),
                                             pairs_output_caption_ids.view(-1))


        else:
            if args.cluster:
                if not args.prototype:


                    decoder_scores, sequence_output , decoder_scores_entity, decoder_scores_action,sequence_output_entity,sequence_output_action= model(video, video_mask,
                                                                                            input_caption_ids=pairs_input_caption_ids,
                                                                                            decoder_mask=pairs_decoder_mask,
                                                                                            schedule_sampler=schedule_sampler,
                                                                                            video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)

                    pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
                    decoder_scores_pos, sequence_output_pos, decoder_scores_pos_entity, decoder_scores_pos_action,sequence_output_pos_entity,sequence_output_pos_action = model(video, video_mask,
                                                                                                input_caption_ids=pairs_input_caption_ids_pos,
                                                                                                decoder_mask=pairs_decoder_mask_pos,
                                                                                                schedule_sampler=schedule_sampler,
                                                                                                video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)

                if args.prototype:
                    decoder_scores, sequence_output , decoder_scores_entity, decoder_scores_action,sequence_output_entity,sequence_output_action,fea_loss, cst_loss, dis_loss= model(video, video_mask,
                                                                                            input_caption_ids=pairs_input_caption_ids,
                                                                                            decoder_mask=pairs_decoder_mask,
                                                                                            schedule_sampler=schedule_sampler,
                                                                                            video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)
                    pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
                    decoder_scores_pos, sequence_output_pos, decoder_scores_pos_entity, decoder_scores_pos_action, sequence_output_entity, sequence_output_action,fea_loss, cst_loss, dis_loss = model(
                        video, video_mask,
                        input_caption_ids=pairs_input_caption_ids_pos,
                        decoder_mask=pairs_decoder_mask_pos,
                        schedule_sampler=schedule_sampler,
                        video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)


                pairs_output_caption_ids_pos = pairs_output_caption_ids_pos.view(-1, pairs_output_caption_ids_pos.shape[-1])
                pairs_output_caption_ids_neg = pairs_output_caption_ids_neg.view(-1, pairs_output_caption_ids_neg.shape[-1])
                cap_nouns_ids = cap_nouns_ids.view(-1, cap_nouns_ids.shape[-1])
                cap_verbs_ids = cap_verbs_ids.view(-1, cap_verbs_ids.shape[-1])
                cap_pos_nouns_ids = cap_pos_nouns_ids.view(-1, cap_pos_nouns_ids.shape[-1])
                cap_pos_verbs_ids = cap_pos_verbs_ids.view(-1, cap_pos_verbs_ids.shape[-1])
                cap_neg_nouns_ids = cap_neg_nouns_ids.view(-1, cap_neg_nouns_ids.shape[-1])
                cap_neg_verbs_ids = cap_neg_verbs_ids.view(-1, cap_neg_verbs_ids.shape[-1])


                loss_an = model.decoder_loss_fct(decoder_scores.view(-1, model.bert_config.vocab_size),
                                                 pairs_output_caption_ids.view(-1))
                loss_pos = model.decoder_loss_fct(decoder_scores_pos.detach().view(-1, model.bert_config.vocab_size),
                                                     pairs_output_caption_ids_pos.view(-1))
                loss_neg = model.decoder_loss_fct(decoder_scores.detach().view(-1, model.bert_config.vocab_size),
                                                     pairs_output_caption_ids_neg.view(-1))  # 我们发现，使用decoder_scores 替换 decoder_scores_neg 效果会更好

                loss_an_entity = model.decoder_loss_fct(decoder_scores_entity.detach().view(-1, 401),
                                                 cap_nouns_ids.view(-1))
                loss_pos_entity  = model.decoder_loss_fct(decoder_scores_pos_entity.detach().view(-1, 401),
                                                     cap_pos_nouns_ids.view(-1))
                loss_neg_entity  = model.decoder_loss_fct(decoder_scores_entity.detach().view(-1, 401),
                                                     cap_neg_nouns_ids.view(-1))  # 我们发现，使用decoder_scores_entity 替换 sequence_output_neg_action效果会更好


                loss_an_action = model.decoder_loss_fct(decoder_scores_action.detach().view(-1, 201),
                                                 cap_verbs_ids.view(-1))
                loss_pos_action  = model.decoder_loss_fct(decoder_scores_pos_action.detach().view(-1, 201),
                                                     cap_pos_verbs_ids.view(-1))
                loss_neg_action  = model.decoder_loss_fct(decoder_scores_action.detach().view(-1, 201),
                                                     cap_neg_verbs_ids.view(-1)) ## 我们发现，使用decoder_scores_action替换 decoder_scores_neg_entity 效果会更好

                loss_sentnece = (loss_pos+loss_an) / (2*( loss_neg + 1e-7))
                loss_entity = (loss_pos_entity + loss_an_entity) / (2 * (loss_neg_entity + 1e-7))
                loss_action = (loss_pos_action + loss_an_action) / (2 * (loss_neg_action + 1e-7))
                loss = (loss_sentnece + loss_action + loss_entity) / 3

                if  args.prototype:
                    #loss = (loss + 0.1 * fea_loss + dis_loss * 0.0001)/2    #       MSRVTT
                    #loss = loss + 0.1 * fea_loss + dis_loss * 0.00001      #
                    loss = loss + fea_loss
                else:
                    pass
            else:
                if not args.prototype:
                    decoder_scores, sequence_output = model(video,  video_mask,
                                                                  input_caption_ids=pairs_input_caption_ids,
                                                                  decoder_mask=pairs_decoder_mask,
                                                                  schedule_sampler=schedule_sampler,
                                                                  video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)
                    # decoder_scores.shape [256, 48, 30522]) [batchsize, 48, vocabsize])

                    pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
                    decoder_scores_pos, sequence_output_pos= model(
                        video, video_mask,
                        input_caption_ids=pairs_input_caption_ids_pos,
                        decoder_mask=pairs_decoder_mask_pos,
                        schedule_sampler=schedule_sampler,
                        video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)

                else:
                    decoder_scores, sequence_output,fea_loss, cst_loss, dis_loss = model(video,  video_mask,
                                                                  input_caption_ids=pairs_input_caption_ids,
                                                                  decoder_mask=pairs_decoder_mask,
                                                                  schedule_sampler=schedule_sampler,
                                                                  video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)
                    decoder_scores_pos, sequence_output_pos,fea_loss, cst_loss, dis_loss = model(
                        video, video_mask,
                        input_caption_ids=pairs_input_caption_ids_pos,
                        decoder_mask=pairs_decoder_mask_pos,
                        schedule_sampler=schedule_sampler,
                        video_swin=video_swin if args.video_feat_type=='Swin+Clip' else None)

                pairs_output_caption_ids_pos = pairs_output_caption_ids_pos.view(-1,
                                                                                 pairs_output_caption_ids_pos.shape[
                                                                                     -1])
                pairs_output_caption_ids_neg = pairs_output_caption_ids_neg.view(-1,
                                                                                 pairs_output_caption_ids_neg.shape[
                                                                                     -1])
                loss_an = model.decoder_loss_fct(decoder_scores.view(-1, model.bert_config.vocab_size),
                                                 pairs_output_caption_ids.view(-1))
                loss_pos = model.decoder_loss_fct(
                    decoder_scores_pos.detach().view(-1, model.bert_config.vocab_size),
                    pairs_output_caption_ids_pos.view(-1))
                loss_neg = model.decoder_loss_fct(decoder_scores.detach().view(-1, model.bert_config.vocab_size),
                                                  pairs_output_caption_ids_neg.view(-1)) #   我们发现，使用 decoder_scores 替换  decoder_scores_neg 效果会更好
                loss = (loss_pos + loss_an) / (2 * (loss_neg + 1e-7))
                if  args.prototype:
                    loss = (loss + 0.1*fea_loss + dis_loss * 0.0001)/2   #        MSVD
                    #loss = loss + 0.1 * fea_loss + dis_loss * 0.0001    #        MSRVTT

                else:
                    pass

        if n_gpu > 1:
            #计算损失的平均值。这是因为在多GPU环境下，每个GPU都会计算一部分数据的损失，所以需要将这些损失取平均，得到整个批次的损失。
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:  #如果梯度累积步数大于1，
            #将损失除以梯度累积步数。这是因为在梯度累积中，每个步骤都会计算一部分数据的损失并累积梯度，所以需要将损失平均，得到每个步骤的损失。
            loss = loss / args.gradient_accumulation_steps
        loss.backward() #反向传播
        total_loss += float(loss) #累加每个批次的损失，以计算总损失
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #对梯度进行裁剪，以防止梯度爆炸

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad() #清零优化器中的梯度，以便于下一步的计算

            global_step += 1 # 全局步数加一。全局步数通常用于记录自训练开始以来已经处理过的批次总数
            if global_step % log_step == 0 and local_rank == 0:
                #记录一条信息，包括当前的周期数、总周期数、当前步数、总步数、学习率、损失值和每步的时间。

                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Total_Loss: %f,  Time/step: %f",
                            epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()  # 更新开始时间为当前时间

    total_loss = total_loss / len(train_dataloader) #计算平均损失，即总损失除以批次的数量。
    return total_loss, global_step


# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    #这个函数接受一个实例索引列表，然后返回一个字典，字典的键是实例索引，值是该实例在张量中的位置。
    #主要作用是创建一个映射，该映射指示了每个实例在张量中的位置。
    #这在处理批量数据时非常有用，因为我们需要跟踪每个实例在批量数据中的位置。
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
#

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''
    #用于收集与活动实例相关的张量部分
    """
    在处理序列数据（如在序列到序列模型中）时，我们通常会遇到一个问题，那就是在每个时间步，不同的实例可能会在不同的时间完成。
    例如，在机器翻译或文本生成任务中，一旦一个实例生成了结束标记，我们就认为这个实例已经完成，不再需要进一步处理。
    然而，由于我们通常是在批次中处理数据，所以在一个批次中，可能同时包含已完成的实例和未完成的实例。
    这就需要我们能够区分这两类实例，以便只对未完成的实例（即活动实例）进行处理。
    collect_active_part函数就是用来解决这个问题的。它接受一个张量和一组活动实例的索引，然后从张量中收集与这些活动实例相关的部分。
    这样，我们就可以有效地跟踪哪些实例仍然需要处理，哪些实例已经完成，从而提高计算效率
    """
    #首先获取张量的维度，然后将张量重塑为新的形状
    _, *d_hs = beamed_tensor.size() 
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)
    
    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)#使用index_select函数从张量中选择活动实例的部分
    beamed_tensor = beamed_tensor.view(*new_shape) #最后再次将张量重塑为新的形状。
    
    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    #用于收集活动实例的信息
    """
    在处理序列数据（如在序列到序列模型中）时，收集与活动实例相关的信息是非常重要的。
    这是因为在每个时间步，我们只对那些还没有完成的实例（即活动实例）进行处理。
    例如，在机器翻译或文本生成任务中，一旦一个实例生成了结束标记，我们就认为这个实例已经完成，不再需要进一步处理。
    通过收集与活动实例相关的信息，我们可以有效地跟踪哪些实例仍然需要处理，哪些实例已经完成。
    这样，我们可以避免在已完成的实例上浪费计算资源，同时确保所有活动实例都得到适当的处理。
    此外，这也使得我们能够在所有实例完成后及时停止处理，从而提高计算效率。
    """
    assert isinstance(input_tuples, tuple)
    visual_output_rpt, video_mask_rpt = input_tuples #：从输入元组中解包出visual_output_rpt和video_mask_rpt。

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map) #：计算之前活动实例的数量。
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list] #从映射中获取活动实例的位置。
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device) #将活动实例的位置转换为长整型张量，并移动到指定的设备上。

    #收集与活动实例相关的visual_output_rpt部分。
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    #：收集与活动实例相关的video_mask_rpt部分。
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    #获取活动实例索引到张量位置的映射。
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return (active_visual_output_rpt, active_video_mask_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):
    ## 实现集束搜索（Beam Search）的解码

    #接收 解码器、集束、解码序列长度、实例索引到位置映射、集束大小、设备和输入元组等参数。如果提供了解码长度，它也会接收这个参数

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        #这个函数从每个未完成的集束中获取当前状态，并将其堆叠成一个张量。然后，它将这个张量的形状变为（-1，len_dec_seq），并返回。
        #inst_dec_beams 包含多个集束（Beam）对象的列表。每个集束对象都有一个get_current_state方法和一个done属性。
        #len_dec_seq 解码序列的长度。

        #dec_partial_seq 一个列表，包含了所有未完成的集束的当前状态。这些状态是通过调用每个集束的get_current_state方法获取的
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        #这行代码将dec_partial_seq列表中的所有张量堆叠成一个新的张量，并将这个新张量移动到指定的设备上。。
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        #这行代码将dec_partial_seq张量的形状变为（-1，len_dec_seq）。这里的-1表示该维度的大小会自动计算，以保证张量中的元素总数不变
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq #返回处理后的dec_partial_seq张量。

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples): #在每个解码步骤中，根据当前的解码状态和输入，预测下一个单词的概率分布
        #接收下一个解码器的id  (next_decoder_ids)、活动实例的数量  (n_active_inst)、集束大小 (n_bm)、设备device和输入元组input_tuples等参数。
        visual_output_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device) #全1的掩码，它的大小与next_decoder_ids相同。
        #解码输出
        dec_output = decoder(visual_output_rpt, video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1) #对解码输出应用对数softmax函数计算 单词概率
        word_prob = word_prob.view(n_active_inst, n_bm, -1) #最后，它将单词概率的形状变为（n_active_inst，n_bm，-1），并返回
        return word_prob




    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):

        """
        收集活动的实例索引。它遍历每个实例索引，并使用单词概率更新对应的集束。
        如果集束还没有完成，它就将实例索引添加到活动实例索引列表中。
        这样，在下一个解码步骤中，我们就只需要处理这些活动的集束，从而提高解码的效率。
        #inst_beams  包含多个集束（Beam）对象的列表。每个集束对象都有一个advance方法.
        #word_prob   单词概率，它是一个张量
        #inst_idx_to_position_map 字典，它将实例索引映射到位置。
        """
        active_inst_idx_list = [] #空列表，用于存储活动的实例索引。
        for inst_idx, inst_position in inst_idx_to_position_map.items(): #遍历inst_idx_to_position_map字典的每一项 
            if decoder_length is None:
                #如果没有提供decoder_length，这行代码会调用集束的advance方法，并将结果赋值给is_inst_complete。
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:#如果is_inst_complete为False，这行代码会将inst_idx添加到active_inst_idx_list。
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list
    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances 
    # #
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    #inst_dec_beams 包含多个集束（Beam）对象的列表。每个集束对象都有一个sort_scores方法和一个get_hypothesis方法
    #要收集的最佳假设的数量。
    all_hyp, all_scores = [], [] #两个空列表，用于存储所有的假设和分数。
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores() #调用集束的sort_scores方法，返回排序后的分数和尾部索引
        all_scores += [scores[:n_best]] #将前n_best个分数添加到all_scores列表。

        #获取前n_best个假设，并将它们添加到hyps列表。
        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps] #将hyps列表添加到all_hyp列表。
    return all_hyp, all_scores

# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None):
    """
        to evaluate the model. the sentence is generated via beam search
    """
    if hasattr(model, 'module'):
        model = model.module.to(device)

    all_result_lists = []
    all_caption_lists = []
    model.eval()

    #计算时间消耗
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0
    #
    for batch in tqdm(test_dataloader, desc='validation'):
        batch = tuple(t.to(device, non_blocking=True) for t in batch) #将批次中的所有张量转移到设备上。

        if args.video_feat_type == 'Swin+Clip':
            input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                video_swin, video_mask_swin, masked_video_swin, video_labels_index_swin, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                pairs_decoder_mask_pos, pairs_decoder_mask_neg, pairs_input_caption_ids_pos, pairs_output_caption_ids_pos, pairs_input_caption_ids_neg, pairs_output_caption_ids_neg, \
                cap_verbs_ids, cap_nouns_ids, cap_pos_verbs_ids, cap_pos_nouns_ids, cap_neg_verbs_ids, cap_neg_nouns_ids = batch
        else:
            input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                pairs_decoder_mask_pos, pairs_decoder_mask_neg, pairs_input_caption_ids_pos, pairs_output_caption_ids_pos, pairs_input_caption_ids_neg, pairs_output_caption_ids_neg, \
                cap_verbs_ids, cap_nouns_ids, cap_pos_verbs_ids, cap_pos_nouns_ids, cap_neg_verbs_ids, cap_neg_nouns_ids = batch



        # print("Before Infer")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
        reserved_memory = torch.cuda.memory_reserved(device)  # 已保留的显存
        free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
        # print("  - 总显存: {:.2f} GB".format(total_memory / (1024 ** 3)))
        # print("  - 已分配的显存: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
        # print("  - 已保留的显存: {:.2f} GB".format(reserved_memory / (1024 ** 3)))
        # print("  - 剩余可用显存: {:.2f} GB".format(free_memory / (1024 ** 3)))


        with torch.no_grad():

            if not args.prototype:
                visual_output = model.get_visual_output(video, video_mask,video_swin=video_swin if args.video_feat_type == 'Swin+Clip' else None)[-1] #获取模型的视觉输出

            else:
                visual_output,fea_loss = model.get_visual_output(video, video_mask,video_swin=video_swin if args.video_feat_type == 'Swin+Clip' else None) #获取模型的视觉输出
            # -- Repeat data for beam search
            n_bm = 5 # beam_size #设置集束搜索的大小为5。
            device = visual_output.device
            n_inst, len_v, v_h = visual_output.size()

            decoder = model.decoder_caption
            # Note: shaped first, then decoder need the parameter shaped=True
            video_mask = torch.ones(visual_output.size(0), visual_output.size(1)).to(video.device)
            video_mask = video_mask.view(-1, video_mask.shape[-1]) #

            #复制视觉输出和视频掩码以进行集束搜索。
            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            # -- Prepare beams #为每个实例创建一个Beam对象，用于执行集束搜索。
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst)) #创建一个活动实例的索引列表
            #获取实例索引到张量位置的映射。
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1): #用于生成最大长度为args.max_words的句子。
                #在每个解码步骤中，我们更新活动实例的列表。
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (visual_output_rpt, video_mask_rpt))
                #：如果没有活动的实例，那么就结束循环。这通常意味着所有实例都已经找到了结束符"<EOS>"。
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>
                #更新视觉输出、视频掩码和实例索引到张量位置的映射。
                (visual_output_rpt, video_mask_rpt), \
                inst_idx_to_position_map = collate_active_info((visual_output_rpt, video_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)
            # print("During Infer")
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
            reserved_memory = torch.cuda.memory_reserved(device)  # 已保留的显存
            free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
            # print("  - 总显存: {:.2f} GB".format(total_memory / (1024 ** 3)))
            # print("  - 已分配的显存: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
            # print("  - 已保留的显存: {:.2f} GB".format(reserved_memory / (1024 ** 3)))
            # print("  - 剩余可用显存: {:.2f} GB".format(free_memory / (1024 ** 3)))

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1) #收集所有假设和得分
            result_list = [batch_hyp[i][0] for i in range(n_inst)] #从假设中获取结果列表。

            ##改变张量为(n, pairs_output_caption_ids.shape[-1])
            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            #用cpu方法将张量从GPU转移到CPU，然后使用detach方法将其从计算图中分离出来，这样可以防止在转换为numpy数组时计算梯度
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()


            for re_idx, re_list in enumerate(result_list):#将结果从标记ID转换为标记，并删除任何不需要的标记（如"[SEP]“和”[PAD]"）
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)

                #检查解码的文本列表中是否包含"[SEP]“或”[PAD]"标记。
                # 如果包含，那么就找到这个标记的索引，并将其之后的所有标记都删除。
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]") #[SEP]表示结束
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                #将解码的文本列表连接成一个字符串，列表中的每个标记之间用空格分隔。
                decode_text = ' '.join(decode_text_list)
                #首先删除所有的" ##“（这是BERT tokenizer用于表示词内分词的特殊标记），然后删除字符串两端的”##"和空格
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                #将处理后的文本添加到all_result_lists列表中
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list): #处理caption_list中的每个结果

                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_caption_lists.append(decode_text)


    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
    reserved_memory = torch.cuda.memory_reserved(device)  # 已保留的显存
    free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
    # print("  - 总显存: {:.2f} GB".format(total_memory / (1024 ** 3)))
    # print("  - 已分配的显存: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
    # print("  - 已保留的显存: {:.2f} GB".format(reserved_memory / (1024 ** 3)))
    # print("  - 剩余可用显存: {:.2f} GB".format(free_memory / (1024 ** 3)))
    # print('total_time_consume',total)
    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")#定义完整结果的保存路径。
        with open(hyp_path, "w", encoding='utf-8') as writer: #打开文件以写入完整的预测结果
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists): #遍历所有的预测结果。
                video_id, sub_id = test_set.iter2video_pairs_dict[idx] #获取视频ID、子ID和开始时间。
                #将视频ID、开始时间和预测的文本写入文件。
                start_time = test_set.data_dict[video_id]['start'][sub_id] 
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    hyp_path = os.path.join(args.output_dir, "hyp.txt") #定义纯预测结果的保存路径。
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")   # 定义实际标签的保存路径。
    with open(ref_path, "w", encoding='utf-8') as writer: # 打开文件并写入所有的实际标签。
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt" or args.datatype == "msvd":
        all_caption_lists = []
        #获取测试数据加载器中的句子字典和视频句子字典。
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        video_ids = set() #存储所有的视频ID。
        for idx in range(len(sentences_dict)): #用于处理句子字典中的每个条目。
            video_id, _ ,_,_, _ ,_,_, _,_,_= sentences_dict[idx] #获取视频ID并将其添加到集合中
            video_ids.add(video_id)
            sentences = video_sentences_dict[video_id] #取视频对应的所有句子，并将它们添加到all_caption_lists列表中。
            all_caption_lists.append(sentences)
        if args.datatype != "msvd": # if number of caption for each video is different, use this
            #如果数据类型不是"msvd"，那么需要对all_caption_lists进行额外的处理。
            #这是因为在"msvd"数据集中，每个视频的标签数量是相同的，而在其他数据集中，每个视频的标签数量可能不同。
            ##使用zip函数将all_caption_lists中的每个元素重新组合，然后将结果转换为列表。
            all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]
    # Evaluate
    #if args.datatype == "msrvtt"or args.datatype == "msvd":
    if args.datatype == "msvd":
        #用于存储所有的预测结果和实际的标签。
        all_result_dict = {}
        all_caption_dict = {}
        #两个for 分别用于处理所有的预测结果和实际的标签，并将预测结果和实际的标签添加到相应的字典中。
        for i in range(len(all_result_lists)):
            all_result_dict[i] = [all_result_lists[i]]
        for i in range(len(all_caption_lists)):
            all_caption_dict[i]=all_caption_lists[i]
        # Evaluate
        metrics_nlg = score(all_caption_dict,all_result_dict) #计算各种自然语言生成的评估指标
    else:
        #如果数据类型不是"msvd"，那么使用另一种方法来计算评估指标。
        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)

    #印出各种评估指标的值。
    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))

    return metrics_nlg

DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_val_test, "test":dataloader_msrvtt_val_test}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_val_test, "test":dataloader_msvd_val_test}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    #从预训练的BERT模型中加载分词器。
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank) #初始化模型。


    #NLGEval是一个用于自然语言生成（NLG）的评估代码库，它提供了各种无监督自动度量标准。
    #这些度量标准可以用于评估生成的文本与参考文本之间的相似性，
    #包括 BLEU，METEOR，ROUGE，CIDEr，SPICE，SkipThought，Embedding Average，Vector Extrema，Greedy Matching 等
    #NLGEval 类是 nlgeval 包的主要接口，提供了计算这些度量标准的方法。
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)
    
    assert args.datatype in DATALOADER_DICT #检查数据集是否在数据加载器字典中。

    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, "val")
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer, "test")


    if args.local_rank == 0: #打印验证和测试的信息。
        logger.info("***** Running val *****") 
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))
        logger.info("***** Running test *****") 
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        #从数据加载器字典中获取训练的数据加载器、数据的长度和采样器。
        train_dataloader, train_length= DATALOADER_DICT[args.datatype]["train"](args, tokenizer) #, train_sampler
        #设置优化训练的步数
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr #设置学习率系数
        if args.init_model:
            coef_lr = 1.0
        ##准备优化器、学习率调度器和模型
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        #初始化最佳分数和最佳模型文件。
        best_score = {"CIDEr": 0.00001}
        best_output_model_file = {"CIDEr": None}
        #检查目标指标和耐心指标是否在最佳分数的键中
        assert args.target_metric in best_score.keys()
        assert args.patience_metric in best_score.keys()
        global_step = 0 #初始化全局步数。
        #创建一个停止信号。这是一个包含两个元素的张量，第一个元素用于基于epoch数量的早停，第二个元素用于低度量值的早停。
        stop_signal = torch.zeros(2).cuda() # index 0 for early stopping based on num of epoch, index 1 for low metrics
        for epoch in range(args.epochs):
            #train_sampler.set_epoch(epoch)
            #在一个训练周期中训练模型，并获取训练损失和全局步数。
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)
            torch.cuda.empty_cache() ###########*
            if args.local_rank == 0: #本地排名为0 打印训练的信息，
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                #output_model_file = save_model(epoch, args, model, type_name="") # 保存模型
                if epoch > -1: #如果已经完成了至少一个周期，那么评估模型的性能
                    output_model_file = save_model(epoch, args, model, type_name="")  # 保存模型
                    torch.cuda.empty_cache() ###########*
                    metric_scores = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    torch.cuda.empty_cache() ###########*
                    for met in best_score.keys(): #用于处理每个评估指标
                        if met == args.target_metric: #检查当前的指标是否是目标指标或耐心指标。
                            # if metric_scores[met] <= 0.001:
                            #     #如果指标的分数小于或等于0.001，那么发送停止信号。
                            #     logger.warning("One of the metrics is less than 0.001. The training will be stopped.")
                            #     stop_signal[1] = 1
                            #     break
                            if best_score[met] <= metric_scores[met]:
                                #如果最佳分数小于或等于当前分数，那么更新最佳分数和最佳模型文件。
                                best_score[met] = metric_scores[met]
                                best_output_model_file[met] = output_model_file
                                if met==args.patience_metric:
                                    stop_signal[0] = 0
                            else: #否则，增加停止信号。
                                if met==args.patience_metric:
                                    stop_signal[0] += 1
                            logger.info("The best model based on {} is: {}, the {} is: {:.4f}".format(met, best_output_model_file[met], met, best_score[met]))
                    #这是一个循环，用于处理每个GPU。
                    # for gp in range(1, args.n_gpu): # TODO, try to use broadcast function of pytorch
                    #     #torch.distributed.send(stop_signal, dst=gp) #发送停止信号。
                    #如果停止信号大于或等于耐心值，那么提前停止训练。
                    if stop_signal[0]>=args.patience: #如果停止信号大于或等于耐心值，那么提前停止训练。
                        logger.warning("Early stopping, no improvement after {} epochs at local rank {}".format(args.patience, args.local_rank))
                        break
                    if stop_signal[1] == 1:
                        break
                else:
                     # for gp in range(1, args.n_gpu): # TODO, try to use broadcast function of pytorch
                    #     #torch.distributed.send(stop_signal, dst=gp)
                    #打印一条警告信息，表示在第epoch+1个周期后跳过了评估。
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))
            else:#本地排名不为0 
                #从源（src）为0的进程接收停止信号。这是为了让所有的GPU都知道是否需要停止训练。
                #torch.distributed.recv(stop_signal, src=0)
                if stop_signal[0]>=args.patience:  #如果停止信号大于或等于耐心值，那么提前停止训练。
                    #打印一条警告信息，表示在args.patience个周期后没有改进，所以提前停止了训练。
                    logger.warning("Early stopping, no improvement after {} epochs at local rank {}".format(args.patience, args.local_rank))
                    break
                elif stop_signal[1]==1:#否则，如果停止信号的第二个元素为1，那么也停止训练。
                    #打印一条警告信息，表示其中一个指标小于0.001。
                    logger.warning("One of the metrics is less than 0.001")
                    break
        #评估模型的性能                     
        if args.local_rank == 0:
            test_scores = {} #空字典，用于存储测试分数。
            for met in best_score.keys(): #用于处理每个评估指标。
                if met == args.target_metric: #检查当前的指标是否是目标指标。
                    model = None
                    #加载最佳模型。
                    model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file[met])
                    #评估模型的性能。
                    torch.cuda.empty_cache()
                    metric_scores = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    test_scores[met] = metric_scores #将评估分数添加到测试分数中。
            for met in test_scores.keys():#用于处理每个测试分数。
                #打印出基于最佳模型的测试分数。
                logger.info("Test score based on the best {} model ({}) : {}".format(met, best_output_model_file[met], str(test_scores[met])))
    elif args.do_eval:#否则，如果需要进行评估，那么执行以下代码。
        #评估模型的性能。
        eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

if __name__ == "__main__":
    main()
