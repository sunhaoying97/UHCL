import dgl
import h5py
import os
import sys
import logging
import re
import spacy
import torch

import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utility.vocabulary import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from sklearn.metrics import precision_score, f1_score, recall_score

tqdm.pandas()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducible result
def init_seed(seed=1, use_cuda=False):
    np.random.seed(seed) #设置NumPy的随机种子
    torch.manual_seed(seed)#设置torch的随机种子
    if use_cuda:
        torch.cuda.manual_seed(seed) # # 如果使用CUDA，也设置CUDA的随机种子


# Initialize file handler object#初始化日志处理器对象
def init_log(save_dir='saved/log/', filename='log.txt', log_format='%(message)s'):
    logger = logging.getLogger(__name__)## 获取当前模块的日志器
    if not logger.hasHandlers():# # 如果日志器没有处理器，那么添加一个
        create_folder(save_dir)## 创建保存日志的文件夹
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)# # 设置基本的日志配置
        fh = logging.FileHandler(os.path.join(save_dir, filename))# # 创建一个文件处理器
        fh.setFormatter(logging.Formatter(log_format))## 设置文件处理器的格式
        logging.getLogger().addHandler(fh)## 将文件处理器添加到日志器中


def init_tensorboard(save_dir='saved/tensorboard/'):
    create_folder(save_dir)
    writer = SummaryWriter(save_dir)## 创建一个SummaryWriter对象
    return writer


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
"""
`dgl.batch(graphs)`是Deep Graph Library (DGL)中的一个函数，它将一组图（`graphs`）批处理成一个大图。
这个大图中的每个输入图都成为一个不相交的组成部分¹。节点和边被重新标记为不相交的段。
批处理图的主要优点是可以同时进行一批图的消息传递和读取，从而提高图计算的效率。这对于处理许多图样本的任务（如图分类任务）非常有用¹。
例如，如果你有一个包含多个图的列表`graphs`，你可以使用`dgl.batch(graphs)`将这些图批处理成一个大图。
然后，你可以在这个大图上执行各种图操作，比如消息传递或图读取。
这些操作在大图上执行时，可以并行处理所有的节点和边，从而大大提高计算效率。
总的来说，`dgl.batch(graphs)`提供了一种有效的方式来批处理和处理图数据，使得在大量图上进行计算变得更加高效。
"""
def convert_batched_graph_feat(features, adj):
    graphs = []
    weights = []
    ## 遍历邻接矩阵
    for k in range(len(adj)):
        u = []
        v = []
        w = []
        ## 创建边的源节点和目标节点列表，以及边的权重列表
        for i in range(100):
            for j in range(100):
                u.append(i)
                v.append(j)
                w.append(adj[k][i][j])
        ## 使用源节点和目标节点列表创建DGL图
        g = dgl.graph((u, v))
        graphs.append(g)
        weights.append(torch.stack(w))
    ## 将权重和特征转换为张量
    weights = torch.stack(weights)
    weights = torch.flatten(weights)
    features = torch.flatten(features, end_dim=1)
    graphs = dgl.batch(graphs)
    return graphs, features, weights

def calculate_longest_sentence(series): #计算给定的句子序列中最长的句子的长度
    tokenizer = spacy.load('en_core_web_sm') ## 加载Spacy的英文模型
    longest_sentence = 0
    for sentence in tqdm(series):
        sentence_len = 0
        # # 使用Spacy的分词器对句子进行分词，并计算句子的长度
        for word in tokenizer(sentence):
            sentence_len += 1
        if sentence_len > longest_sentence:
            longest_sentence = sentence_len #更新最长句子的长度

    return longest_sentence

def generate_caption_data(data='msvd_train', n_video=5, vocab=None, device="cuda",path="dataset/MSVD/captions/sents_%s_lc_nopunc.txt",
                         min_word_count=0):
    
    if 'msvd' in data:#读取字幕数据
        used_captions = pd.read_csv(path % data.split("_")[1],\
                                    sep='\t', header=None, names=["vid_id", "caption"])
        
    # Start index for MSVD data
    start_index = {"msvd_train": 1, "msvd_val": 1201, "msvd_test": 1301} 
    
    # Create a video_id query # #
    chosen_keys = ["vid%s" % x for x in range(start_index[data], start_index[data]+n_video)] #创建一个video_id查询
    used_captions = used_captions[used_captions['vid_id'].isin(chosen_keys)] ## 选择符合条件的字幕
    
    if vocab is None:
        # Instantiate new vocabulary ## 实例化新的词汇表
        vocab = Vocabulary()

        # Populate vocabulary ## 填充词汇表
        print("Populating vocab with %s..." % data)
        for caption in tqdm(used_captions['caption']):
            vocab.add_sentence(caption)
            
        print("Original number of words:",vocab.num_words)
        if min_word_count>0:
            vocab.filter_vocab(min_word_count)
        print("Filtered number of words:",vocab.num_words)
        
        # Create vector caption
        print("Converting sentences to indexes...")
        used_captions['vector'] = used_captions['caption'].progress_apply(lambda x: vocab.generate_vector(x))
        longest_sentence = vocab.longest_sentence
        
    else:
        # If using val_data/test_data ## 如果使用val_data/test_data
        longest_sentence = calculate_longest_sentence(used_captions['caption'])
        used_captions['vector'] = used_captions['caption'].progress_apply(lambda x: vocab.generate_vector(x, longest_sentence))
    ## 将字幕转换为张量
    flatten_captions = torch.tensor(used_captions['vector']).to(device=device)
    captions_vector = used_captions.groupby("vid_id", sort=False)['vector'].sum()\
                                                                .apply(lambda x: torch.tensor(x).reshape(-1, longest_sentence+2)\
                                                                .to(device=device)).to_dict()
    ## 返回字幕向量，展平的字幕，词汇表和使用的字幕
    return captions_vector, flatten_captions, vocab, used_captions

def generate_2d_3d_features(data='msvd_train', n_video=5,
                            f2d_path="MSVD-2D.hdf5", f3d_path="MSVD-3D.hdf5", device="cuda"):
    scn_2d = h5py.File(f2d_path, "r")
    scn_3d = h5py.File(f3d_path, "r")
    
    # Start index for MSVD data
    start_index = {"msvd_train": 1, "msvd_val": 1201, "msvd_test": 1301} 
    
    # Create a video_id query## 创建一个video_id查询
    chosen_keys = ["vid%s" % x for x in range(start_index[data], start_index[data]+n_video)]
    
    scn_2d_src, scn_3d_src = [], []
    for key in chosen_keys:
        scn_2d_src.append(scn_2d.get(key))
        scn_3d_src.append(scn_3d.get(key))
    ## 返回2D和3D特征
    return torch.tensor(scn_2d_src).to(device=device), torch.tensor(scn_3d_src).to(device=device)

def generate_node_features(data="msvd_train", n_video=5,
                             fo_path="MSVD_FO_FASTERRCNN_RESNET50.hdf5",
                             stgraph_path="MSVD_IOU_STG_FASTERRCNN_RESNET50.hdf5", device="cuda", generate_fo=True):
    #生成节点特征，这些特征可以用于后续的计算或分析
    if generate_fo:
        fo_file = stack_node_features(fo_path)
    stgraph_file = h5py.File(stgraph_path, "r")

    # Start index for MSVD data ## 定义MSVD数据的起始索引
    start_index = {"msvd_train": 1, "msvd_val": 1201, "msvd_test": 1301} 
    
    # Create a video_id query ## 创建一个video_id查询
    excluded_keys = []
    for vid in fo_file.keys():
        if len(fo_file[vid]) != 100:
            excluded_keys.append(vid)
    chosen_keys = ["vid%s" % x for x in range(start_index[data], start_index[data]+n_video) if "vid%s" % x not in excluded_keys]
    
    
    fo_input, stgraph = [], []
    for key in chosen_keys:
        if generate_fo:
            fo_input.append(fo_file.get(key))
        stgraph.append(stgraph_file.get(key))
    
    if generate_fo:
        return torch.tensor(fo_input).to(device=device), torch.tensor(stgraph).to(device=device), excluded_keys
    return torch.tensor(stgraph).to(device=device)

def stack_node_features(pathfile):

    fo_input = h5py.File(pathfile, "r") #
    fo_list = {} #存储读取的数据
    for i,key in tqdm(enumerate(fo_input.keys()), total=len(fo_input.keys())):
        a = key.split('-')## 将键按照'-'进行分割

        if a[0] not in fo_list:
            fo_list[a[0]] = {}
        fo_list[a[0]][int(a[1])] = fo_input[key][:] ## 将数据存储到字典中

    fo_stacked = {} #存储处理后的数据
    for key in fo_list.keys():
        stacked = []
        for k_fr in sorted(fo_list[key].keys()): #排序
            stacked.append(fo_list[key][k_fr]) 
        fo_stacked[key] = np.vstack(stacked)## 将列表中的数据沿着第0维堆叠起来，然后存储到字典中
        
    return fo_stacked

def score(ref, hypo, metrics=[]):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
    
    metrics, eg. ['bleu', 'meteor','rouge_l','cider']
    """
    scorers = {
        "bleu" : (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        "meteor" : (Meteor(),"METEOR"),
        "rouge_l" : (Rouge(), "ROUGE_L"),
        "cider" : (Cider(), "CIDEr")
    }
    final_scores = {}
    for key in metrics:
        scorer, method = scorers[key]
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
        'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
        
        'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
        
        'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
        'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
        'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }