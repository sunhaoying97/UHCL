"""
Manage beam search info structure.
Heavily borrowed from OpenNMT-py.
For code in OpenNMT-py, please check the following link (maybe in oldest version):
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch

class Constants():
    #存储和处理分词器的常量。它提供了一个方便的方式来获取和使用这些常量，
    """ Default constants for tokenizer """
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.BOS_WORD = '[CLS]' 
        #BOS_WORD代表"开始符号"（Beginning Of Sentence），在这里被定义为[CLS]。
        #在许多自然语言处理任务中，[CLS]通常被用作句子的开始标记
        self.EOS_WORD = '[SEP]' 
        #EOS_WORD代表"结束符号"（End Of Sentence），在这里被定义为[SEP]。在许多自然语言处理任务中，[SEP]通常被用作句子的结束标记

    @classmethod
    def from_tokenizer(cls, tokenizer): #根据给定的分词器创建一个包含正确常量的Constants实例
        instance = cls()
        instance.PAD = tokenizer.vocab[instance.PAD_WORD] #使用分词器的词汇表更新这个实例的常量
        instance.UNK = tokenizer.vocab[instance.UNK_WORD]
        instance.BOS = tokenizer.vocab[instance.BOS_WORD]
        instance.EOS = tokenizer.vocab[instance.EOS_WORD]
        return instance

class Beam():#实现集束搜索 Beam search
    '''Implementation of the beam search from the `"Beam Search Strategies for Neural Machine Translation"
        <https://aclanthology.org/W17-3207.pdf>` paper.
        Params:
            size: beam search width.
            device: device for running the algorithm.
            tokenizer: whether to use default or predefined tokenizer.
    '''

    def __init__(self, size, device=False, tokenizer=None):
        if tokenizer is None:
            self.constants = Constants()
        else:
            self.constants = Constants.from_tokenizer(tokenizer)

        self.size = size #集束的大小
        self._done = False
        # The score for each interface on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device) #分数
        self.all_scores = [] #

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), self.constants.BOS, dtype=torch.long, device=device)]

    def get_current_state(self): #返回当前时间步的解码序列
        "Get the outputs for the current timestep."
        """
        在集束搜索（Beam Search）算法中，"当前时间步的解码序列"指的是在当前时间步，我们已经解码出来的序列。
        例如，如果我们正在解码一个句子，当前时间步可能是第三个单词，那么当前时间步的解码序列就是前三个单词。
        get_current_state方法返回的是当前时间步的解码序列。
        具体来说，它返回的是一个包含了当前时间步所有候选序列的列表。每个候选序列都是一个单词的列表，表示一个可能的解码结果。
        例如，假设我们在解码一个句子，当前的候选序列是[“我”, “喜欢”]和[“我”, “讨厌”]，
        那么get_current_state方法就会返回这两个序列的列表：[[“我”, “喜欢”], [“我”, “讨厌”]]。
        这个方法的返回值在集束搜索算法中非常重要，因为它代表了我们当前的解码状态，可以用来生成下一个时间步的候选序列
        """
        return self.get_tentative_hypothesis()

    def get_current_origin(self):#返回的是当前时间步的前向指针
        #在集束搜索（Beam Search）算法中，"前向指针"是一个术语，用于追踪每个候选序列的来源。
        # 具体来说，对于每个时间步，我们都会生成一些新的候选序列，这些序列是通过在现有序列的末尾添加一个新的单词来生成的。
        # "前向指针"就是用来记录每个新生成的候选序列是从哪个现有序列生成的。
        #"前向指针"的信息在最后的回溯过程中非常重要，因为我们需要通过回溯"前向指针"来找出最优的解码序列
        """
        它返回的是一个包含了当前时间步所有候选序列的来源的列表。
        每个来源都是一个索引，表示对应的候选序列是从哪个现有序列生成的。
        例如，假设我们在解码一个句子，当前的候选序列是[“我”, “喜欢”]和[“我”, “讨厌”]，
        然后在下一个时间步，我们在这两个序列的末尾分别添加了"苹果"和"香蕉"，
        生成了四个新的候选序列：[“我”, “喜欢”, “苹果”]、[“我”, “喜欢”, “香蕉”]、[“我”, “讨厌”, “苹果”]和[“我”, “讨厌”, “香蕉”]。
        在这个过程中，"前向指针"就是用来记录[“我”, “喜欢”, “苹果”]是从[“我”, “喜欢”]生成的，[“我”, “讨厌”, “香蕉”]是从[“我”, “讨厌”]生成的，等等。
        所以，get_current_origin方法返回的就是这些来源的列表。
        """
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self): #表示集束搜索是否完成
        return self._done

    def advance(self, word_prob, word_length=None):#
        #接收单词概率和单词长度，更新集束的状态，并检查是否完成。
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1) #num_words单词的数量，等于word_prob的第二个维度的大小   # word_prob 单词概率
        # Sum the previous scores.
        if len(self.prev_ks) > 0: #检查是否有前向指针
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob) #将word_prob和self.scores相加，得到新的beam_lk。
        else:
            beam_lk = word_prob[0] #将beam_lk设置为word_prob的第一个元素。
        flat_beam_lk = beam_lk.view(-1) #将beam_lk的形状变为一维。

        #找出flat_beam_lk中最大的self.size个元素，返回它们的值和索引。
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort #
        self.all_scores.append(self.scores) #将当前的分数添加到self.all_scores列表
        self.scores = best_scores
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words #计算每个最佳分数对应的前向指针。
        self.prev_ks.append(prev_k) #将prev_k添加到self.prev_ks列表
        self.next_ys.append(best_scores_id - prev_k * num_words) #计算每个最佳分数对应的下一个单词的索引，并将它添加到self.next_ys列表。
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.constants.EOS:#检查最佳的候选序列是否已经结束。如果是，它会将self._done设置为True。
            self._done = True

        return self._done #表示集束搜索是否已完成。

    def sort_scores(self): #对集束中的分数进行排序，（以选择最佳候选序列）
        "Sort the scores."
        return torch.sort(self.scores, 0, True) # 0表示排序的维度 True表示降序

    def get_the_best_score_and_idx(self): #用于获取集束中最佳候选序列的分数和索引
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1] #这个方法返回第二高的分数和对应的索引。索引1表示列表的第二个元素，索引从0开始。

    def get_tentative_hypothesis(self): #用于获取当前时间步的解码序列
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1: #检查self.next_ys的长度是否为1。self.next_ys是一个列表，包含了每个时间步的输出
            dec_seq = self.next_ys[0].unsqueeze(1)#如果self.next_ys的长度为1，获取self.next_ys的第一个元素，并增加一个维度，结果赋值给dec_seq。
        else:
            _, keys = self.sort_scores()#对分数进行排序，并返回排序后的分数和对应的索引。这里我们只关心索引，所以只保留了keys
            hyps = [self.get_hypothesis(k) for k in keys]#对每个索引k调用get_hypothesis方法，获取对应的假设序列，并将所有的假设序列组成一个列表。
            hyps = [[self.constants.BOS] + h for h in hyps] #在每个假设序列的开头添加一个开始符号（BOS）
            dec_seq = torch.LongTensor(hyps) #将hyps列表转换为一个长整数张量，并将结果赋值给dec_seq。

        return dec_seq #当前时间步的解码序列。

    def get_hypothesis(self, k):#用于构造完整的假设序列
        #k 表示我们想要获取哪个假设序列
        """ Walk back to construct the full hypothesis. """
        hyp = [] #用于存储假设序列。
        for j in range(len(self.prev_ks) - 1, -1, -1):#这行代码遍历self.prev_ks列表的每一个元素。self.prev_ks是一个列表，包含了每个时间步的前向指针。
            hyp.append(self.next_ys[j+1][k]) #将self.next_ys[j+1][k]添加到hyp列表。self.next_ys是一个列表，包含了每个时间步的输出。
            k = self.prev_ks[j][k] #更新k的值，使其指向前一个时间步的前向指针

        return list(map(lambda x: x.item(), hyp[::-1])) #返回hyp列表的逆序，并将每个元素转换为Python的标准数据类型
