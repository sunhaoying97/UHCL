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
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import os
import sys
import logging

from .file_utils import cached_path

logger = logging.getLogger(__name__)
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'base-uncased': 512,
    'large-uncased': 512,
    'base-cased': 512,
    'large-cased': 512,
    'base-multilingual-uncased': 512,
    'base-multilingual-cased': 512,
    'base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    #用于从文件中加载词汇表（vocabulary），并将其存储在一个有序字典（vocab）中。每个词汇表中的词语都与一个索引（index）关联。
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    #对输入的文本进行基本的空格清理和分割，返回一个由单词组成的列表。
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting"""
    #vocab_file 词汇表文件的路径 #do_lower_case 是否将单词转换为小写（默认为 True） 
    #max_len 最大序列长度（如果不指定，默认为一个很大的值）。
    #never_split 不需要拆分的特殊标记（默认为 ("[UNK]", "[SEP]", "[MASK]", "[CLS]")）
    def __init__(self, vocab_file, do_lower_case=True, max_len=None, never_split=("[UNK]", "[SEP]", "[MASK]", "[CLS]")):
        if not os.path.isfile(vocab_file): #检查词汇表文件是否存在，如果不存在则引发一个错误。
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file) #加载词汇表并存储在 self.vocab 中。
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]) #创建一个有序字典 self.ids_to_tokens，将词汇表中的词语与其索引关联。
        #初始化基本分词器 self.basic_tokenizer 和 WordPiece 分词器 self.wordpiece_tokenizer
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):#将一系列输入的标记（tokens）转换为对应的词汇表中的索引（ids）。 #tokens 一个包含标记的列表 
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token not in self.vocab: #对于输入的每个标记，检查它是否在词汇表中。
                ids.append(self.vocab["[UNK]"]) #如果标记不在词汇表中，将其替换为 [UNK]（表示未知标记），并记录错误日志。
                logger.error("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                ids.append(self.vocab[token]) #否则，将标记对应的词汇表索引添加到结果列表中。
        if len(ids) > self.max_len: #如果结果列表的长度超过了最大序列长度（self.max_len），则引发一个错误。
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids#一个包含对应词汇表索引的列表

    def convert_ids_to_tokens(self, ids): #将一系列词汇表索引转换为对应的标记 # ids 一个包含词汇表索引的列表
        """Converts a sequence of ids in tokens using the vocab.""" 
        #对于输入的每个索引，查找它在词汇表中对应的标记，并将标记添加到结果列表中。
        tokens = []
        for i in ids:

            tokens.append(self.ids_to_tokens[i])
        return tokens  # tokens 一个包含对应标记的列表
    def convert_ids_to_tokens_1(self, ids,k): #将一系列词汇表索引转换为对应的标记 # ids 一个包含词汇表索引的列表
        """Converts a sequence of ids in tokens using the vocab."""
        #对于输入的每个索引，查找它在词汇表中对应的标记，并将标记添加到结果列表中。
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i[k-1]])
        return tokens  # tokens 一个包含对应标记的列表

    @classmethod
    def from_pretrained(cls, pretrained_model_name, cache_dir=None, *inputs, **kwargs):
        #，该方法用于从预训练模型文件中实例化一个 PreTrainedBertModel 对象。如果需要，它会下载并缓存预训练模型文件
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        #是预训练模型的词汇表文件的路径。如果该文件不存在，那么它会在 PRETRAINED_VOCAB_ARCHIVE_MAP 中查找预训练模型的名称
        # 如果找不到，那么它会假设 pretrained_model_name 就是词汇表文件的路径
        vocab_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_model_name)
        if os.path.exists(vocab_file) is False:
            if pretrained_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP:
                vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name]
            else:
                vocab_file = pretrained_model_name
        if os.path.isdir(vocab_file): #如果vocab_file 是一个目录，那么它会尝试在该目录下查找名为 VOCAB_NAME 的词汇表文件
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        print(vocab_file) 
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir) 
            #函数会尝试在缓存目录中查找词汇表文件。如果找不到，那么它会抛出一个 FileNotFoundError 异常，并返回 None
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found. "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    vocab_file))
            return None
        #如果词汇表文件在缓存中被找到，那么它的路径会被赋值给 resolved_vocab_file。否则，resolved_vocab_file 的值就是 vocab_file
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            #如果预训练模型的名称在 PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP 中，那么它会确保分词器不会索引超过位置嵌入数量的序列。
            # 它还会设置一些永远不会被分割的标记，如 “[UNK]”, “[SEP]”, “[PAD]”, “[CLS]”, “[MASK]”。
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
            kwargs['never_split'] = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")

        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs) #使用 resolved_vocab_file 和其他参数来实例化一个分词器，并返回这个分词器。

        return tokenizer

    def add_tokens(self, new_tokens, model): ##用于向分词器类中添加新的令牌
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.
        Args:
            new_tokens: list of string. Each string is a token to add. Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).
        Returns:
            Number of tokens added to the vocabulary.
        Examples::
            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        #new_tokens 表示要添加的新令牌的列表。 #model 表示要调整其嵌入大小的模型
        to_add_tokens = [] #用于存储要添加的令牌。
        for token in new_tokens:
            assert isinstance(token, str) #检查token是否是字符串。如果不是，它会引发一个错误。
            to_add_tokens.append(token)
            # logger.info("Adding %s to the vocabulary", token)

        vocab = collections.OrderedDict() #创建一个有序字典，用于存储词汇表。
        for token in self.vocab.keys(): #遍历self.vocab字典的每一个键。
            vocab[token] = self.vocab[token] #将self.vocab字典中的每一个键值对复制到vocab字典。
        for token in to_add_tokens: #将token添加到vocab字典，并将其索引设置为vocab的长度。
            vocab[token] = len(vocab)
        self.vocab = self.wordpiece_tokenizer.vocab = vocab #更新self.vocab和self.wordpiece_tokenizer.vocab为新的vocab字典。
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]) #创建一个新的有序字典，其中键是索引，值是令牌。

        model.resize_token_embeddings(new_num_tokens=len(vocab)) #码调整模型的嵌入大小，以匹配新的词汇表大小。

class BasicTokenizer(object):#执行基本的文本分词（如标点符号分割、小写转换等）
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
    #定义了是否进行小写转换和永不分割的令牌列表等属性
    def __init__(self, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        #接收一段文本，首先对其进行清理，然后对中文字符进行分词，接着对空白处进行分词，
        # 最后对每个令牌进行进一步的处理（如小写转换、去除重音、分割标点符号等），并返回处理后的令牌列表。
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        #使用unicodedata.normalize函数对text进行Unicode规范化。"NFD"是规范化的形式，表示规范分解
        text = unicodedata.normalize("NFD", text)
        output = [] #用于存储处理后的字符。
        for char in text:
            cat = unicodedata.category(char)#使用unicodedata.category函数获取char的类别。unicodedata.category函数会返回一个字符串，表示char的类别。
            if cat == "Mn":
                #检查char的类别是否是"Mn"。"Mn"是Unicode字符类别，表示非间距组合记号，通常是重音符号。如果char的类别是"Mn"，函数会跳过这个字符
                continue
            output.append(char) #不是"Mn"，这行代码会将char添加到output列表。
        return "".join(output) #返回output列表中的字符组成的字符串。

    def _run_split_on_punc(self, text): #分割标点符号
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0 #计数器，用于追踪当前处理的字符的索引
        start_new_word = True #表示是否开始一个新的令牌。
        output = []
        while i < len(chars):
            char = chars[i]
            #检查char是否是标点符号。如果是，函数将char添加到output列表，并设置start_new_word为True
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char) #将char添加到output列表的最后一个列表
            i += 1

        return ["".join(x) for x in output] #返回output列表中的每个列表组成的字符串的列表。

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp): #检查char是否是中文字符。如果是，函数在output列表中添加一个空格、char和另一个空格。
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output) #返回output列表中的字符组成的字符串

    def _is_chinese_char(self, cp):
        #用于检查一个码点是否是中日韩（CJK）字符的码点
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.

        #检查cp是否在CJK Unicode块的范围内。这些范围包括了大部分的中文、日文和韩文汉字。如果cp在这些范围内，函数返回True。
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False #cp不在CJK Unicode块的范围内，函数返回False。

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char): #检查char是否是无效字符。如果是，函数会跳过这个字符。
                continue
            if _is_whitespace(char): #检查char是否是空白字符。如果是，函数在output列表中添加一个空格。
                output.append(" ")
            else:
                output.append(char) #如果char不是空白字符，这行代码将char添加到output列表
        return "".join(output) #返回output列表中的字符组成的字符串。

class WordpieceTokenizer(object): #用于执行 WordPiece 分词（tokenization）
    #WordPiece 是一种子词分词方法，用于将文本分割成更小的单元，例如词根、前缀、后缀等。它通常用于自然语言处理任务中，如机器翻译、文本分类等。
    #WordPiece 分词的目标是将输入文本拆分成一系列子词（word pieces），以便后续处理。
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab #词汇表，一个字典，将子词映射到其索引。
        self.unk_token = unk_token #未知标记（默认为 [UNK]）。
        self.max_input_chars_per_word = max_input_chars_per_word #每个单词的最大输入字符数（默认为 100）

    def tokenize(self, text): #用于将输入的文本分割成 WordPiece 子词。
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args: 
          #text 一个包含单词或由空格分隔的单词的字符串。
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens. #一个包含 WordPiece 子词的列表。
        """

        output_tokens = []
        for token in whitespace_tokenize(text):  #对于输入的每个单词，将其拆分成字符。
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word: #如果单词的字符数超过了最大输入字符数，将其替换为未知标记。
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars): #使用贪心的最长匹配算法，将单词拆分成 WordPiece 子词。
                end = len(chars)
                cur_substr = None
                while start < end:  
                    #它会从单词的开头开始，并尝试找到最长的子词，这个子词必须在词汇表中。
                    #这就是为什么它会将 end 设置为字符列表的长度，然后在一个循环中逐渐减小 end 的值。
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    #在每次迭代中，它都会尝试创建一个从 start 到 end 的子字符串。如果这个单词不是在单词的开头，那么它会在子字符串前面添加 “##”。
                    # 这是因为在 WordPiece 分词中，“##” 是用来表示一个子词是另一个单词的一部分，而不是一个独立的单词。
                    
                    
                    if substr in self.vocab:
                    #如果这个子字符串在词汇表中，那么它就会被接受为一个有效的子词，跳出循环，并将 start 设置为 end，然后继续寻找下一个子词
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None: #如果在 start 和 end 之间没有找到任何有效的子词，那么这个单词就被认为是 “bad”，并且会被替换为未知标记。
                    is_bad = True
                    break
                sub_tokens.append(cur_substr) # 如果找到了有效的子词，那么这些子词就会被添加到输出的子词列表中。
                start = end
            #如果无法拆分，将该单词替换为未知标记。
            if is_bad:
                output_tokens.append(self.unk_token)
            #否则，将子词添加到结果列表中。
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def _is_whitespace(char): #检查一个字符是否是空白字符
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r": #检查char是否是空格、制表符、换行符或回车符。
        return True #如果是，函数返回True
    cat = unicodedata.category(char) #使用unicodedata.category函数返回一个字符串， 返回char的类别
    if cat == "Zs":#检查char的类别是否是"Zs"。"Zs"是Unicode字符类别，表示空格分隔符。如果char的类别是"Zs"，函数返回True
        return True
    return False #如果char既不是空格、制表符、换行符或回车符，也不是空格分隔符，函数返回False。


def _is_control(char):#检查一个字符是否是控制字符
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        #检查char是否是制表符、换行符或回车符。
        #这些字符虽然技术上是控制字符，但在这里我们将它们视为非控制字符，所以如果char是这些字符中的任何一个，函数返回False
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):#检查char的类别是否以"C"开头。在Unicode字符类别中，以"C"开头的类别都是控制字符
        return True # 如果char的类别以"C"开头，函数返回True
    return False #如果char既不是制表符、换行符或回车符，也不是以"C"开头的类别，函数返回False。


def _is_punctuation(char):#检查一个字符是否是标点符号
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char) #使用ord函数获取char的Unicode码点。
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)): #检查cp是否在ASCII标点符号的范围内。如果是，函数返回True。
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"): 
        #检查char的类别是否以"P"开头。在Unicode字符类别中，以"P"开头的类别都是标点符号。如果char的类别以"P"开头，函数返回True
        return True
    return False#：如果char既不是ASCII标点符号，也不是以"P"开头的类别，函数返回False
