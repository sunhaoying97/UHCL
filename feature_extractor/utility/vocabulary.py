import spacy
# we do not used this vocabulary class
class Vocabulary:#创建和管理词汇表
    PAD_token = 0   # Used for padding short sentences
    BOS_token = 1   # Beginning-of-sentence token
    EOS_token = 2   # End-of-sentence token
    UNK_token = 3   # Unknown word token

    def __init__(self):
        self.word2index = {}## 单词到索引的映射
        self.word2count = {} #单词的计数
        self.index2word = {self.PAD_token: "<PAD>", self.BOS_token: "<BOS>", self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}## 索引到单词的映射
        self.num_words = 4## 词汇表中的单词数量
        self.num_sentences = 0# # 句子的数量
        self.longest_sentence = 0 ## 最长的句子的长度
        self.tokenizer = spacy.load('en_core_web_sm')## 分词器

    def add_word(self, word):## 向词汇表中添加一个单词
        if word not in self.word2index:#
            # First entry of word into vocabulary#  如果是单词在词汇表中的第一次出现
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1## 如果单词已经存在，增加单词的计数
            
    def add_sentence(self, sentence):## 向词汇表中添加一个句子
        sentence_len = 0
        for word in self.tokenizer(sentence):
            sentence_len += 1
            self.add_word(str(word))
        if sentence_len > self.longest_sentence: ## 如果这是最长的句子
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences #  # 计数句子的数量
        self.num_sentences += 1
    
    def generate_vector(self, sentence="Hello", longest_sentence=None): #一个句子转换为一个向量
        # Validation data/test data may have longer sentence, so a parameter longest sentence provided
        if longest_sentence is None:
            longest_sentence = self.longest_sentence
        
        vector = [self.BOS_token] ## 向量开始于句子开始标记
        sentence_len = 0 
        for word in self.tokenizer(sentence):
            vector.append(self.to_index(str(word))) # # 将单词转换为索引并添加到向量中
            sentence_len += 1
        vector.append(self.EOS_token) ## 向量结束于句子结束标记
        
        # Add <PAD> token if needed      ## 如果需要，添加<PAD>标记
        if sentence_len < longest_sentence:
            for i in range(sentence_len, longest_sentence):
                vector.append(self.PAD_token)
        
        return vector

    def to_word(self, index):
        return self.index2word[index] ## 返回索引对应的单词

    def to_index(self, word): ## 如果单词不在词汇表中，返回未知单词标记
        if word not in self.word2index:
            return self.UNK_token
        
        return self.word2index[word]
    
    def filter_vocab(self, min_word_count=0): #过滤掉词汇表中出现次数少于min_word_count的单词
        word2count = self.word2count## 获取当前词汇表的单词计数
        ## 重置词汇表
        self.num_words = 4
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "<PAD>", self.BOS_token: "<BOS>", self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}
        for word, count in word2count.items(): ## 遍历当前词汇表的单词计数
            if count>=min_word_count:# # 如果单词的计数大于或等于min_word_count，那么将其添加到新的词汇表中
                self.word2index[word] = self.num_words
                self.word2count[word] = count
                self.index2word[self.num_words] = word
                self.num_words += 1
