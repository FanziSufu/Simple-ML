import numpy as np
from collections import Counter


class Word2vec:
    def __init__(self, dim=100, learning_rata=0.01, window_size=5, neg=10):
        self.dim = dim  # 词向量维度
        self.learning_rate = learning_rata  # 学习率
        self.window_size = window_size  # 窗口大小
        self.neg = neg  # 负采样个数

    def get_context(self, word_index, data):
        """
        获取目标单词的上下文,返回在输入语料集的索引
        :param word_index: 目标样本在输入语料集中的索引
        :param data: 输入语料集,语料集格式为一维数组list
        :return: 上下文在语料集的索引的数组
        """
        min_context_index = max(0, word_index - self.window_size)
        max_context_index = min(len(data), word_index + self.window_size)
        return data[min_context_index: word_index] + data[word_index: max_context_index]

    def get_neg_sample(self, word_index, array):
        """
        获取负采样样本,通过在辅助数组array随机选取索引值的方式
        :param word_index: 目标样本在词汇表中的索引
        :param array: 有1亿个元素的辅助数组
        :return: 负例样本在词汇表中的索引的数组
        """
        neg_sample = []
        while len(neg_sample) < self.neg:
            neg_sample_index = array[np.random.randint(10**8)]
            if neg_sample_index == word_index:  # 负采样样本不能与目标样本相同
                continue
            neg_sample.append(neg_sample_index)
        return neg_sample

    def training(self, data):
        """使用负采样的skip_gram模型训练数据,返回训练完毕的词向量和其权重"""
        words = Counter(data)  # 获取每个单词出现的次数
        vec = np.random.rand(len(words), self.dim) - 0.5  # 初始化词向量
        theta = np.random.rand(len(words), self.dim)  # 初始化词向量的权重
        vocab = {word: i for i, word in enumerate(words.keys())}  # 词汇表,每个单词对应一个索引值
        array = [0 for _ in range(10**8)]  # 创建辅助数组
        i_start = 0
        for word in vocab.keys():
            i_end = (words[word] / len(data)) * 10**8  # 这里简单地以频率作为构建依据,源码是用频率的3/4次方
            array[i_start: i_end] = vocab[word]  # 把辅助数组的值更新为词向量的索引值
            i_start += i_end
        np.random.shuffle(array)  # 洗牌

        for i in range(len(data)):  # 对输入语料表的每一个单词进行训练
            word = data[i]  # 目标单词
            context = self.get_context(i, data)  # 这里的i是目标单词在数据集的索引
            neg_sample = self.get_neg_sample(vocab[word], array)  # 这里的vocab[word]是目标单词在词汇表的索引
            sample = [vocab[word]] + neg_sample  # 把正例放在第0位组成包含正负例的训练样本
            # 根据公式更新词向量和权重
            for ct in context:
                e = 0
                x = data[ct]
                for j in range(len(sample)):
                    f = 1 / (1 + np.exp(np.dot(-vec[vocab[x]], theta[sample[j]].T)))  # 二元逻辑回归,使用sigmoid函数
                    # 正负例的处理方式稍有不同,j=0时,意味着对正例进行处理
                    g = (1 - f) * self.learning_rate if j == 0 else -f * self.learning_rate
                    e += g * theta[sample[j]]
                    theta[sample[j]] += g * vec[vocab[x]]  # 更新负例的权重
                vec[vocab[x]] += e  # 更新上下文的词向量
        return vec, theta
