import numpy as np
from collections import Counter
from sklearn import datasets


class KNN:
    def __init__(self, k=1):
        self.k = k  # 近邻数量

    def training(self, text_vec, dataset):
        """
        线性扫描分类，即计算待测点与训练集所有实例的距离，选出最近k个实例，投票表决分类结果。
        距离度量使用欧氏距离。 该方法不适合样本巨多的数据集。
        :param text_vec: 待分类的样本 np.array   shape=(n,)
        :param dataset: 训练集 np.array shape=(m,n+1), 包含标签列
        :return: 分类结果
        """
        distances = {}  # 利用字典存储 索引-距离 信息
        for index, example in enumerate(dataset[:, : -1]):
            distance = np.sum(np.power(text_vec - example, 2)) ** 0.5  # 欧氏距离
            distances[index] = distance
        distances = [(a, b) for a, b in distances.items()]  # 把字典转换成元组列表，方便排序
        distances.sort(key=lambda x: x[1])  # 按距离从小到大排序
        k_nearest_neighbor = distances[: self.k]  # 选出前k个最近邻
        labels = [dataset[a[0], -1] for a in k_nearest_neighbor]  # 利用索引，找到k个近邻的类别
        return Counter(labels).most_common(1)[0][0]  # 投票选择出现次数最多的类别，并返回该类别


def test():
    dataset = datasets.make_classification()
    features, target = dataset
    target = target.reshape(-1, 1)
    dataset = np.concatenate((features, target), axis=1)
    knn = KNN(k=6)
    prediction = []
    for x in features:
        prediction.append(knn.training(x, dataset))
    correct = [1 if a == b else 0 for a, b in zip(prediction, target)]
    print(correct.count(1) / len(correct))


test()
