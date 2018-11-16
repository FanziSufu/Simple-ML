import numpy as np
import pandas as pd


class KMeans:
    def __init__(self, k=4):
        self.k = k  # 质心数量
    
    @staticmethod
    def rand_cent(features, k):
        """随机选取k个样本,作为初始质心"""
        m = features.shape[0]
        rand_x = np.random.choice(m, k, replace=False)  # 随机取样
        return features[rand_x, :]

    def training(self, features, k=None):
        """
        根据K均值算法,找出K个质心,对所有训练样本聚类,返回质心列表,和样本聚类结果
        本例只考虑连续型数据的聚类,数据格式为np.array
        :param features: 训练集m*n,m为样本数,n为特征数
        :param k: 指定质心数量
        :return: 质心列表,样本聚类结果数组（m*2,第0列存储类别索引,第1列存储到质心的欧式距离的平方）
        """
        if k is None: 
            k = self.k
        cen_list = self.rand_cent(features, k)  # 初始化质心
        cluster = np.zeros((features.shape[0], 2))  # 定义样本的属性
        cent_changed = True  # 控制开关，判断样本的所属质心是否发生改变
        while cent_changed:
            cent_changed = False  # 每次循环开始，先关闭开关
            for i, x in enumerate(features):  # 把每个样本划分到与其最近的质心
                max_dist = np.inf
                for j, cent in enumerate(cen_list):
                    dist = np.sqrt(np.sum(np.power(x - cent, 2)))
                    if dist < max_dist:
                        max_dist = dist
                        cluster[i, :] = j, dist**2  # 更新样本信息,前者为质心索引,后者为到质心距离的平方,用于二分K均值,计算SSE（平方误差和）
            for j in range(k):  # 更新质心
                xj = features[cluster[:, 0] == j, :]  # 选出所有该质心下的样本
                temp = np.mean(xj, axis=0)  # 按列计算样本均值 1*n
                if any(temp != cen_list[j]):  # 如果质心向量发生变化,打开开关,并更新质心向量
                    cent_changed = True
                    cen_list[j] = temp
        return cen_list, cluster

    def binary_kmeans(self, features):
        """
        根据二分K均值算法,初始化训练集为一个簇,按使总体SSE最小的方向,一次增加一个簇,最终生成k个簇
        本例只考虑连续型数据的聚类,数据格式为np.array
        :param features: 训练集m*n,m为样本数,n为特征数
        :return: 质心列表,样本聚类结果数组（m*2,第0列存储类别索引,第1列存储到质心的橘绿）
        """
        m = features.shape[0]
        centroid0 = np.mean(features, axis=0)  # 初始化，把整个数据集视为一个簇，质心是数据集的均值向量
        centroids = [centroid0]  # 二分法是 质心增殖 的过程，用列表来保存并更新质心
        cluster_assment = np.zeros((m, 2))  # 定义样本的属性， 所属质心（的索引）， 与 到质心的距离（的平方）
        for i in range(m):  # 更新初始簇的样本距离
            cluster_assment[i, 1] = np.sqrt(np.sum(np.power(features[i] - centroid0, 2))) ** 2
        while len(centroids) < self.k:
            min_sse = np.inf  # 初始化最小平方误差为无穷大
            best_cent2split, best_centroid, best_cluster_ass = None, None, None  # 声明变量
            for i in range(len(centroids)):  # 尝试对每一个簇进行 2-均值聚类，选择是总SSE最小的分法。
                split_x = features[np.nonzero(cluster_assment[:, 0] == i)[0], :]  # 截取待聚类的数据集
                if len(split_x) == 1:  # 如果数据集只有一个样本,那么跳过
                    continue  
                split_cent, split_cluster_ass = self.training(split_x, k=2)  # 使用第i簇的样本数据进行聚类
                split_sse = np.sum(split_cluster_ass[:, 1])  # 计算该簇聚类后的SSE
                # 计算非该簇的其他样本数据的SSE
                not_split_sse = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0] != i)[0], 1])
                if split_sse + not_split_sse < min_sse:  # 如果当前的总SSE是最小的，那么更新以下信息
                    min_sse = split_sse + not_split_sse  # 最小平法误差
                    best_cent2split = i  # 最佳聚类簇（的索引）
                    best_centroid = split_cent  # 最佳聚类簇聚类后的新质心
                    best_cluster_ass = split_cluster_ass.copy()  # 最佳聚类簇聚类后更新的样本属性信息
            # 把新样本信息的 所属质心 == 1的值，替换为质心列长度
            best_cluster_ass[np.nonzero(best_cluster_ass[:, 0] == 1)[0], :] = len(centroids)
            # 把新样本信息的 所属质心 == 0的值，替换为最佳质心的索引值
            best_cluster_ass[np.nonzero(best_cluster_ass[:, 0] == 0)[0], :] = best_cent2split
            # 更新外围的样本信息
            cluster_assment[np.nonzero(cluster_assment[:, 0] == best_cent2split)] = best_cluster_ass
            centroids[best_cent2split] = best_centroid[0]  # 更新质心列表
            centroids.append(best_centroid[1])  # 添加新质心
        return centroids, cluster_assment


def test():
    data = pd.read_csv('data/testSet.txt', delim_whitespace=True, names=['x1', 'x2'])
    data = np.array(data)
    km = KMeans(k=7)
    result = km.training(data)
    print(np.sum(result[1][:, 1]))
    result2 = km.binary_kmeans(data)
    print(np.sum(result2[1][:, 1]))


test()
