import numpy as np
from sklearn import preprocessing as pp
from sklearn import datasets


class PCA:
    def __init__(self, k=10):
        self.k = k  # 目标维度

    def training(self, dataset, svd=False):
        """
        使用主成分分析算法，把n维数据集降为k维数据
        :param dataset: 数据集n*m,n为特征数,m为样本数,np.array
        :param svd: 是否使用奇异值分解求特征值
        :return: 降维后的数据集k*m，和信息量占比t
        """
        dataset = pp.scale(dataset, axis=1)  # 去均值化
        cov_mat = np.cov(dataset)  # 计算协方差矩阵
        if svd:  # 使用SVD（奇异值分解）求解
            u, s, vt = np.linalg.svd(cov_mat)  # s存储奇异值，u存储对应的特征向量
            eig_vals = s[: self.k]  # 选取前K个奇异值，对应特征值
            red_eig_vects = u[:, : self.k]  # 选取对应的特征向量， n*k
            t = np.sum(eig_vals) / np.sum(s)  # 计算信息保留度

        else:  # 使用求解特征值和特征矩阵的方式
            eig_vals, eig_vects = np.linalg.eig(cov_mat)  # 计算协方差矩阵的特征值和特征向量
            eig_vals_ind = np.argsort(eig_vals)[:: -1][: self.k]  # 获取前k个最大特征值的索引
            red_eig_vects = eig_vects[:, eig_vals_ind]  # 根据索引获取对应的特征向量
            # 在PCA，特征值等于对应特征向量*原数据后的方差，这里用方差代表信息量，该值衡量降维后保留的原数据多少的信息量
            t = np.sum(eig_vals[eig_vals_ind]) / np.sum(eig_vals)
        low_dim_data = np.dot(red_eig_vects.T, dataset)  # 特征向量*去均值化的原数据=降维后的数据
        return low_dim_data, t


def test():
    dataset = datasets.make_classification(n_samples=3000, n_features=100)[0]
    pca = PCA(k=60)
    print(pca.training(dataset, svd=True)[1])
    print(pca.training(dataset, svd=False)[1])


test()
