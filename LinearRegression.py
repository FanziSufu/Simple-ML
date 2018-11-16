import numpy as np
from sklearn import datasets


class LinearRegression:
    def __init__(self, learning_rate=0.01, lamb=0.0001, iters=10000):
        self.learning_rate = learning_rate  # 学习速率,默认值为1
        self.lamb = lamb  # 正则化参数,默认值为1
        self.iters = iters  # 最大迭代次数
        self.theta = np.zeros((1, 1))  # 声明参数是二维array格式
        self.cost = []  # 记录损失值

    @staticmethod  # 声明下方函数是静态的
    def cal_cost(features, target, theta, lamb):
        """
        最小均方算法（LMS algorithm），使用L2正则化,数据集均为np.array格式
        :param features: 特征集m*n, m=样本数, n=特征数
        :param target: 目标集m*1
        :param theta: 参数集1*(n+1)
        :param lamb: 正则化参数
        :return: 最小均方误差
        """
        features, target = map(np.array, (features, target))
        m = features.shape[0]
        inner = np.power(features.dot(theta.T) - target, 2)  # 核心方程式
        reg = lamb / (2 * m) * np.sum(np.power(theta[:, 1:], 2))  # 正则项
        return np.sqrt(np.sum(inner)) / (2 * m) + reg

    def training(self, features, target):
        """
        使用批量梯度下降算法
        :param features: 特征集m*n, m=样本数, n=特征数
        :param target: 目标集m*1
        :return: 训练好的参数theta
        """
        m, n = features.shape
        target = target.reshape(m, 1)
        features = np.insert(features, 0, 1, axis=1)  # 特征集增加一列x0,且令x0=1,以便于矩阵运算
        self.theta = np.zeros((1, n + 1))  # 初始化参数theta

        for _ in range(self.iters):  # 利用矩阵运算,一次性计算梯度
            error = np.dot(features, self.theta.T) - target  # 误差
            grad = np.dot(error.T, features) / m + self.lamb / m * self.theta  # 计算梯度
            grad[0, 0] = np.sum(error) / m  # 上一步对所有theta都进行了正则化,这一步重新计算theta0的梯度，以取消正则化
            self.theta -= self.learning_rate * grad  # 更新theta
            self.cost.append(self.cal_cost(features, target, self.theta, self.lamb))
        return

    def predict(self, features):
        """
        输入待预测样本,输出预测结果
        :param features: 待预测样本m*n
        :return: 预测值
        """
        features = np.array(features)
        return self.theta[0, 0] + np.dot(features, self.theta[0, 1:].T)


def test():
    """使用datasets生成的回归数据测试,输出不同阶段的损失值"""
    features, target = datasets.make_regression()
    lr = LinearRegression()
    lr.training(features, target)
    print(lr.cost[:: 1000])


test()
