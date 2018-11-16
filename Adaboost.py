import numpy as np
from sklearn import datasets


class Adaboost:
    def __init__(self, n_learners=10):
        self.n_learners = n_learners  # 弱学习器的最大个数

    @staticmethod
    def cal_gm(features, feature, value, ineq):
        """
        以决策树桩作为基学习器,该基学习器,会将指定特征值<=value的分为-1类,>value的分为1,返回分类结果
        本例中base_learner表示基学习器,gm代表该学习器的分类结果
        数据集为np.array
        :param features: 特征集m*n,连续型数据
        :param feature: 给定的特征索引
        :param value: 给定的特征值
        :param ineq: 不等号的方向,'lt' 等于 '<=', 'gt' 等于 '>'
        :return: 返回以feature为最优特征,value为最优特征值的 决策树桩的 分类结果m*1
        """
        gm = np.ones((features.shape[0], 1))
        if ineq == 'lt':  # 把小于等于特征值的样本分为-1
            gm[features[:, feature] <= value, 0] = -1.0
        else:  # 把大于特征值的样本分为-1
            gm[features[:, feature] > value, 0] = -1.0
        return gm

    @staticmethod
    def cal_error(gm, target, weight):
        """
        计算基学习器base_learner的分类误差率
        :param gm: base_learner的分类结果m*1
        :param target: 标签集m*1
        :param weight: 训练集的样本权重m*1
        :return: base_learner的分类误差率
        """
        temp = np.multiply(gm, target)  # 结果为1则预测正确,结果为-1则预测错误
        temp[temp[:, 0] == 1, 0] = 0  # 预测正确的设为0
        temp[temp[:, 0] == -1, 0] = 1  # 预测错误的设为1
        return np.dot(weight.T, temp)[0, 0]  # base_learner的分类误差率

    @staticmethod
    def cal_alpha(error):
        """计算alpha, 分母的处理是预防error=0的情况"""
        return np.log((1 - error) / max(error, 1e-16)) / 2

    @staticmethod
    def update_weight(weight, alpha, target, gm):
        """
        更新下一轮迭代的样本权值并返回
        :param weight: 待更新的样本权值D,m*1
        :param alpha: 基学习器的系数
        :param target: 标签集m*1
        :param gm: base_learner的分类结果m*1
        :return: 用于下一轮迭代的样本权值
        """
        next_weight = np.multiply(weight, np.exp(-alpha * np.multiply(target, gm)))
        return next_weight / np.sum(next_weight)

    def create_base_learner(self, features, target, weight):
        """
        创建误分类率最小的基学习器
        :param features: 特征集m*n
        :param target: 标签集m*1
        :param weight: 样本权值
        :return: 分类误差率最小的基学习器和训练时的预测结果
        """
        m, n = features.shape
        num_steps = 10  # 定义特征值取值的个数
        base_learner = {}
        min_error, best_feature, best_value, best_ineq = np.inf, -1, 0, 'lt'
        bestgm = np.zeros((m, 1))
        for feature in range(n):
            range_min = features[:, feature].min()
            range_max = features[:, feature].max()
            step = (range_max - range_min) / num_steps  # 根据该特征下特征值的最大最小值,设定步长
            for i in range(num_steps + 1):  # 根据步长选取10个值
                value = range_min + float(i) * step
                for ineq in ('lt', 'gt'):
                    gm = self.cal_gm(features, feature, value, ineq)  # 预测结果
                    error = self.cal_error(gm, target, weight)  # 误差分类率
                    if error < min_error:  # 更新最小误差分类率
                        min_error = error
                        best_feature = feature
                        best_value = value
                        best_ineq = ineq
                        bestgm = gm
        alpha = self.cal_alpha(min_error)  # 根据最小误差,计算alpha
        base_learner['Feature'] = best_feature  # 保存决策树桩的信息
        base_learner['Value'] = best_value
        base_learner['Alpha'] = alpha
        base_learner['ineq'] = best_ineq
        return base_learner, bestgm

    def training(self, features, target):
        """
        根据Adaboost算法,得出最终的线性加法模型,
        :param features: 特征集m*n
        :param target: 标签集m*1
        :return: 加法模型,基学习器组成的列表
        """
        features = np.array(features)
        target = np.array(target).reshape(features.shape[0], 1)  # 可把一维数组转化成二维的m*1
        m = features.shape[0]
        weight = np.ones((m, 1)) / m  # 初始化样本权值
        learner_arr = []  # 加法模型,存储基学习器的列表
        fx = np.zeros((m, 1))  # 预测值
        for i in range(self.n_learners):
            base_learner, gm = self.create_base_learner(features, target, weight)
            learner_arr.append(base_learner)
            fx += base_learner['Alpha'] * gm
            predicton = [1 if x >= 0 else -1 for x in fx]
            corrct = [1 if a == b else 0 for a, b in zip(predicton, target)]
            if corrct.count(1) / len(corrct) == 1:  # 对训练集分类正确率达百分百,则跳出循环
                break
            weight = self.update_weight(weight, base_learner['Alpha'], target, gm)  # 如果没达到百分百,那么继续
        return learner_arr

    def predict(self, learner_arr, features):
        """
        根据训练好的加法模型,对样本X进行分类
        :param learner_arr: 训练好的加法模型
        :param features: 待分类样本
        :return: 分类结果
        """
        features = np.array(features)
        if features.ndim == 1:  # 如果是1维的单个样本,则转换成二维的
            features = features.reshape(1, features.shape[0])
        fx = np.zeros((features.shape[0], 1))
        for base_learner in learner_arr:
            gm = self.cal_gm(features, base_learner['Feature'], base_learner['Value'], base_learner['ineq'])
            fx += base_learner['Alpha'] * gm
        return [1 if x >= 0 else -1 for x in fx]


def test():
    dataset = datasets.make_classification()
    features = dataset[0]
    target = dataset[1]
    target[target[:] == 0] = -1
    ada = Adaboost(n_learners=50)
    learner_arr = ada.training(features, target)
    predicton = ada.predict(learner_arr, features)
    correct = [1 if a == b else 0 for a, b in zip(predicton, target)]
    print(correct.count(1) / len(correct))


test()
