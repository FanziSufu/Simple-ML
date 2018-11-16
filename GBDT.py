import numpy as np
from sklearn import datasets


class GBDT:
    def __init__(self, loss='MSE', alpha=0.1, n_learners=10):
        self.loss = loss  # 损失函数
        self.alpha = alpha  # 学习率
        self.n_learners = n_learners  # 基学习器的个数

    @staticmethod
    def cal_se(dataset):
        """计算数据集的方差"""
        return np.var(dataset[:, -1]) * dataset.shape[0] if dataset.shape[0] > 0 else 0

    @staticmethod
    def split_data(dataset, feature, value):
        """切分数据集.本例只考虑连续型数据"""
        left = dataset[dataset[:, feature] <= value, :]
        right = dataset[dataset[:, feature] > value, :]
        return left, right

    def create_tree(self, dataset):
        """生成单层CART回归树"""
        min_error = np.inf
        best_feature, best_value = -1, 0
        num_steps = 10  # 每次选取10个值作为划分点
        for feature in range(dataset.shape[1] - 1):
            range_min = dataset[:, feature].min()
            range_max = dataset[:, feature].max()
            step = (range_max - range_min) / num_steps
            for i in range(num_steps):
                value = range_min + float(i) * step
                left, right = self.split_data(dataset, feature, value)
                error = self.cal_se(left) + self.cal_se(right)
                if error < min_error:
                    min_error = error
                    best_feature = feature
                    best_value = value
        tree = dict()
        tree['Feature'] = best_feature
        tree['value'] = best_value
        return tree

    def training(self, dataset):  # 训练模型
        dataset = np.array(dataset)
        features = dataset[:, : -1]
        target = dataset[:, -1:]
        learner_arr = []  # 存储基学习器
        cost = []  # 存储损失
        if self.loss == 'MSE':
            fx = np.ones((dataset.shape[0], 1)) * np.mean(target)  # 预测值
            learner_arr.append(np.mean(target))  # 先把初始值加入到列表
            for i in range(self.n_learners):
                newy = target - fx  # 平方误差损失的负梯度,作为此轮基学习器的残差
                new_data = np.concatenate((features, newy), axis=1)
                tree = self.create_tree(new_data)
                left, right = self.split_data(new_data, tree['Feature'], tree['value'])
                tree['left'] = np.mean(left[:, -1])  # 叶子结点的更新要根据所选择的损失函数.本例的损失函数是平方误差,叶子结点的值即为样本均值
                tree['right'] = np.mean(right[:, -1])
                learner_arr.append(tree)

                newy[new_data[:, tree['Feature']] <= tree['value'], 0] = tree['left']  # 更新此轮基学习器的预测值
                newy[new_data[:, tree['Feature']] > tree['value'], 0] = tree['right']
                fx += self.alpha * newy  # 更新所有学习器的预测值
                cost.append(np.sqrt(np.sum(np.power(target - fx, 2))) / (2 * target.shape[0]))  # 记录均方根误差
        return learner_arr, cost

    def predict(self, learner_arr, features):
        features = np.array(features)
        if features.ndim == 1:  # 如果是1维的单个样本,则转换成二维的
            features = features.reshape(1, features.shape[0])
        if self.loss == 'MSE':
            fx = np.ones((features.shape[0], 1)) * learner_arr[0]
            for tree in learner_arr[1:]:
                pred = np.zeros((features.shape[0], 1))
                pred[features[:, tree['Feature']] <= tree['value'], 0] = tree['left']
                pred[features[:, tree['Feature']] > tree['value'], 0] = tree['right']
                fx += self.alpha * pred
            return fx


def test():
    dataset = datasets.make_regression()  # 生成随机回归数据集
    features = dataset[0]
    target = dataset[1].reshape(features.shape[0], 1)
    dataset = np.concatenate((features, target), axis=1)

    gbdt = GBDT(alpha=1, n_learners=100)
    learner_arr, cost = gbdt.training(dataset)
    print(cost[:: 10])


test()
