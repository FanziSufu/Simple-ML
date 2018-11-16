import numpy as np
from collections import Counter
from sklearn import datasets


class RandomForest:
    def __init__(self, k=1, t=10):
        self.k = k  # 每次随机选取k个特征,从中选择最优特征和特征值
        self.T = t  # 决策树的数量
        self.validData = {}  # 存储每颗树未使用的样本索引
        self.accuracy = 0  # 初始化包外数据准确率
    
    @staticmethod
    def cal_gini(dataset):
        """计算基尼指数"""
        gini = 1
        labels = Counter(dataset[:, -1].tolist())
        for amount in labels.values():
            prob = amount / dataset.shape[0]
            gini -= np.power(prob, 2)
        return gini

    @staticmethod
    def split_dataset(dataset, feature, value):
        """分离数据集.指针对离散型数据集"""
        left = dataset[np.nonzero(dataset[:, feature] == value)[0], :]
        right = dataset[np.nonzero(dataset[:, feature] != value)[0], :]
        return left, right

    @staticmethod
    def choose_dataset(dataset):
        """从原始训练集中,使用自助采样法获取训练集,与未被选上的样本索引（作为包外验证集）"""
        m = dataset.shape[0]
        choosed_feature = np.random.choice(m, m)
        unchoosed_feature = set(range(m)) - set(choosed_feature.tolist())
        train_data = dataset[choosed_feature, :]
        return train_data, unchoosed_feature   
    
    def choose_best_feature(self, dataset):
        """从k个特征中选择最优特征和特征值"""
        best_feature, min_gini, best_value, split_gini = -1, np.inf, 0, 0  # 定义各种变量
        n = dataset.shape[1] - 1  # 特征数
        rand_feature = np.random.choice(n, self.k, replace=False)  # 从range(n)中选择k个不重复的值
        for feature in rand_feature:  # 遍历每个特征的每个特征值
            values = np.unique(dataset[:, feature])
            for value in values:
                left, right = self.split_dataset(dataset, feature, value)
                split_gini = left.shape[0] / dataset.shape[0] * self.cal_gini(left) + right.shape[0] / \
                             dataset.shape[0] * self.cal_gini(right)
                if split_gini < min_gini:
                    min_gini = split_gini
                    best_feature = feature
                    best_value = value
        return best_feature, best_value

    def create_tree(self, dataset):
        """创建CART分类树"""
        if dataset.shape[0] == 0:  # 如果数据集为空,返回空
            return
        if np.unique(dataset[:, -1]).shape[0] == 1: 
            return dataset[0, -1]
        best_feature, best_value = self.choose_best_feature(dataset)
        tree = dict()
        tree['Feature'] = best_feature
        tree['Value'] = best_value
        left, right = self.split_dataset(dataset, best_feature, best_value)
        tree['left'] = self.create_tree(left)
        tree['right'] = self.create_tree(right)
        return tree

    def predict_bytree(self, tree, test_data):  # 返回单棵决策树预测的结果
        if not isinstance(tree, dict): 
            return tree
        feature = tree['Feature']
        value = tree['Value']
        if test_data[feature] == value:
            return self.predict_bytree(tree['left'], test_data)
        else: 
            return self.predict_bytree(tree['right'], test_data)

    def training(self, dataset):
        """训练生成T棵决策树,组成随机森林"""
        dataset = np.array(dataset)
        rand_forest = []

        for i in range(self.T):
            train_data, unchoosed_feature = self.choose_dataset(dataset)
            tree = self.create_tree(train_data)
            rand_forest.append(tree)
            self.validData[i] = unchoosed_feature  # 记录各树未使用的样本

        # 以下计算包外数据的准确率
        totol_count = count = 0  # 分别记录预测数,和预测正确的个数
        for i, test_data in enumerate(dataset[:, : -1]):
            temp = []
            for tree_index, validData in self.validData.items():
                if i in validData:  # 如果i样本未被tree_index使用,则计算它的预测值
                    temp.append(self.predict_bytree(rand_forest[tree_index], test_data))
            if temp:
                totol_count += 1
                if Counter(temp).most_common(1)[0][0] == dataset[i, -1]:  # 投票法获取预测结果
                    count += 1  
        self.accuracy = count / totol_count  # 更新包外数据准确率

        return rand_forest

    def predict(self, rand_forest, test_data):
        """使用随机森林预测"""
        test_data = np.array(test_data)
        prediction = []
        for data in test_data:
            temp = []
            if isinstance(data, np.ndarray):  # 判断test_data是二维还是一维的
                for tree in rand_forest:
                    temp.append(self.predict_bytree(tree, data))
                prediction.append(Counter(temp).most_common(1)[0][0])
            else:
                for tree in rand_forest:
                    temp.append(self.predict_bytree(tree, test_data))
                prediction.append(Counter(temp).most_common(1)[0][0])
                break
        return prediction


def test():
    """使用鸢尾花数据集测试"""
    data = datasets.load_iris()
    features = data['data']
    target = data['target'].reshape(150, 1)
    data = np.concatenate((features, target), axis=1)
    np.random.shuffle(data)  # 打乱数据
    features = data[:, : -1]
    target = data[:, -1]
    rf = RandomForest(k=3, t=10)
    forest = rf.training(data)
    print(rf.accuracy)  # 检查包外数据准确率
    prediction = rf.predict(forest, features)
    correct = [1 if a == b else 0 for a, b in zip(prediction, target)]
    print(correct.count(1) / 150)  # 训练数据准确率


test()
