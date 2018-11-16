import numpy as np
from sklearn import preprocessing
from sklearn import datasets


class LogisticRegression:
    def __init__(self, learning_rate=0.1, lamb=0.001, iters=1000, kernel='sigmoid'):
        self.learning_rate = learning_rate  # 学习速率,默认值为1
        self.lamb = lamb  # 正则化参数,默认值为1
        self.iters = iters  # 最大迭代次数
        self.kernel = kernel  # 内核函数,sigmoid或者softmax
        self.theta = np.zeros((1, 1))  # 声明参数是二维array格式
        self.cost = []  # 记录损失值

    @staticmethod
    def sigmoid(features, theta):
        """
        逻辑函数,用于二元分类,数据集均为np.array格式
        :param features: 特征集m*(n+1),m为样本数,n为特征数
        :param theta: 参数集k*(n+1),k为标签的类别数，n为特征数
        :return: 函数计算结果
        """
        inner = np.dot(features, theta.T)  # 计算内核
        return 1 / (1 + np.exp(-inner))

    @staticmethod
    def softmax(features, theta):
        """
        softmax函数,用于多元分类,数据集均为np.array格式
        :param features: 特征集m*(n+1),m为样本数,n为特征数
        :param theta: 参数集k*(n+1),k为标签的类别数，n为特征数
        :return: 函数计算结果
        """
        inner = features.dot(theta.T)
        return np.exp(inner) / np.sum(np.exp(inner), axis=1, keepdims=True)  # inner的格式为m*k,如此设置np.sum的参数,可使按行相加后的结果m*1

    def cal_cost(self, features, target, theta, lamb):
        """
        计算损失函数(对数损失)，使用L2正则化
        :param features: 特征集m*(n+1),m为样本数,n为特征数
        :param target: 目标集m*k,k为类别数
        :param theta: 参数集k*(n+1),k为标签的类别数，n为特征数
        :param lamb: 正则化参数,默认值为1
        :return: 对数损失
        """
        m = features.shape[0]  # 样本数
        if self.kernel == 'sigmoid':
            inner = self.sigmoid(features, theta)  # softmax和sigmoid的损失函数格式上一致
        else:
            inner = self.softmax(features, theta)
        first = np.multiply(-target, np.log(inner))  # 前半部分
        second = np.multiply((1 - target), np.log(1 - inner))  # 后半部分
        reg = lamb / (2 * m) * np.sum(np.power(theta[:, 1:], 2))  # 正则化
        return np.sum(first - second) / m + reg

    def training(self, features, target):
        """
        使用批量梯度下降算法优化
        :param features: 特征集m*n,m为样本数,n为特征数
        :param target: 目标集m*k,k为类别数
        :return: 更新参数和损失值,无返回
        """
        features = np.insert(features, 0, 1, axis=1)  # 特征集增加一列x0,且令x0=1,以便于矩阵运算
        m, n = features.shape
        k = target.shape[1]  # 目标类别数
        self.theta = np.zeros((k, n))  # 此时n=特征数+1
        for _ in range(self.iters):  # 梯度下降
            if self.kernel == 'sigmoid':
                inner = self.sigmoid(features, self.theta)
            else:
                inner = self.softmax(features, self.theta)

            error = inner - target  # 误差
            grad = error.T.dot(features) / m + self.lamb / m * self.theta  # 计算梯度
            grad[:, 0] = np.sum(error, axis=0) / m  # 上一步对所有theta都进行了正则化，这一步重新计算theta0的梯度，以取消正则化
            self.theta -= self.learning_rate * grad  # 更新theta
            self.cost.append(self.cal_cost(features, target, self.theta, self.lamb))  # 添加当前损失值
        return

    def predict(self, features, threshold=0.5):
        """
        根据输入特征集和参数theta,输出预测值
        :param features: 待测样本1*n,n为特征数
        :param threshold： 阀值,默认值为0.5,大于0.5输出正类别,反之负类别.仅当kernel=sigmoid时使用
        :return: 若kernel=sigmoid,输出1或0（表示正类别或负类别）；若干kernel=softmax,输出概率最大类别的索引,m*1
        """
        features = np.insert(features, 0, 1, axis=1)
        if self.kernel == 'sigmoid':
            inner = self.sigmoid(features, self.theta)
            return [1 if i[0] >= threshold else 0 for i in inner]
        else:
            inner = self.softmax(features, self.theta)
            return np.argmax(inner, axis=1)  # 概率最大类别的索引


def test_sigmoid():  # 使用sklearn生成的双类别数据测试sigmoid
    features, target = datasets.make_classification(n_samples=300)
    target = target.reshape(target.shape[0], 1)

    lr = LogisticRegression()
    lr.training(features, target)
    predict = lr.predict(features)  # 获取最大预测索引
    correct = [0 if a ^ b else 1 for a, b in zip(predict, target)]
    accuracy = correct.count(1) / len(correct)  # 计算准确度
    print('accuracy={}%'.format(accuracy * 100))


def test_softmax():
    """使用鸢尾花进行测试的时候,可以做到93%的预测准确率
    使用sklearn生成的多类别数据测试softmax,当类别增多时,准确率迅速下降,原因可能是生成的数据不存在良好的线性关系
    """
    # features, target = datasets.make_classification(n_samples=5000, n_informative=4, n_classes=5)
    dataset = datasets.load_iris()  # 鸢尾花数据集
    features, target = dataset['data'], dataset['target']
    target = target.reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    target_train = enc.fit_transform(target).toarray()  # 对目标集独热编码

    lr = LogisticRegression(learning_rate=0.001, lamb=0, iters=5000, kernel='softmax')
    lr.training(features, target_train)
    predict = lr.predict(features)
    correct = [1 if a == b else 0 for a, b in zip(predict, target)]  # 本例中,索引值正好等于原数据
    accuracy = correct.count(1) / len(correct)
    print('accuracy={}%'.format(accuracy * 100))


test_sigmoid()
test_softmax()
