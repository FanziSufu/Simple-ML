import numpy as np
from sklearn import datasets
from sklearn import preprocessing


class NeutalNetwork:
    def __init__(self, q=10, learning_rate=0.1, iters=10000, batch_size=10):
        self.q = q  # 隐藏层神经元个数
        self.learning_rate = learning_rate  # 学习率
        self.iters = iters  # 迭代次数
        self.batch_size = batch_size  # 小批量包含的样本数,即每次迭代的样本数
        self.para = []  # 存储参数组合
        self.cost = []  # 储存损失值

    @staticmethod
    def sigmoid(x):
        """激活函数sigmoid"""
        return 1 / (1 + np.exp(-x))

    def rand_parameter(self, features, target):
        """
        初始化参数,输入层到隐藏层的权值v,隐藏层的阀值gamma,隐藏层到输出层的权值w,输出层的阀值theta
        :param features: 已增加 x0=1 列的特征集m*n,m为样本数,n表示特征数（实际特征是n-1）np.matrix
        :param target: 已经过独热处理的标签集m*k,k表示类别数
        :return: 取值在(0,1)的随机参数值
        """
        n, k = features.shape[1], target.shape[1]
        v = np.mat(np.random.rand(n, self.q))  # n*q
        gamma = np.mat(np.random.rand(1, self.q))  # 1*q
        w = np.mat(np.random.rand(self.q, k))  # q * k
        theta = np.mat(np.random.rand(1, k))  # 1 * k
        return v, gamma, w, theta

    @staticmethod
    def cal_g(y_pre, y_true):
        """
        计算theta的一阶导数,并令其等于g,用于计算别的参数的导数
        :param y_pre: 预测结果m*k,m=self.batch_size
        :param y_true: 真实结果m*k
        :return: 参数theta的一阶导数m*k
        """
        return np.multiply(-y_pre, np.multiply(y_pre - y_true, 1 - y_pre))

    @staticmethod
    def cal_e(g, w, b):
        """
        计算gamma的一阶导数,并令其等于e,用于计算别的参数的导数
        :param g: theta的一阶导数m*k,m=self.batch_size
        :param w: 隐藏层到输出层的权值去q*k
        :param b: 隐藏层的输出m*q
        :return: 参数gamma的一阶导数m*q
        """
        return np.multiply(np.multiply(1 - b, b), np.sum(np.dot(g, w.T), axis=1))

    @staticmethod
    def cal_cost(y_pre, y_true):
        """
        计算损失函数值
        :param y_pre: 预测结果m*k
        :param y_true: 真实结果m*k
        :return: 损失值
        """
        return np.sum(np.power(y_pre - y_true, 2)) / 2

    def training(self, features, target):
        """
        使用随机梯度下降法训练神经网络,输出训练好的参数组合
        :param features: 特征集m*n,m为样本数,n为特征数
        :param target: 标签集m*k,k为类别数
        :return: 更新成员变量后,无返回值
        """
        features = np.insert(features, 0, 1, axis=1)  # 插入x0=1列,作为偏差的特征
        features, target = np.mat(features), np.mat(target)  # 矩阵运算更方便. 若用数组运算,会出现二维数组相乘之后变成一维的情况
        m, n, k = features.shape[0], features.shape[1], target.shape[1]
        v, gamma, w, theta = self.rand_parameter(features, target)  # 初始化参数

        for i in range(self.iters):
            rand_batch = np.random.randint(0, m, self.batch_size)  # 从所有样本中随机选择batch_size个样本,作为此次迭代样本
            feature_train, target_train = features[rand_batch, :], target[rand_batch, :]

            alpha = np.dot(feature_train, v)  # 计算隐藏层的输入
            b = self.sigmoid(alpha - gamma)  # 计算隐藏层的输出
            beta = np.dot(b, w)  # 计算输出层的输入
            target_pre = self.sigmoid(beta - theta)  # 计算输出层的输出,即预测为各类别的概率
            error = self.cal_cost(target_pre, target_train) / self.batch_size  # 计算均方误差
            self.cost.append(error)  # 记录误差

            g = self.cal_g(target_pre, target_train)  # 推导过程中的辅助公式
            e = self.cal_e(g, w, b)  # 推导过程中的辅助公式

            # 计算各参数的平均梯度（一阶导数）
            grad_theta = np.sum(g, axis=0) / self.batch_size
            grad_w = -np.dot(b.T, g) / self.batch_size
            grad_gamma = np.sum(e, axis=0) / self.batch_size
            grad_v = -np.dot(feature_train.T, e) / self.batch_size

            # 更新各参数
            theta -= self.learning_rate * grad_theta
            w -= self.learning_rate * grad_w
            gamma -= self.learning_rate * grad_gamma
            v -= self.learning_rate * grad_v
        self.para.extend([v, gamma, w, theta])  # 存储组合变量
        return

    def predict(self, features):
        """
        预测输入样本的类别
        :param features: 待测样本
        :return: 独热编码格式的类别
        """
        features = np.mat(np.insert(features, 0, 1, axis=1))
        v, gamma, w, theta = self.para  # 获取参数
        # 前向传播计算输出
        alpha = np.dot(features, v)
        b = self.sigmoid(alpha - gamma)
        beta = np.dot(b, w)
        target_pre = self.sigmoid(beta - theta)

        target_pre = np.array(target_pre)  # 转换成np.array
        argmax = np.argmax(target_pre, axis=1)  # 获取最大概率值的索引
        # 把最大概率值变成1,其他值变成0
        target_pre[:, :] = 0
        for i in range(features.shape[0]):
            target_pre[i, argmax[i]] = 1
        return target_pre


def test():
    features, target = datasets.make_classification(n_samples=5000, n_informative=5, n_classes=5)  # 3000个具有3个类别的训练样本
    target = target.reshape(features.shape[0], 1)
    enc = preprocessing.OneHotEncoder()
    target = enc.fit_transform(target).toarray()  # 对y进行独热编码

    nn = NeutalNetwork(q=5, learning_rate=0.1, iters=10000, batch_size=10)
    nn.training(features, target)
    prediction = nn.predict(features)
    correct = [1 if (a == b).all() else 0 for a, b in zip(prediction, target)]
    print(correct.count(1) / len(correct))
    print(nn.cost[::500])


test()
