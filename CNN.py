import numpy as np
from scipy.io import loadmat


class CNN:
    def __init__(self, layer1=2, learning_rate=0.1, iters=10000):
        self.layer1 = layer1  # 第一个卷积层卷积核的个数
        self.iters = iters  # 最大迭代次数
        self.learning_rate = learning_rate  # 学习率
        self.maxindex = []  # 存储池化区域最大值索引
        self.k = 0  # 池化后矩阵索引
        self.cost = []  # 损失值
        self.parameter = []  # 存储训练好的参数

    @staticmethod
    def relu(mat):  
        """定义激活函数relu,对矩阵mat的所有元素操作一遍"""
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i, j] = max(0, mat[i, j])
        return mat
    
    @staticmethod
    def sigmoid(x):
        """定义sigmoid函数"""
        return 1 / (1 + np.exp(x))

    @staticmethod
    def softmax(x):
        """定义softmax函数,x是列向量"""
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def cal_cost(y_pred, y_ture):
        """
        计算平方损失误差
        :param y_pred: 预测值k*1
        :param y_ture: 真实值
        :return: 平方误差
        """
        return np.sum(np.power(y_pred - y_ture, 2)) / 2

    @staticmethod
    def cal_conv(mat_a, mat_b, b=0, step=1, padding=0):
        """
        二维卷积
        对矩阵mat_a填充padding后,使用卷积核大小为F*F的矩阵,以步伐step进行卷积计算,返回卷积后的矩阵
        :param mat_a: 被卷积矩阵h*w
        :param mat_b: 卷积核矩阵f*f
        :param b: 卷积层偏置
        :param step: 步伐S
        :param padding: 对矩阵mat_a外围的填充0的层数
        :return: 返回卷积后的矩阵
        """
        h, w, f = mat_a.shape[0], mat_a.shape[1], mat_b.shape[0]  # 高,宽,卷积核边长（卷积核为方阵,且通常边长为奇数）
        conved_h, conved_w = (h - f + 2 * padding) // step + 1, (w - f + 2 * padding) // step + 1
        conved = np.mat(np.zeros((conved_h, conved_w)))  # 定义卷积后矩阵
        if padding:  # 填充操作
            new_a = np.mat(np.zeros((h + 2 * padding, w + 2 * padding)))
            new_a[padding: -padding, padding: - padding] = mat_a
            mat_a = new_a
        for i in range(conved_h):  # 卷积计算
            for j in range(conved_w):
                conved[i, j] = np.sum(np.multiply(mat_a[i: i + f, j: j + f], mat_b))
        return conved + b

    def pool(self, mat, f=2):
        """
        对矩阵mat,使用边长为f的矩阵池化,采用最大值池化法,返回池化后的矩阵
        :param mat: 被池化层H*w
        :param f: 过滤器的边长
        :return: 池化后的矩阵
        """
        h, w = mat.shape[0], mat.shape[1]  # 高,宽
        pooled_h, pooled_w = h // f, w // f
        pooled = np.mat(np.zeros((pooled_h, pooled_w)))
        for i in range(pooled_h):
            for j in range(pooled_w):
                temp = mat[i * f: i * f + f, j * f: j * f + f]
                self.maxindex.append(np.argmax(temp))  # 记录最大值索引
                pooled[i, j] = np.max(temp)
        return pooled

    def upsample(self, mat, f=2):
        """
        对梯度误差矩阵A,执行最大池化法的上采样,还原成输入矩阵的结构
        :param mat: 梯度误差矩阵
        :param f: 原池化区域大小
        :return: 上一层的梯度误差矩阵
        """
        pooled_h, pooled_w = mat.shape
        h, w = pooled_h * f, pooled_w * f
        origin = np.mat(np.zeros((h, w)))  # 定义原始矩阵
        for i in range(pooled_h):
            for j in range(pooled_w):
                temp = origin[i * f: i * f + f, j * f: j * f + f]
                temp[self.maxindex[self.k] // f, self.maxindex[self.k] % f] = mat[i, j]
                origin[i * f: i * f + f, j * f: j * f + f] = temp
                self.k += 1
        return origin

    @staticmethod
    def creat_conv(n_conv, f, dim3=1):
        """
        创建卷积层,初始化参数
        :param n_conv: 卷积核个数
        :param f: 卷积核边长
        :param dim3: 输入张量第三维的数值,默认为1,表示输入是一个矩阵,若为2,则表示输入了2个矩阵,以此类推
        :return: 卷积层的初始参数
        """
        weight = []
        for _ in range(n_conv):
            temp = []
            for _ in range(dim3):
                temp.append(np.mat((2 * np.random.rand(f, f) - np.ones((f, f))) / (np.sqrt(n_conv) * 100)))  # 尽量使参数值小
            weight.append(temp)
        b = np.mat((2 * np.random.rand(n_conv, 1) - np.ones((n_conv, 1))) / (np.sqrt(n_conv) * 100))
        return weight, b

    @staticmethod
    def creat_fc(n_neural, n_put):
        """
        创建全连接层,初始化参数
        :param n_put: 上一层层神经元个数
        :param n_neural: 全连接层神经元个数
        :return: 全连接层的参数
        """
        weight = np.mat((2 * np.random.rand(n_neural, n_put) - np.ones((n_neural, n_put))) / (np.sqrt(n_neural) * 100))
        b = np.mat((2 * np.random.rand(n_neural, 1) - np.ones((n_neural, 1))) / (np.sqrt(n_neural) * 100))
        return weight, b

    def training(self, features, target):
        """
        根据CNN算法训练模型,使用的结构为
        input->conv->pool->fc->output
        本例只考虑通道为1的灰度图像
        :param features: 特征集m*n,m为样本个数,n为图片像素总数,如图片为28*28,则n=784,训练时,要把一维数据重新转换成28*28的矩阵
        :param target: 标签集m*k,k为类别数量,要求y已进行过独热编码
        :return: 模型参数
        """
        m, k = features.shape[0], target.shape[1]
        features = np.mat(features)
        target = np.mat(target)
        weight2, b2 = self.creat_conv(self.layer1, 5, dim3=1)
        weight4, b4 = self.creat_fc(100, 288)
        weight5, b5 = self.creat_fc(k, 100)

        for index in range(self.iters):
            print(index)
            if index == m:
                break
            train_x = features[index, :]
            y_true = target[index, :].T
            train_x = train_x.reshape(28, 28)

            a1 = [train_x]
            a2 = []
            for i in range(self.layer1):  # 第二层卷积核个数
                temp = np.mat(np.zeros((24, 24)))
                for j in range(1):  # dim3=1
                    temp += self.cal_conv(a1[j], weight2[i][j], b=b2[i, 0])
                a2.append(self.relu(temp))
            a3 = []
            for i in range(self.layer1):
                a3.append(self.pool(a2[i]))
            a3flat = np.mat(np.zeros((1, self.layer1 * 12 * 12)))
            for i in range(self.layer1):
                a3flat[0, 144 * i: 144 * i + 144] = a3[i].flatten()
            a3 = a3flat.T
            a4 = self.sigmoid(np.dot(weight4, a3) + b4)
            a5 = self.softmax(np.dot(weight5, a4) + b5)
            self.cost.append(self.cal_cost(a5, y_true))

            delta5 = np.multiply(a5, np.multiply(a5 - y_true, 1 - a5))  # 计算第五层delta和梯度
            grad_weight5 = np.dot(delta5, a4.T)
            grad_b5 = delta5

            delta4 = np.multiply(np.dot(weight5.T, delta5), np.multiply(a4, 1 - a4))  # 计算第四层delta和梯度
            grad_weight4 = np.dot(delta4, a3.T)
            grad_b4 = delta4

            delta3 = np.dot(weight4.T, delta4)[::-1]  # 计算第三层delta和梯度
            delta3_ = []
            for i in range(self.layer1):
                temp = delta3[i * 144: i * 144 + 144, 0].reshape(12, 12)
                delta3_.append(temp)
            delta3 = delta3_

            delta2 = []  # 计算第二层delta和梯度
            for i in range(self.layer1):
                delta2.append(self.upsample(delta3[i]))

            grad_weight2 = []
            grad_b2 = np.mat(np.zeros((self.layer1, 1)))
            for i in range(self.layer1):
                temp = []
                for j in range(1):
                    temp.append(self.cal_conv(a1[j], delta2[i]))
                grad_b2[i, 0] = np.sum(delta2[i])
                grad_weight2.append(temp)

            # 更新参数值
            weight5 -= self.learning_rate * grad_weight5
            b5 -= self.learning_rate * grad_b5
            weight4 -= self.learning_rate * grad_weight4
            b4 -= self.learning_rate * grad_b4
            for i in range(self.layer1):
                for j in range(1):
                    weight2[i][j] -= self.learning_rate * grad_weight2[i][j]
            b2 -= self.learning_rate * grad_b2

            self.maxindex = []  # 下一轮迭代器重置
            self.k = 0

        self.parameter.extend([weight2, b2, weight4, b4, weight5, b5])  # 保存参数
        return

    def predict(self, features):
        features = np.mat(features)
        preiction = []
        m = features.shape[0]
        for index in range(m):
            x = features[index, :]
            x = x.reshape(28, 28)
            weight2, b2, weight4, b4, weight5, b5 = self.parameter
            # 前向传播求输出
            a1 = [x]
            a2 = []
            for i in range(self.layer1):  # 第二层卷积核个数
                temp = np.mat(np.zeros((24, 24)))
                for j in range(1):  # dim3=1
                    temp += self.cal_conv(a1[j], weight2[i][j], b=b2[i, 0])
                a2.append(self.relu(temp))
            a3 = []
            for i in range(self.layer1):
                a3.append(self.pool(a2[i]))
            a3flat = np.mat(np.zeros((1, self.layer1 * 12 * 12)))
            for i in range(self.layer1):
                a3flat[0, 144 * i: 144 * i + 144] = a3[i].flatten()
            a3 = a3flat.T
            a4 = self.sigmoid(np.dot(weight4, a3) + b4)
            a5 = self.softmax(np.dot(weight5, a4) + b5)

            # 本例中最大值索引即为对应的数字
            max_index = np.argmax(a5)
            preiction.append(max_index)

        return preiction


def test():
    """使用手写数字集进行测试,效果不理想,不知道问题出在哪"""
    dataset = loadmat('data/mnist_all')
    train_features = np.zeros((1, 784))
    train_target = np.zeros((1, 10))
    valid_features = np.zeros((1, 784))
    valid_target = []
    for i in range(10):  # 每个数字获取1000个训练数据和50个验证数据
        temp_train = dataset['train%d' % i]
        temp_valid = dataset['test%d' % i]
        rand_index = np.random.choice(temp_train.shape[0], 1000)  # 随机选取1000个训练样本
        valid_index = np.random.choice(temp_valid.shape[0], 50)
        temp_train = temp_train[rand_index, :]
        temp_valid = temp_valid[valid_index, :]
        train_features = np.concatenate((train_features, temp_train))
        valid_features = np.concatenate((valid_features, temp_valid))
        target = np.zeros((1000, 10))
        target[:, i] = 1
        train_target = np.concatenate((train_target, target))
        valid_target.extend([i] * 50)

    train_features = train_features[1:, :]
    train_target = train_target[1:, :]
    valid_features = valid_features[1:, :]
    train_data = np.concatenate((train_features, train_target), axis=1)
    np.random.shuffle(train_data)  # 洗牌
    train_features = train_data[:, : 784]
    train_target = train_data[:, 784:]

    cnn = CNN(learning_rate=0.1, iters=1000)
    cnn.training(train_features, train_target)
    prediction = cnn.predict(valid_features)
    correct = [1 if a == b else 0 for a, b in zip(prediction, valid_target)]
    print(correct.count(1) / len(correct))
    cost = cnn.cost
    print(cost[::100])


test()
