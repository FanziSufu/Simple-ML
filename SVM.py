import numpy as np
from sklearn import datasets


class SVM:
    def __init__(self, c=1, toler=0.001, iters=500, kernel_option=('', 1)):
        self.C = c  # 正则化参数
        self.toler = toler  # 容错率
        self.maxIter = iters  # 最大迭代次数
        self.kernel_opt = kernel_option  # 选择的核函数
        # 声明在self.training中需要用到的变量
        self.features = np.mat(np.zeros((1, 1)))
        self.target = np.mat(np.zeros((1, 1)))
        self.m = self.features.shape[0]
        self.alpha = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.Ecache = np.mat(np.zeros((self.m, 2)))
        self.kernel_matrix = np.mat(np.zeros((1, 1)))

    @staticmethod
    def cal_kernel_value(features, features_i, kernel_option):
        """
        计算所有样本跟第i个样本的核函数的值,返回m*1的矩阵. 使用np的矩阵运算, 格式都为np.matrix
        :param features: 特征集m*n,m为样本数,n为特征数
        :param features_i: 第i个样本1*n
        :param kernel_option: 所选择的核函数(高斯核函数或线性核函数)
        :return: 第i个样本与所有样本的核函数的值,m*1
        """
        m = features.shape[0]
        kernel_value = np.mat(np.zeros((m, 1)))  # 定义返回矩阵
        if kernel_option[0] == 'rbf':  # 计算高斯核函数
            sigma = kernel_option[1]  # 高斯函数的参数值，默认为1
            for i in range(m):  # 根据高斯函数的公式,逐个更新
                diff = features[i, :] - features_i
                kernel_value[i] = np.exp(np.dot(diff, diff.T) / (-2 * sigma**2))
        else:  # 计算线性核函数
            kernel_value = np.dot(features, features_i.T)  # 一次性更新
        return kernel_value

    def cal_kernel(self, features, kernel_option):
        """
        计算所有样本相互之间的核函数的值.返回一个m*m的矩阵M,第i个样本和第j个样本的值=M[i,j]=M[j,i],0<=i,j<m
        :param features: 特征集m*n,m为样本数,n为特征数
        :param kernel_option: 所选择的核函数(高斯核函数或线性核函数)
        :return: m*m的矩阵M,储存个样本间的核函数的值
        """
        m = features.shape[0]
        kernel_matrix = np.mat(np.zeros((m, m)))  # 定义返回矩阵
        for i in range(m):  # 逐列更新
            kernel_matrix[:, i] = self.cal_kernel_value(features, features[i, :], kernel_option)
        return kernel_matrix

    def j_rand(self, i):
        """获取与i不同的随机的索引值"""
        j = i
        while j == i:
            j = np.random.randint(0, self.m)
        return j

    @staticmethod
    def clip_alpha(alpha, l, h):
        """裁剪alpha"""
        if alpha > h:
            return h
        if alpha < l:
            return l
        return alpha

    def cal_e(self, i):
        """计算Ei=h（xi）-yi"""
        hxi = float(np.dot(self.kernel_matrix[i, :], np.multiply(self.alpha, self.target)) + self.b)
        return hxi - float(self.target[i])

    def update_e(self, i):
        """更新Ecache中Ei的值和标识"""
        ei = self.cal_e(i)
        self.Ecache[i] = [1, ei]
        return

    def select_second_alpha(self, i, ei):
        """选取第二个alpha变量"""
        j, ej, maxsteps = 0, 0, 0
        self.Ecache[i] = [1, ei]  # 更新Ecache
        valid_e = np.nonzero(self.Ecache[:, 0])[0]  # 获取所有更新过值的E的索引
        if len(valid_e) > 1:
            for k in valid_e:
                if k == i:
                    continue
                ek = self.cal_e(k)
                delta_e = abs(ek - ei)
                if delta_e > maxsteps:  # 获取abs(ej-Ei)最大的ej,这样可以加快迭代的幅度,以尽快抵达终点
                    j = k
                    ej = ek
                    maxsteps = delta_e
        else:  # 第一次遍历,随机选取一个不同与i的索引
            j = self.j_rand(i)
            ej = self.cal_e(j)
        return j, ej

    def inner_loop(self, i):
        """内部循环,判断i样本是否满足KKT条件,满足则返回0,若不满足,则更新alpha[i]和alpha[j],更新成功返回1,更新失败返回0"""
        ei = self.cal_e(i)
        r = self.target[i] * ei  # 拆开来，等价于 y(wx+b)-1
        # 如果没有容错率，则分别是 r<0 和 r>0. 容错率的意思是，在（-toler,toler）之间的点，就当做是满足KKT条件了，而放过不做优化
        if (self.alpha[i] < self.C and r < -self.toler) or (self.alpha[i] > 0 and r > self.toler):
            j, ej = self.select_second_alpha(i, ei)  # 选取第二个alpha变量
            alpha_i_old = self.alpha[i].copy()  # 定义改变前的alpha
            alpha_j_old = self.alpha[j].copy()

            # 根据0<=alpha<=C,计算alpha的上下边界
            if self.target[i] == self.target[j]:  # 分目标值相等或不等两种情况
                low = max(0, alpha_i_old + alpha_j_old - self.C)
                high = min(self.C, alpha_i_old + alpha_j_old)
            else:
                low = max(0, alpha_j_old - alpha_i_old)
                high = min(self.C, self.C + alpha_j_old - alpha_i_old)
            if low == high:  # 这种情况,意味着alpha不会再改变,直接返回0
                return 0

            eta = self.kernel_matrix[i, i] + self.kernel_matrix[j, j] - 2 * self.kernel_matrix[i, j]  # alphaj的二阶导
            if eta <= 0:  # eta是alphaj的二阶导数.根据二阶导数性质,只有当二阶导数>0时,原函数才能取到最小值. 所以小于等于0时，直接返回0
                return 0

            alpha_j_new = alpha_j_old + self.target[j] * (ei - ej) / eta  # 根据推导过程中的公式,计算新的alphaj
            alpha_j_new = self.clip_alpha(alpha_j_new, low, high)  # 裁剪alphaj
            if abs(alpha_j_new - alpha_j_old) < 0.00001:  # 如果变化量太小,也视为没有改变,返回0
                return 0

            # 根据推导过程中的公式,计算新的alphai
            alpha_i_new = alpha_i_old + self.target[i] * self.target[j] * (alpha_j_old - alpha_j_new)
            bi = float(-ei + self.target[i] * self.kernel_matrix[i, i] * (alpha_i_old - alpha_i_new) +
                       self.target[j] * self.kernel_matrix[i, j] * (alpha_j_old - alpha_j_new) + self.b)
            bj = float(-ej + self.target[i] * self.kernel_matrix[i, j] * (alpha_i_old - alpha_i_new) +
                       self.target[j] * self.kernel_matrix[j, j] * (alpha_j_old - alpha_j_new) + self.b)
            if 0 < alpha_i_new < self.C:  # 如果alpha_i_new是支持向量,那么根据公式,此时bi=b
                self.b = bi
            elif 0 < alpha_j_new < self.C:  # 同理
                self.b = bj
            else:  # 如果都不是支持向量,取均值
                self.b = (bi + bj) / 2

            self.alpha[i] = alpha_i_new  # 更新alphai
            self.alpha[j] = alpha_j_new
            self.update_e(i)  # 更新Ecache
            self.update_e(j)
            return 1
        return 0

    def training(self, features, target):
        """
        训练模型,得到拉格朗日乘子alpha和偏差b.使用矩阵计算.
        :param features: 特征集m*n,m为样本数,n为特征数
        :param target: 目标集m*1
        :return: 无返回值,通过更新类变量来得到训练好的模型参数
        """
        self.features = np.mat(features)  # 特征集
        self.target = np.mat(target)  # 目标集
        self.m = self.features.shape[0]  # 样本数
        self.alpha = np.mat(np.zeros((self.m, 1)))  # 拉格朗日乘子
        self.b = 0  # 偏差
        self.Ecache = np.mat(np.zeros((self.m, 2)))  # 存储E的矩阵.E为预测值与目标值的差,Ei=h（xi）-yi.
        # 分2列,第一列作为标记（默认为0,如果更新过,则设值为1）,第二列存储值
        self.kernel_matrix = self.cal_kernel(self.features, self.kernel_opt)  # 计算并储存核函数矩阵

        switch = True  # 用于控制全遍历或局部遍历的开关,局部遍历指,只遍历支持向量.True全遍历,False局部遍历
        alpha_changed = 0  # 用于记录所有alpha在本次迭代中,是否有所改变,若值为0,说明都无改变,若值大于0,说明存在改变,执行下一次迭代
        iters = 0  # 用于记录迭代次数

        while iters < self.maxIter and (alpha_changed > 0 or switch):  # 当迭代轮次超过最大值或者 遍历全集后alpha值无变化， 则跳出外循环，训练结束
            alpha_changed = 0  # 每次迭代，重置为0
            if switch:  # 全遍历,验证每个样本
                for i in range(self.m):
                    alpha_changed += self.inner_loop(i)  # innerL返回0或1,分别表示无变化和有变化.只要有一个alpha发生过变化,则认为整个alpha集发生变化
                iters += 1  # 一次更新完毕，迭代次数+1
            else:  # 全遍历后,再遍历所有支持向量,直到所有支持向量的alpha无变化,再进行全遍历.如果此次全遍历,整个alpha集都无变化,则训练结束,否则再次遍历支持向量,如此循环.
                bound_alpha = [i for i, a in enumerate(self.alpha) if 0 < a < self.C]  # 获取所有支持向量的索引
                for i in bound_alpha:
                    alpha_changed += self.inner_loop(i)
                iters += 1

            if switch:  # 全遍历后,进入支持向量的遍历
                switch = False
            elif alpha_changed == 0:  # 支持向量遍历后,如果所有支持向量的alpha无变化,则进行全遍历
                switch = True
        print('Total Iter:', iters)
        return

    def predict(self, features):
        """
        先找出alpha>0的索引,得到相应的训练样本（根据公式,只有alpha>0的样本才对预测结果产生影响）,再使用这些训练样本与待测样本进行核函数计算,得到核函数矩阵,最后利用h(xi)的公式算出结果
        注意,利用alpha的值计算出权重w,再用h(xi)=wx+b的公式计算预测结果的方式,只针对线性核函数有效.而不能用于其他核函数.
        """
        features = np.mat(features)
        alpha_nonzero = np.nonzero(self.alpha)[0]  # 获取非零alpha的索引
        uesful_alpha = self.alpha[alpha_nonzero, :]  # 截取非零alpha,以下同理
        useful_features = self.features[alpha_nonzero, :]
        useful_target = self.target[alpha_nonzero, :]

        p, q = features.shape[0], useful_features.shape[0]  # p,q分别为待测样本数,和选取的训练样本数
        kernel_mat = np.mat(np.zeros((p, q)))  # 定义待测样本与训练样本的核函数矩阵
        for i in range(p):  # 更新核函数矩阵的值
            kernel_mat[i, :] = self.cal_kernel_value(useful_features, features[i, :], self.kernel_opt).T
        pred = np.dot(kernel_mat, np.multiply(uesful_alpha, useful_target)) + self.b  # 根据公式计算预测结果
        return [1 if x >= 0 else -1 for x in pred]

    def accuracy(self, features, target):
        """
        计算预测准确率
        :param features: 待测样本p*n,p为样本数,n为特征数
        :param target: 待测样本的标签值p*1
        :return: 预测准确率
        """
        predictions = self.predict(features)  # 计算预测值
        correct = [1 if a == b else 0 for a, b in zip(predictions, target)]  # 预测值与原值相等则为1,否则0
        return correct.count(1) / len(correct)


def test():  # 使用sklearn生成的二分类集分别测试线性核函数和高斯核函数的表现情况
    features, target = datasets.make_classification(n_samples=300)
    target = np.array([1 if x == 1 else -1 for x in target]).reshape((-1, 1))

    svm1 = SVM(kernel_option=('', 0))  # 线性核函数
    svm2 = SVM(kernel_option=('rbf', 1))  # 高斯核函数
    svm1.training(features, target)
    svm2.training(features, target)
    print(svm1.accuracy(features, target))
    print(svm2.accuracy(features, target))


test()
