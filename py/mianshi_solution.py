"""
Analysis:

问题可以简化为如下描述：

     1      x1           -1      x2       1                 -1             1
    --------|--------------------|-----------------|-----------------|----------------
            12                   50                80                120
即：一维的线性不可分问题。
如果问题退化为最简单的形式：

     1       x1           -1     
    --------|--------------------
            50                
那么可以用一个简单的y=w*x+b感知机回归出一个感知元进行二分类。下面简单的代码尝试了最简单的感知机回归分类.
如果生成gen_data2()的时候，可以看到我们的精简版感知器在大概20个迭代的时候误差就为零了。最终输出的结果是
大于50就为1，小于50就为-1. 可以比较清晰的得到w和b。

如果增加一下breaks，使得数据本身在一维上不可分。此时是无法通过单一线性函数w*x+b区分的，但是可以统一化的
通过一些系列的：
[(w1, b1), (w2, b2), (w3, b3)]来进行区分....
尽管如此，由于水平有限，依旧不知道如何用一个函数来表达这个关系，尝试了几个思路，无疾而终：

1. 通过学习这个参数(x-a)/|x-a| * x + b, 也就是a和b，来拟合存在两个breaks的情况，即通过学习一个参数使得第一个参数
可以通过x自动判断取正还是取负，但如果是N个breaks就不知道如何处理了..
2. 假设数据是如我上面画的那样，那么应该是可以先把这些breaks点找到的，这样两个相邻的breaks构成的两个数据区域的类别肯定是不同
的，可以通过类似于作差然后求异或等方式判断类别。
3. 直接看成了非线性问题.... 有多少个灭点就用多少阶去拟合... 但似乎与题意不符..

(可能领会了意思，但还是没有解决核心问题，时间仓促，只能暂时实现仅存在一个灭点的情况,
 是一个有趣值得以后慢慢深思的问题)

"""
from random import choice
from numpy import array, dot, random
import numpy as np

np.random.seed(345)


def unit_step(x): return -1 if x < 0 else 1


class Perceptron(object):
    """
    定义一个最简单的感知器，它的输入为1维度或者多个维度的x，y是类别，当然目前仅仅支持二分类。
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        # 初始化权重，check一下x和y尺寸
        print(x.shape)
        print(y)
        self.w_ = np.zeros(1 + x.shape[1])
        # 初始化误差列表
        self.errors_ = []
        # 开始迭代
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                # 计算预测与实际值之间的误差在乘以学习率
                update = self.eta * (target - self.predict(xi))
                # 更新权重
                self.w_[1:] += update * xi
                # 更新W0
                self.w_[0] += update * 1
                # 当预测值与实际值之间误差为0的时候,errors=0否则errors=1
                errors += int(update != 0)
                print('iter: {}, erro: {}'.format(i, errors))
            # 将错误数据的下标加入到列表中
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        # 即 sum(i*j for i,j in zip(x,self.w_[1:]))
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        # 当self.net_input(x) >= 0.0 成立时返回1，否则返回-1
        return np.where(self.net_input(x) >= 0.0, 1, -1)


def gen_data():
    data = []
    for i in range(9000):
        # max 100 years old
        old = np.random.random()*100
        if old >= 50 or old < 12:
            data.append((old, 1))
        else:
            data.append((old, -1))
    return data


def gen_data2():
    data = []
    for i in range(9000):
        # max 100 years old
        old = np.random.random()*100
        if old >= 50:
            data.append((old, 1))
        else:
            data.append((old, -1))
    return data


def evaluation(a, b):
    gt = []
    preds = []
    for i in range(100):
        # max 100 years old
        old = np.random.random()*100
        if old >= 50 or old < 12:
            gt.append(1)
        else:
            gt.append(-1)
        pred = a*old + b
        if pred > 0:
            preds.append(1)
        else:
            preds.append(-1)
    erro = np.array(gt) - np.array(preds)
    print('precision: {}'.format(1 - np.count_nonzero(erro)/len(erro)))


if __name__ == "__main__":
    """
    regression a and b on data
        1   x1           -1      x2     1
    --------|--------------------|------------
            12                   50
    a single a*x+b can not divid then properly 
    since the class is not linear-dividable
    to do this, regression 2 values:
    (x-a)/|x-a| * x + b
    """

    training_data2 = gen_data2()
    # training_data2 = gen_data()
    data = np.array(training_data2)
    print(data)
    p = Perceptron(n_iter=150)
    p.fit(np.expand_dims(data[:, 0], axis=-1), data[:, 1])

    for d in training_data2[-20:-1]:
        res = p.predict(d[0])
        print("input: {} out: {} -> class: {}".format(d[0], res, unit_step(res)))
    evaluation(p.w_[1:], p.w_[0])
