'''
regression y=ax+b
to classify a vector with 2 classes
'''
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


np.random.seed(2019)

# if x >= 50, then 1, <50, then -1
# if x >= 12, then -1, <12, then 1
# combined above, if x < 12 or >= 50, then 1, else -1
# regression a,b, make a*x+b outputs right label


def gen_data():
    x = []
    y = []
    for i in range(1000):
        # max 100 years old
        old = np.random.random()*100
        x.append([old])
        if old >= 50 or old < 12:
            y.append([1])
        else:
            y.append([-1])
    return x, y


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


def regression(data):
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
    a, b = 0, 0
    return a, b


def get_jenks_breaks(data_list, number_class):
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass

if __name__ == "__main__":
    x, y = gen_data()
    data = np.concatenate([x, y], axis=1)
    print(data[3:20, :])
    a, b = regression(data)
    evaluation(a, b)

    x = np.random.random(30)
    breaks = get_jenks_breaks(x, 5)

    for line in breaks:
        plt.plot([line for _ in range(len(x))], 'k--')

    plt.plot(x)
    plt.grid(True)
    plt.show()