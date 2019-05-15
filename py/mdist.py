"""

calculate 马氏距离

still hard to understand how to calculate
"""
import numpy as np
import pandas as pd
import scipy as sp
import numpy as np


def mashi_dist2(x, y):
    """
    suppose x, y is the 2 vector to calculate
    """
    x = np.vstack([x, y])
    print('x ', x)
    # we have 10 samples, every sample has 4 dims
    xt = x.T
    print('xt ', xt)
    cov_s = np.cov(x)
    print('covariance ', cov_s)
    cov_s_invert = np.linalg.inv(cov_s)
    print('cov invert ', cov_s_invert)
    n = xt.shape[0]
    ds = []
    print(xt.shape)
    for i in range(n):
        for j in range(i+1, n):
            delta = xt[i] - xt[j]
            print('delta ', delta)
            d = np.sqrt(np.dot(np.dot(delta, cov_s_invert), delta.T))
            ds.append(d)
    print('ds ', ds)


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    print(np.mean(data))
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def mahalanobis2(data=None, cov=None):
    """
    calculate every mahala distance in every row in data
    160, 60000
    160, 70000
    170, 60000
    """
    if isinstance(data, list):
        data = np.array(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    n_samples = data.shape[0]
    ds = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            delta = data[i] - data[j]
            left_term = np.dot(delta, inv_covmat)
            d = np.dot(left_term, delta.T)
            ds.append(d)
    print(ds)
    return ds


if __name__ == "__main__":
    # new dets
    # a = [
    #     [1, 3, 54, 5],
    #     [0.6, 11.3, 43, 15],
    #     [21, 63, 6.4, 51],
    #     [31, 33, 4.8, 50],
    # ]
    # # tracks
    # b = [
    #     [0.1, 3, 5.4, 50],
    #     [6.6, 1.3, 643, 1.5],
    #     [2.1, 6.3, 66.4, 851],
    #     [30, 23, 64.8, 5.0],
    # ]

    # a = [160, 160, 170]
    # b = [60000, 70000, 60000]
    # mashi_dist2(a, b)
    # filepath = 'data.csv'
    # df = pd.read_csv(filepath).iloc[:, [0,4,6]]
    # print(df.head())

    # df_x = df[['carat', 'depth', 'price']].head(500)
    # df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
    # print(df_x.head())

    data = [
        [160, 60000],
        [160, 70000],
        [170, 60000],
        [180, 80000],
    ]
    mahalanobis2(data)
    # how to calculate distance anyway?