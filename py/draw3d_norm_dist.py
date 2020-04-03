import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


N = 60
x = np.linspace(-3, 3, N)
y = np.linspace(-3, 4, N)
x, y = np.meshgrid(x, y)

mu = np.array([0, 1])
sigma = np.array([[1, -0.5], [-0.5, 1.5]])

pos = np.empty(x.shape+(2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

def multivariate_gaussian(pos, mu, sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(sigma)
    Sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N


# a = np.array([0, 1, 2])
# b = np.array([[0, 1, 2, 3],
# [4, 5, 6, 7],
# [8, 9, 10, 11]])
# c = np.dot(a, b)
# print(c)
# print(a[:, np.newaxis]*b)

z = multivariate_gaussian(pos, mu, sigma)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z, rstride=3, cstride=3, linewidth=1, cmap=cm.viridis)

ax.view_init(27, -21)
plt.show()