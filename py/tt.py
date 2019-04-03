"""

calculate 2 vector angle
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


a = [0, 1, 0]
b = [0.4, -0.9, 0]

# print(a*b)
print(np.dot(a, b))

norm_a = a/LA.norm(a)
norm_b = b/LA.norm(b)
print(norm_a)
print(norm_b)

theta = np.arccos(np.dot(a, b) / np.dot(norm_a, norm_b) )
print(theta)