"""

calculate 2 vector angle
"""
import numpy as np
import cython_bbox as bu



a = np.random.rand(10, 4).astype(np.float32)
print(a.dtype)
bu.bbox_overlaps(a, a)