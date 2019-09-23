import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 150], [100, 250]])
M = cv2.getAffineTransform(pts2, pts1)

res = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('affine', res)
cv2.imshow('original', img)
cv2.waitKey(0)