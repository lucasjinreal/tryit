from PIL import Image
import numpy as np
import cv2
import sys


a = sys.argv[1]

im = cv2.imread(a).astype(np.float32)
im = cv2.resize(im, (1280, 768))
print(im.shape)
im /= 255.
im = np.transpose(im, (2, 1, 0))
print(im)
# im = Image.open(path)
# b, g, r = im.split()
# im = Image.merge("RGB", (r, g, b))
# print(np.array(im))
# im_pil = Image.fromarray(np.uint8(im))
# print(im_pil)
# print(np.array(im_pil))