from PIL import Image
import numpy as np
import cv2
import sys


a = sys.argv[1]

im = cv2.imread(a).astype(np.float32)
im /= 255.
# im = Image.open(path)
# b, g, r = im.split()
# im = Image.merge("RGB", (r, g, b))
# print(np.array(im))
im_pil = Image.fromarray(np.uint8(im))
print(im_pil)
print(np.array(im_pil))