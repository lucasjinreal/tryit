import cv2
import sys
import glob
import os


all_imgs = glob.glob(os.path.join(sys.argv[1], '*.jpg'))
for i in all_imgs:
    a = cv2.imread(i)
    cv2.imshow('rr', a)
    cv2.waitKey(0)