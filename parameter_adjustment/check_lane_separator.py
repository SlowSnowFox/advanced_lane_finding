import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../pipeline')

from filter_classes import *
from helper_functions import do_nothing, combine_images


img = cv2.imread("../data/sample_step_outputs/b_trans.jpg")

ls = LaneSeparator(4)
print(img.shape)
while True:
    k = cv2.waitKey(1) & 0xFF # (ESC)
    if k == 27:
        break
    hist = ls.create_hist_img(img, slice_nr=0)
    cv2.imshow("blub", hist)
