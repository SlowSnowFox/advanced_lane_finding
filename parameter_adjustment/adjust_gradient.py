import numpy as np
import cv2
from itertools import product
import sys
sys.path.insert(0, '../pipeline')

from filter_classes import *
from helper_functions import do_nothing, combine_images

window_name = "Sobel"
mag_params = ["low_mt", "high_mt"]
dir_params = ["low_dt", "high_dt"]
abs_params = ["low_x", "high_x","low_y", "high_y"]
mag_max_min = [[0,255], [0,255]]
dir_max_min = [[0, 100], [0,100]]
abs_max_min = 2*[[0,255], [0,255]]
init_mag_values = [0, 255]
init_dir_values = [0, 100]

cv2.namedWindow(window_name)
[cv2.createTrackbar(name, window_name, *params, do_nothing) for name, params in zip(mag_params, mag_max_min)]
[cv2.createTrackbar(name, window_name, *params, do_nothing) for name, params in zip(dir_params, dir_max_min)]
[cv2.createTrackbar(name, window_name, *params, do_nothing) for name, params in zip(abs_params, abs_max_min)]
[cv2.setTrackbarPos(name, window_name, value) for name, value in zip(mag_params, init_mag_values)]
[cv2.setTrackbarPos(name, window_name, value) for name, value in zip(dir_params, init_dir_values)]

or_img = cv2.imread("../data/test_images/test1.jpg")
img = cv2.cvtColor(or_img, cv2.COLOR_BGR2GRAY)

while True:
    k = cv2.waitKey(1) & 0xFF # (ESC)
    if k == 27:
        break
    mag_values = [cv2.getTrackbarPos(name, window_name) for name in mag_params]
    dir_values = [cv2.getTrackbarPos(name, window_name) for name in dir_params]
    dir_values = [(x * np.pi/2)/100 for x in dir_values]
    abs_values = [cv2.getTrackbarPos(name, window_name) for name in abs_params]
    abs_val_x, abs_val_y = np.split(np.array(abs_values), 2)
    gf = GradientFilter(mag_values, dir_values, sobel_kernel=3)
    dir_img = gf._apply_dir_thresh(img)
    mag_img = gf._apply_mag_thresh(img)
    abs_x_img = gf._apply_abs_thresh(img, thresh=abs_val_x)
    abs_y_img = gf._apply_abs_thresh(img, thresh=abs_val_y, orient='y')
    comb_img = np.zeros_like(dir_img)
    comb_img[(dir_img == 1) | (mag_img == 1)] = 255
    for sub_img in [abs_x_img, abs_y_img, dir_img, mag_img]:
        sub_img[(sub_img==1)] = 255
    final_img = combine_images(or_img, abs_x_img, abs_y_img, mag_img)
    cv2.imshow(window_name, final_img)
