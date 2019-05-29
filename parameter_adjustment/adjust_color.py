import numpy as np
import cv2
from itertools import product
import sys
sys.path.insert(0, '../pipeline')

from filter_classes import *
from helper_functions import do_nothing, combine_images

window_name = "hsl"
hsl_names = ['hue', 'saturation', 'lightness']
hsl_colors = ['white', "yellow"]
hsl_bounds = ['lower', 'upper']
hsl_params = [str(x) for x in product(hsl_colors, hsl_bounds, hsl_names)]
hsl_max_min = 2*2*[[0,180], [0, 255], [0, 255]]

cv2.namedWindow(window_name)
[cv2.createTrackbar(name, window_name, *params, do_nothing) for name, params in zip(hsl_params, hsl_max_min)]

img = cv2.imread("../data/test_images/test1.jpg")

while True:
    k = cv2.waitKey(1) & 0xFF # (ESC)
    if k == 27:
        break
    hsl_values = np.array([cv2.getTrackbarPos(name, window_name) for name in hsl_params])
    hsl_white_l, hsl_white_u, hsl_yellow_l, hsl_yellow_u = np.split(hsl_values, 4)
    lt_white = ColorFilter(hsl_white_l, hsl_white_u)
    lt_yellow = ColorFilter(hsl_yellow_l, hsl_yellow_u)
    adj_img_white = lt_white.apply(img)
    adj_img_yellow = lt_yellow.apply(img)
    comb = np.array([adj_img_white, adj_img_yellow])
    comb = np.amax(comb, axis=0)
    final_img = combine_images(img, comb, adj_img_white, adj_img_yellow)
    cv2.imshow(window_name, final_img)
