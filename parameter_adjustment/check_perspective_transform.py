import numpy as np
import cv2
import sys
sys.path.insert(0, '../pipeline')

from filter_classes import *
from helper_functions import *

img = cv2.imread("../data/test_images/straight_lines1.jpg")
# src_points = np.float32([[682,284], [479,562], [479, 757], [682, 1083]])
src_points = np.float32([[600, 450], [260, 682], [1060, 682], [690, 450]])
dst_points = np.float32([[300, 0], [300,img.shape[1]], [900, img.shape[1]], [900, 0]])

for point in src_points:
    cv2.circle(img, (point[0], point[1]), 10, (0,0,255))
transform_m = cv2.getPerspectiveTransform(src_points, dst_points)
view_adj_img = cv2.warpPerspective(img, transform_m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
print(dst_points)
while True:
    f_img = combine_images(view_adj_img, view_adj_img, img, img)
    cv2.imshow("blub", f_img)
    k = cv2.waitKey(1) & 0xFF # (ESC)
    if k == 27:
        break
