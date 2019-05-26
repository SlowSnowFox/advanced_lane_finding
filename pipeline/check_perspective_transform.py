import numpy as np
import cv2
from filter_classes import *


img = cv2.imread("../data/test_images/test1.jpg")
# src_points = np.float32([[682,284], [479,562], [479, 757], [682, 1083]])
src_points = np.float32([[562, 479], [284, 682],[1083,682] ,[757,479]])
dst_points = np.float32([[300, 0], [300,img.shape[1]], [900, img.shape[1]], [900, 0]])
transform_m = cv2.getPerspectiveTransform(src_points, dst_points)
view_adj_img = cv2.warpPerspective(img, transform_m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
print(dst_points)
while True:
    cv2.imshow("blub", view_adj_img)
    k = cv2.waitKey(1) & 0xFF # (ESC)
    if k == 27:
        break
