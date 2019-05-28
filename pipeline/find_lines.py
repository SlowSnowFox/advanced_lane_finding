import pickle
import pandas as pd
import numpy as np
from collections import deque
import cv2
from helper_functions import *
from filter_classes import *


if __name__ == "__main__":
    cam_conf_path = "../data/camera.conf"
    video_path = "../data/videos/project_video.mp4"
    cv2.namedWindow("blub")
    cap = cv2.VideoCapture(video_path)
    ret, or_img = cap.read()

    src_points = np.float32([[562, 479], [284, 682],[1083,682] ,[757,479]])
    dst_points = np.float32([[300, 0], [300,or_img.shape[1]], [900, or_img.shape[1]], [900, 0]])
    mag_values = [10, 30]
    dir_values = [0.4*np.pi/2, 0.65*np.pi/2]
    abs_values = {"x":[5, 15], "y":[5, 31]}
    hsl_lower_bounds = np.array([20, 35,155])
    hsl_upper_bounds = np.array([180, 255, 255])

    color_filter = ColorFilter(hsl_lower_bounds, hsl_upper_bounds)
    gradient_filter = GradientFilter(abs_values, mag_values, dir_values)
    cam_adj = CamerAdjuster(cam_conf_path)
    perpserctive_adj = PerspectiveAdjuster(src_points, dst_points)
    lane_sep = LaneSeparator()
    lt = LaneTracer(cam_adj, color_filter, gradient_filter, perpserctive_adj, lane_sep)

    while True:
        k = cv2.waitKey(1) & 0xFF # (ESC)
        if k == 27:
            break
        elif k == 110: # (n)
            ret, or_img = cap.read()
        lane_mask = lt.detect_lanes(or_img)
        
        comb_img = combine_images(mod_img, mod_img, or_img, or_img)
        if k == 115: # (s)
            cv2.imwrite("../data/sample_step_outputs/b_trans.jpg",mod_img)
        cv2.imshow("blub", comb_img)
