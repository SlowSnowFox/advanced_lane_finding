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

    src_points = np.float32([[600, 450], [270, 700], [1060, 700], [690, 450]])
    dst_points = np.float32([[250, 0], [250,or_img.shape[1]], [1050, or_img.shape[1]], [1050, 0]])
    mag_values = [50, 100]
    dir_values = [0.4*np.pi/2, 0.65*np.pi/2]
    abs_values = {"x":[5, 15], "y":[5, 31]}
    hsl_lower_bounds_y = np.array([20, 160, 80])
    hsl_upper_bounds_y = np.array([45, 255, 170])
    hsl_lower_bounds_w = np.array([0, 170,0])
    hsl_upper_bounds_w = np.array([180, 255, 255])
    color_filter_white = ColorFilter(hsl_lower_bounds_w, hsl_upper_bounds_w)
    color_filter_yellow = ColorFilter(hsl_lower_bounds_y, hsl_upper_bounds_y)
    color_filters = [color_filter_white, color_filter_yellow]
    gradient_filter = GradientFilter(abs_values, mag_values, dir_values, sobel_kernel=9)
    cam_adj = CamerAdjuster(cam_conf_path)
    perpserctive_adj = PerspectiveAdjuster(src_points, dst_points)
    lane_sep = LaneSeparator()
    lt = LaneTracer(cam_adj, color_filters, gradient_filter, perpserctive_adj, lane_sep)

    while True:
        k = cv2.waitKey(1) & 0xFF # (ESC)
        if k == 27:
            break
        elif k == 110: # (n)
            ret, or_img = cap.read()
        elif k == 115: # (s)
            cv2.imwrite("../data/sample_step_outputs/b_trans.jpg", mod_img)

        lane_mask = lt.detect_lanes(or_img)
        final_mask = np.dstack([lane_mask, lane_mask, lane_mask])
        lf, lr = lane_sep.create_lanes(lane_mask)
        lr.draw(final_mask)
        lf.draw(final_mask)
        persp_img = perpserctive_adj.apply(or_img)
        comb_img = combine_images(or_img, lane_mask, final_mask, persp_img)
        cv2.imshow("blub", comb_img)
