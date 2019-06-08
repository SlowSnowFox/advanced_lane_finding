import pickle
import pandas as pd
import numpy as np
from collections import deque
import cv2
from helper_functions import *
from filter_classes import *
from moviepy.editor import VideoFileClip


if __name__ == "__main__":
    cam_conf_path = "../data/camera.conf"
    video_path = "../data/videos/project_video.mp4"
    cv2.namedWindow("blub")
    cap = cv2.VideoCapture(video_path)
    ret, or_img = cap.read()

    src_points = np.float32([[600, 450], [270, 700], [1060, 700], [690, 450]])
    dst_points = np.float32([[250, 0], [250,or_img.shape[0]], [1050, or_img.shape[0]], [1050, 0]])
    mag_values = [50, 100]
    dir_values = [0.4*np.pi/2, 0.65*np.pi/2]
    abs_values = {"x":[5, 15], "y":[5, 31]}
    hsl_lower_bounds_y = np.array([20, 120, 60])
    hsl_upper_bounds_y = np.array([45, 255, 255])
    hsl_lower_bounds_w = np.array([0, 170,0])
    hsl_upper_bounds_w = np.array([180, 255, 255])
    color_filter_white = ColorFilter(hsl_lower_bounds_w, hsl_upper_bounds_w)
    color_filter_yellow = ColorFilter(hsl_lower_bounds_y, hsl_upper_bounds_y)
    color_filters = [color_filter_white, color_filter_yellow]
    gradient_filter = GradientFilter(abs_values, mag_values, dir_values, sobel_kernel=9)
    cam_adj = CamerAdjuster(cam_conf_path)
    perpserctive_adj = PerspectiveAdjuster(src_points, dst_points)
    lane_sep = LaneSeparator(side_range=100)
    lt = LaneTracer(cam_adj, color_filters, gradient_filter, perpserctive_adj, lane_sep)
    while True:
        k = cv2.waitKey(1) & 0xFF # (ESC)
        if k == 27:
            break
        elif k == 110: # (n)
            ret, or_img = cap.read()
        elif k == 115: # (s)
            cv2.imwrite("../data/output_images/sf.jpg", mod_img)

        applied_frame = lt.next_frame(or_img)
        comb_img = combine_images(or_img, or_img, applied_frame, or_img)
        cv2.imshow("blub", comb_img)

cap.release()
cv2.destroyAllWindows()
