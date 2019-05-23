import pickle
import pandas as pd
import numpy as np
from collections import deque
import cv2
from helper_functions import *


class Lane:
    pass

class ColorFilter:
    pass

class GradientFilter:
    pass

class LaneTracer:

    def __init__(self, cam_conf_path, color_filter, gradient_filter, roi):
        cc = pickle.load(open(cam_conf_path, "rb"))
        self.dist = cc['dist']
        self.mtx = cc['mtx']
        self.color_filter = color_filter
        self.gradient_filter = gradient_filter
        self.roi = roi

    def find_lanes(self, img):
        u_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return u_img

    def draw_lanes(self, left_lane, right_lane):
        pass

    def next_frame(self, img):
        find_lanes(img)







if __name__ == "__main__":
    cv2.namedWindow("blub")

    cam_conf_path = "../data/camera.conf"
    video_path = "../data/project_video.mp4"
    color_filter = None
    gradient_filter = None
    roi = None
    lt = LaneTracer(cam_conf_path, None, None, None)
    cap = cv2.VideoCapture(video_path)
    ret, or_img = cap.read()

    while True:
        k = cv2.waitKey(1) & 0xFF # (ESC)
        if k == 27:
            break
        elif k == 110: # (n)
            ret, or_img = cap.read()
        mod_img = lt.find_lanes(or_img)
        comb_img = combine_images(mod_img, mod_img, or_img, or_img)
        cv2.imshow("blub", comb_img)
