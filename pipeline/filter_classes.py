import cv2
import numpy as np
import pickle


class Lane:
    pass


class ColorFilter:

    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def apply(self, img):
        c_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        r_img = cv2.inRange(c_img, self.lower_bounds, self.upper_bounds)
        return r_img


class GradientFilter:

    def __init__(self, magnitude_threshold=(0, 255), dir_threshold=(0, np.pi/2), sobel_kernel=3):
        self.magnitude_threshold = magnitude_threshold
        self.dir_threshold = dir_threshold
        self.sobel_kernel = sobel_kernel

    def apply(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adj_img = self._apply_mag_thresh(img)
        return adj_img

    def _apply_mag_thresh(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        mag = np.sqrt(np.square(sobelx) + np.square(sobely))
        mag_sc = np.uint8(mag*255/np.max(mag))
        binary = np.zeros_like(mag_sc)
        binary[(mag_sc >= self.magnitude_threshold[0]) & (mag_sc <= self.magnitude_threshold[1])] = 1
        return binary

    def _apply_dir_thresh(self, img):
        sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel))
        sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel))
        dir_sobel = np.arctan2(sobely, sobelx)
        binary = np.zeros_like(np.uint8(dir_sobel))
        binary[(dir_sobel >= self.dir_threshold[0]) & (dir_sobel <= self.dir_threshold[1])] = 1
        return binary


class CamerAdjuster:

    def __init__(self, config_path):
        cc = pickle.load(open(config_path, "rb"))
        self.dist = cc['dist']
        self.mtx = cc['mtx']

    def apply(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class PerspectiveAdjuster:

    def __init__(self, src_points, dst_points):
        self.t_m = cv2.getPerspectiveTransform(src_points, dst_points)
        self.t_minv = cv2.getPerspectiveTransform(dst_points, src_points)

    def apply(self, img):
        view_adj_img = cv2.warpPerspective(img, self.t_m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return view_adj_img


class LaneTracer:

    def __init__(self, cam_adj, color_filter, gradient_filter, perspective_adj):
        self.cam_adj = cam_adj
        self.color_filter = color_filter
        self.gradient_filter = gradient_filter
        self.perspective_adj = perspective_adj

    def trace_lanes(self, img):

        return img

    def detect_lanes(self, img):
        img = self.cam_adj.apply(img)
        hsl_mask = self.color_filter.apply(img)
        canny_mask = self.gradient_filter.apply(img)
        comb_mask = hsl_mask | canny_mask
        view_adj = self.perspective_adj.apply(img)
        return view_adj


    def draw_lanes(self, left_lane, right_lane):
        pass



    def next_frame(self, img):
        find_lanes(img)
