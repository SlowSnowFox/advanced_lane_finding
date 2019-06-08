import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


class Lane:

    def __init__(self, polynomial, pixels, env, color="blue", img_size=(720, 1280)):
        self.polynomial = polynomial
        self.env = env
        self.pixels = pixels
        self.color = [255, 0, 0]
        self.img_size = img_size
        self._line_pixels = None
        self._curvature = None
        self.x_mpx = 3.7 / 800
        self.y_mpx = 30 / img_size[0]
        if color == "red":
            self.color = [0, 0, 255]

    @property
    def line_pixels(self):
        if self._line_pixels:
            return self._line_pixels
        ploty = np.linspace(0, self.img_size[0]-1, self.img_size[0])
        plotx = self.polynomial[0]*ploty**2 + self.polynomial[1]*ploty + self.polynomial[2]
        ploty = ploty.astype(int)
        plotx = plotx.astype(int)
        out_of_bounds = [i for i, y in enumerate(plotx) if (y >= self.img_size[1]) or (y < 0.0)]
        plotx = np.delete(plotx, out_of_bounds)
        ploty = np.delete(ploty, out_of_bounds)
        self._line_pixels = [plotx, ploty]
        return [plotx, ploty]

    def _draw_pixels(self, img):
        img[self.pixels[0], self.pixels[1]] = self.color

    def _draw_env(self, img):
        if type(self.env) is list: # Lane was created by Seperator
            for box in self.env:
                cv2.rectangle(img,(box[0], box[1]),
                (box[2], box[3]),(0,255,0), 2)
        else: # Lane was created by Tracer TODO make prettier
            img[self.env[0], self.env[1]] = [0, 255, 0]

    def _draw_polynomial(self, img):
        img[self.line_pixels[1], self.line_pixels[0]] = [255, 255, 255]

    def draw(self, img):
        self._draw_pixels(img)
        self._draw_env(img)
        self._draw_polynomial(img)

    @property
    def curvature(self):
        if self._curvature:
            return self._curvature
        # convert parabel parameters from pixel into meter space
        # x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c
        pol_factor_0 = self.x_mpx / (self.y_mpx ** 2) * self.polynomial[0]
        pol_factor_1 = (self.x_mpx/self.y_mpx) * self.polynomial[1]
        curvature = ((1 + (2*pol_factor_0*self.img_size[0] + pol_factor_1)**2)**1.5) / np.absolute(2*pol_factor_0)
        self._curvature = curvature
        return curvature


class ColorFilter:

    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def apply(self, img):
        c_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        r_img = cv2.inRange(c_img, self.lower_bounds, self.upper_bounds)
        return r_img


class GradientFilter:

    def __init__(self, abs_threshold, magnitude_threshold=(0, 255), dir_threshold=(0, np.pi/2), sobel_kernel=3):
        self.abs_x = abs_threshold['x']
        self.abs_y = abs_threshold['y']
        self.magnitude_threshold = magnitude_threshold
        self.dir_threshold = dir_threshold
        self.sobel_kernel = sobel_kernel

    def apply(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3,3), 0)
        dir_img = self._apply_dir_thresh(img)
        mag_img = self._apply_mag_thresh(img)
        abs_x_img = self._apply_abs_thresh(img)
        abs_y_img = self._apply_abs_thresh(img, orient='y')
        comb_1 = np.zeros_like(dir_img)
        comb_1[(dir_img == 1) | (mag_img == 1)] = 255
        comb_2 = np.zeros_like(dir_img)
        comb_2[(abs_x_img == 1) & (abs_y_img == 1)] = 255
        comb_3 = comb_2 & comb_1
        return comb_3

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

    def _apply_abs_thresh(self, img, orient='x'):
        gradient = [1,0]
        thresh = self.abs_x
        if orient == "y":
            gradient = [0, 1]
            thresh = self.abs_y
        sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, *gradient, ksize=self.sobel_kernel))
        sobel = np.uint8(255*sobel/np.max(sobel))
        binary = np.zeros_like(sobel)
        binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
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

    def reverse(self, img):
        rev_adj_img = cv2.warpPerspective(img, self.t_minv, (img.shape[1], img.shape[0]))
        return rev_adj_img


class LaneSeparator:

    def __init__(self, average=None, slices=9, side_range=100, minpix=50, screen_dpi=96):
        self.slices = slices
        self.screen_dpi = screen_dpi # needed for drawing the histogram in opencv
        self.average = None
        self.side_range = side_range
        self.minpix = minpix # nr of pixels needed to readjust base

    def create_hist(self, img, slice_nr=0, nr_of_slices=2):
        start_i = int(img.shape[0]/nr_of_slices * slice_nr)
        end_i = int(img.shape[0]/nr_of_slices * (slice_nr + 1))
        img_slice = img[start_i:end_i,:]
        hist = np.sum(img_slice, axis=0)
        if self.average:
            hist = (hist[(self.average-1):] + hist[:-(self.average-1)])/hist(self.average)
            hist = np.pad(hist, 2, "constant", constant_values=0)
        return hist

    def create_lanes(self, img):
        window_height = window_height = np.int(img.shape[0]//self.slices)
        left_lane_inds = []
        right_lane_inds = []
        comp_hist = self.create_hist(img, slice_nr=1)
        midpoint = img.shape[1]//2
        init_l_base = np.argmax(comp_hist[:midpoint])
        init_r_base = np.argmax(comp_hist[midpoint:]) + midpoint
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        rightx_current = init_r_base
        leftx_current = init_l_base
        boxes_left = []
        boxes_right = []
        for slice in range(self.slices):
            win_y_low = img.shape[0] - (slice+1)*window_height
            win_y_high = img.shape[0] - slice*window_height
            win_xleft_low = leftx_current - self.side_range
            win_xleft_high = leftx_current + self.side_range
            win_xright_low = rightx_current - self.side_range
            win_xright_high = rightx_current + self.side_range
            boxes_left.append([win_xleft_low, win_y_low, win_xleft_high, win_y_high])
            boxes_right.append([win_xright_low, win_y_low, win_xright_high, win_y_high])

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return boxes_left, [lefty, leftx], boxes_right, [righty, rightx]

    def create_hist_img(self, img, slice_nr):
        hist = self.create_hist(img, slice_nr)
        hist_fig = plt.figure(figsize=(1280/self.screen_dpi, 720/self.screen_dpi), dpi=self.screen_dpi)
        ax = hist_fig.add_subplot(111)
        ax.plot(hist)
        hist_fig.canvas.draw()
        np_image = np.fromstring(hist_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        np_image = np_image.reshape(hist_fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(hist_fig)
        return np_image


class LaneTracer:

    def __init__(self, cam_adj, color_filters, gradient_filter, perspective_adj, lane_separator, trace_margin=100, font=cv2.FONT_HERSHEY_SIMPLEX):
        self.cam_adj = cam_adj
        self.color_filters = color_filters
        self.gradient_filter = gradient_filter
        self.perspective_adj = perspective_adj
        self.lane_separator = lane_separator
        self.past_left_lanes = np.array([None])
        self.past_right_lanes = np.array([None])
        self.trace_margin = trace_margin
        self.font = font

    def trace_lanes(self, img):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_fit = self.past_left_lanes[0].polynomial
        right_fit = self.past_right_lanes[0].polynomial
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                        left_fit[2] - self.trace_margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                        left_fit[1]*nonzeroy + left_fit[2] + self.trace_margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                        right_fit[2] - self.trace_margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                        right_fit[1]*nonzeroy + right_fit[2] + self.trace_margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_lane_pixel = [lefty, leftx]
        right_lane_pixel = [righty, rightx]
        self.past_left_lanes[0] = self.pixel_to_lane(left_lane_pixel, [])
        self.past_right_lanes[0] = self.pixel_to_lane(right_lane_pixel, [])

    def create_binary_mask(self, img):
        img = self.cam_adj.apply(img)
        canny_mask = self.gradient_filter.apply(img)
        hsl_masks = np.array([filter.apply(img) for filter in self.color_filters])
        hsl_mask = np.amax(hsl_masks, axis=0)
        comb_mask = hsl_mask | canny_mask
        view_adj = self.perspective_adj.apply(comb_mask)
        return view_adj

    def detect_lanes(self, img):
        left_lane_boxes, left_lane_pixel, right_lane_boxes, right_lane_pixel = self.lane_separator.create_lanes(img)
        self.past_left_lanes[0] = self.pixel_to_lane(left_lane_pixel, left_lane_boxes)
        self.past_right_lanes[0] = self.pixel_to_lane(right_lane_pixel, right_lane_boxes)

    def pixel_to_lane(self, pixels, env):
        try:
            poli_fit = np.polyfit(pixels[0], pixels[1], 2)
        except: # no fit was found thats why a straight line is predicted
            poli_fit = [0,1,1]
        return Lane(poli_fit, pixels, env)

    def calculate_offset(self):
        midpoint_img = 1280/2
        midpoint_lane = (min(self.past_left_lanes[0].line_pixels[0]) + max(self.past_right_lanes[0].line_pixels[0]))/2
        diff = midpoint_img-midpoint_lane
        diff = diff * (3.7/800)# convert to meters
        return diff

    def calculate_curvature(self):
        left = self.past_left_lanes[0].curvature
        right = self.past_right_lanes[0].curvature
        return (left+right) / 2

    def sanity_check(self):
        """
        checks if the most recently line was valid
        """
        if not self.past_left_lanes[0]:
            return False
        return True

    def draw_road(self, img):
        left_pts = self.past_left_lanes[0].line_pixels
        right_pts = self.past_right_lanes[0].line_pixels
        pts_left = np.array([np.transpose(np.vstack(left_pts))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack(right_pts)))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(img, np.int_([pts]), (0,255,0))

    def next_frame(self, img):
        bin_img = self.create_binary_mask(img)
        if not self.sanity_check():
            self.detect_lanes(bin_img)
        else: # there is a lane
            self.trace_lanes(bin_img)
        bin_img = np.dstack([bin_img, bin_img, bin_img])
        self.past_left_lanes[0].draw(bin_img)
        self.past_right_lanes[0].draw(bin_img)
        self.draw_road(bin_img)
        fin_img = self.perspective_adj.reverse(bin_img)
        fin_img = cv2.addWeighted(img, 1, fin_img, 0.3, 0)
        curv_text = "curvature: " + str(int(self.calculate_curvature()))
        offset_text = "offset: " + str(self.calculate_offset())[:4] + " meters"
        cv2.putText(fin_img,curv_text,(10,50), self.font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(fin_img,offset_text, (10,100), self.font, 1, (255,255,255),2,cv2.LINE_AA )
        return fin_img
