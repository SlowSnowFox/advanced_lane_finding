import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gradient = [1,0]
    if orient == "y":
        gradient = [0, 1]
    sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, *gradient, ksize=sobel_kernel))
    sobel = np.uint8(255*sobel/np.max(sobel))
    binary = np.zeros_like(sobel)
    binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    return binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    mag_sc = np.uint8(mag*255/np.max(mag))
    binary = np.zeros_like(mag_sc)
    binary[(mag_sc >= mag_thresh[0]) & (mag_sc <= mag_thresh[1])] = 1
    return binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    dir_sobel = np.arctan2(sobely, sobelx)
    binary = np.zeros_like(dir_sobel)
    binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return dir_binary

gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
