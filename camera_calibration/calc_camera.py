import cv2
import numpy as np
import os
import pickle


def calculate_params(img_paths, img_size, save_dir, checkboard_size):
    objp = np.zeros((checkboard_size[1]*checkboard_size[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkboard_size[0], 0:checkboard_size[1]].T.reshape(-1,2)
    objpoints = []
    imgpoints = []


    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkboard_size, None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    camera_config = {}
    camera_config['mtx'] = mtx
    camera_config['dist'] = dist
    pickle.dump(camera_config, open(save_dir, "wb"))


if __name__ == "__main__":
    img_dir = "../data/camera_cal/"
    img_paths = [img_dir + x for x in os.listdir(img_dir)]
    img_size = (1280, 960)
    save_dir = "../data/camera.conf"
    checkboard_size = (9,6)
    calculate_params(img_paths, img_size, save_dir, checkboard_size)
