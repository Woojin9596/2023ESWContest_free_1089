import cv2
import glob
import numpy as np


def find_chessboard_corners(images_folder):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    images = glob.glob(images_folder + '/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    return objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints, image_shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_shape, 1, image_shape)

    return mtx, dist, newcameramtx, roi


def undistort_image(image_path, mtx, dist, newcameramtx, roi):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return dst


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error

    return tot_error / len(objpoints)


def main(images_folder):
    objpoints, imgpoints = find_chessboard_corners(images_folder)
    image_shape = cv2.imread(glob.glob(images_folder + '/*.jpg')[0]).shape[:2][::-1]

    mtx, dist, newcameramtx, roi = calibrate_camera(objpoints, imgpoints, image_shape)
    print("Calibration matrix (mtx):\n", mtx)
    print("Distortion coefficients (dist):\n", dist)

    # Calculate reprojection error
    rvecs = [np.zeros(3) for _ in range(len(objpoints))]
    tvecs = [np.zeros(3) for _ in range(len(objpoints))]
    tot_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print("Total reprojection error:", tot_error)

    # Undistort example image
    image_path = glob.glob(images_folder + '/*.jpg')[0]
    undistorted_img = undistort_image(image_path, mtx, dist, newcameramtx, roi)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_folder = './image'  # Path to the folder containing calibration images
    main(images_folder)