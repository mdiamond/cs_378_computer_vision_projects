"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy


def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """

    """ FEATURE EXTRACTION """ 
    kp_left, desc_left = cv2.SIFT().detectAndCompute(image_left, None)
    kp_right, desc_right = cv2.SIFT().detectAndCompute(image_right, None)
    FLANN_INDEX_KDTREE = 0
    TREES = 5
    CHECKS = 100
    KNN_ITERS = 2
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=TREES)
    search_params = dict(checks=CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_left, desc_right, k=KNN_ITERS)
     # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    img_left = np.float32(
        [kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    img_right = np.float32(
        [kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    """ Computing the fundamental matrix """ 
    F, mask = cv2.findFundamentalMat(img_left,img_right,cv2.FM_RANSAC)
    img_left = img_left[mask.ravel()==1]
    img_right = img_right[mask.ravel()==1]

    """ Rectifying the images """
     x,y,_ = image_left.shape
     cv2.stereoRectifyUncalibrated(img_left, img_right, F, [x,y]) # I am not too sure of this.. the x and y points that is 




def disparity_map(image_left, image_right):
    """Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """
    pass


def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    pass
