
"""
Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy as np
import StringIO


def rectify_pair(image_left, image_right, viz=False):
    """
    Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """
    # Constants
    FLANN_INDEX_KDTREE = 0
    TREES = 5
    CHECKS = 100
    KNN_ITERS = 2
    LOWE_RATIO = 0.8

    # Feature extraction
    kp_left, desc_left = cv2.SIFT().detectAndCompute(image_left, None)
    kp_right, desc_right = cv2.SIFT().detectAndCompute(image_right, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=TREES)
    search_params = dict(checks=CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_left, desc_right, k=KNN_ITERS)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    pts_left = np.float32(
        [kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_right = np.float32(
        [kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Computing the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    pts_left = pts_left[mask.ravel() == 1]
    pts_right = pts_right[mask.ravel() == 1]

    # Rectifying the images
    width, height, _ = image_left.shape
    _, h1, h2 = cv2.stereoRectifyUncalibrated(
        pts_left, pts_right, F, (width, height))

    # Return the fundamental matrix,
    # the homography for warping the left image,
    # and the homography for warping the right image
    return F, h1, h2


def disparity_map(image_left, image_right):
    """
    Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """
    stereo = cv2.StereoSGBM(minDisparity=20,
                            numDisparities=128,
                            SADWindowSize=7,
                            uniquenessRatio=5,
                            speckleWindowSize=100,
                            speckleRange=5,
                            disp12MaxDiff=1,
                            P1=8 * 3 * 7 ** 2,
                            P2=32 * 3 * 7 ** 2,
                            fullDP=False
                            )
    disparity = stereo.compute(
        image_left, image_right).astype(np.float32) / 16.0
    disparity = np.uint8(disparity)
    return disparity


def point_cloud(disparity_image, image_left, focal_length):
    """
    Create a point cloud from a disparity image and a focal length.
    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    # The header for a PLY point cloud
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''

    # Generate point cloud
    h, w = image_left.shape[:2]
    Q = np.float32([[1, 0, 0,            0.5 * w],
                    [0, -1, 0,           0.5 * h],
                    [0, 0, focal_length,       0],
                    [0, 0, 0,                  1]])
    points = cv2.reprojectImageTo3D(disparity_image, Q)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    mask = disparity_image > disparity_image.min()
    out_points = points[mask]
    out_colors = colors[mask]
    verts = out_points.reshape(-1, 3)
    colors = out_colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    verts_string_IO = StringIO.StringIO()
    verts_string_IO.write((ply_header % dict(vert_num=len(verts))))
    np.savetxt(verts_string_IO, verts, '%f %f %f %d %d %d')

    return verts_string_IO.getvalue()
