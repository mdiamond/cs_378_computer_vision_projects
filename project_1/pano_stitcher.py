"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np

b1 = cv2.imread('test_data/books_1.png')
b2 = cv2.imread('test_data/books_2.png')


def _match_features(image_a, image_b):
    sift = cv2.SIFT()

    # We experimented with the FLANN based matcher, but achieved
    # better results with the bruteforce matcher
    # using the NORM_L2 distance measurement since we're using
    # SIFT as the feature detector
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Get key points and descriptors
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Filter matches
    initial_matches = matcher.knnMatch(des_a, des_b, k=2)
    filtered_matches = [m for m, n
                        in initial_matches if m.distance < (0.75 * n.distance)]

    # Create source and destination matrices for matches
    src = np.float32([kp_a[m.queryIdx].pt
                      for m in filtered_matches]).reshape(-1, 1, 2)
    dest = np.float32([kp_b[m.trainIdx].pt
                       for m in filtered_matches]).reshape(-1, 1, 2)
    return src, dest


def homography(image_a, image_b):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """
    # Match features and get the results,
    # pass them along to findHomography to generate
    # a tranformation matrix and return it
    src, dest = _match_features(image_a, image_b)
    M, _ = cv2.findHomography(dest, src, cv2.RANSAC, 5)
    return M


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """
    # Get the dot product of the image and homography
    dp = image.dot(homography)
    # Warp the image by homography, use
    # dot product * scale factor for image dimensions
    dsize = (
        (int)(dp.shape[1] * homography[0][0]),
        (int)(dp.shape[0] * homography[1][1])
    )
    res = cv2.warpPerspective(image, homography, dsize)
    cv2.imshow('res', res)
    cv2.waitKey()
    # dp is definitely not the correct thing to return here,
    # see docstring
    return res, dp
    pass


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """
    pass
