"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np
import copy

b1 = cv2.imread('test_data/books_1.png')
b2 = cv2.imread('test_data/books_2.png')
b3 = cv2.imread('test_data/books_3.png')


def show(img):
    try:
        while True:
            cv2.imshow('img', img)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()


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
    h1, w1 = image.shape[:2]

    # Make matrix of image points
    pts = np.matrix([
        [0, 0, w1, w1],
        [0, h1, 0, h1],
        [1, 1, 1, 1]
    ])
    # Get product of homography, image points
    warped = np.dot(homography, pts)
    # Transform homogeneous coordinates back into image coords
    img_1 = np.divide(warped[0], warped[2])
    img_2 = np.divide(warped[1], warped[2])

    min_length = np.amin(img_1)
    min_height = np.amin(img_2)

    dsize_len = np.amax(img_1) - min_length
    dsize_height = np.amax(img_2) - min_height

    dsize = (
        int(dsize_len),
        int(dsize_height)
    )

    ul = (int(min_length), int(min_height))

    tx = homography[0][2]
    ty = homography[1][2]

    t = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]])
    ht = t.dot(homography)
    t = np.array([[1, 0, -min_length], [0, 1, -min_height], [0, 0, 1]])
    ht = t.dot(homography)

    res = cv2.warpPerspective(image, ht, dsize)

    blue, green, red = cv2.split(res)
    alpha = np.zeros(blue.shape, dtype=np.uint8)
    alpha.fill(255)

    for row in range(0, res.shape[0]):
        for col in range(0, res.shape[1]):
            pix = res[row, col]
            if pix[0] == 0 and pix[1] == 0 and pix[2] == 0:
                alpha[row][col] = 0
            else:
                alpha[row][col] = 255

    res = cv2.merge([blue, green, red, alpha])
    return res, ul


def _find_mosaic_size(images, origins):
    """Return the size of the final mosaic image"""
    highX = 0
    highY = 0

    # Locate the positive and negative boundaries of the image
    for i in range(len(images)):
        origin = origins[i]
        if(images[i].shape[0] + origin[0] > highX):
            highX = images[i].shape[0] + origin[0]
        if(images[i].shape[1] + origin[1] > highY):
            highY = images[i].shape[1] + origin[1]

    # Return the width and height, calculated using the boundaries
    return (highX, highY)


def _adjust_origins(origins):
    """Translate all origins to account for any that are negative

    Turns all origins into positive coordinates that are still
    accurate relative to each other
    """
    lowX = 0
    lowY = 0

    # Locate the lowest x and y origins
    for o in origins:
        if(o[0] < lowX):
            lowX = o[0]
        if(o[1] < lowY):
            lowY = o[1]

    # Translate each origin by the lowest origin
    for i in range(len(origins)):
        t = (origins[i][0] - lowX, origins[i][1] - lowY)
        origins[i] = t

    return origins


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
    # Swap all origins tuples because our code is weird
    swapped = []
    for origin in origins:
        swapped.append((origin[1], origin[0]))
    origins = tuple(swapped)

    # Adjust origins so that all images start from 0,0
    origins = tuple(_adjust_origins(list(origins)))
    # Find the overall size of the mosaic
    mosaic_size = _find_mosaic_size(images, origins)

    # Initialize an empty destination image for compositing
    dst = np.zeros((mosaic_size[0], mosaic_size[1], 4), dtype=np.uint8)

    # Composite each warped image into the destination image
    for i in range(len(images)):
        image = images[i]
        origin = origins[i]
        s = image.shape
        w = s[1]
        h = s[0]
        for row in range(origin[0], origin[0] + h):
            for col in range(origin[1], origin[1] + w):
                pix = dst[row, col]
                if pix[0] == 0 and pix[1] == 0 and pix[2] == 0 and pix[3] == 0:
                    dst[row, col] = image[row - origin[0], col - origin[1]]
                if pix[0] == 0 and pix[1] == 0 and pix[2] == 0:
                    pix[3] = 0
                else:
                    pix[3] = 255

    return dst
