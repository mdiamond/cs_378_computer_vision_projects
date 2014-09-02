"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""

# TODO: Implement!

import cv2


def flip_image(image, horizontal, vertical):
    """Flips the image vertically, horizontally, or both"""
    result = image
    if(horizontal):
        result = cv2.flip(result, 1)
    if(vertical):
        result = cv2.flip(result, 0)
    return result


def negate_image(image):
    """Negates the colors in the image"""
    return cv2.bitwise_not(image)


def swap_blue_and_green(image):
    """Swaps the B and G color channels in the image"""
    b, g, r = cv2.split(image)
    return cv2.merge((g, b, r))
