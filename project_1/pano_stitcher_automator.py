"""
A program that will attempt too stich together any
images passed to it.

You must pass 4 or more image paths as arguments.
The very last argument must be the name of the output mosaic image file.
All other arguments must be the images to be stitched into the mosaic.

Assumes that the input image paths passed are in the order of
left to right in terms of the content of the image.
An arbitrary image is used as the center, and
all other images are warped to match it.
"""

import cv2
import numpy as np
import pano_stitcher as ps
import sys

if len(sys.argv) < 4:
    print "MUST PASS IN AT LEAST 3 IMAGES TO STITCH"
    exit()

# Remove the program name from the arguments
sys.argv.remove(sys.argv[0])

# Get the name for the output mosaic, then remove it from the list of arguments
output_mosaic_name = sys.argv[len(sys.argv) - 1]
sys.argv.remove(sys.argv[len(sys.argv) - 1])

middle = len(sys.argv) / 2
left = list(reversed(range(len(sys.argv) / 2)))
right = range(middle + 1, len(sys.argv))
print "Left, middle, right: ", list(reversed(left)), middle, right

# The image that all the other images are warped to match
middle_image = cv2.imread(sys.argv[middle])

# All of the other images, their respective warped
# images, and their warped images coordinates
right_images = []
right_warped_images = []
right_images_origins = []
left_images = []
left_warped_images = []
left_images_origins = []


# LEFT IMAGES
print "LEFT IMAGES:"

# Arbitrary index
a = 0

for i in left:
    # Show the name of the image
    print sys.argv[i]

    # Open it and append it to the left images list
    left_images.append(cv2.imread(sys.argv[i]))

    # If closest to the middle image, warp into that perspective
    # Otherwise, warp into the perspective of the last image
    # that was warped
    if a is 0:
        h = ps.homography(middle_image, left_images[a])
    else:
        h = ps.homography(left_warped_images[a - 1], left_images[a])
    warped, origins = ps.warp_image(left_images[a], h)
    left_warped_images.append(warped)

    # If closest to the middle image, use normal origins
    # Otherwise, use the origins + the origins of the
    # last image that was warped
    if a is 0:
        left_images_origins.append(origins)
    else:
        prev_origins = left_images_origins[a - 1]
        new_origins = (
            origins[0] + prev_origins[0],
            origins[1] + prev_origins[1]
        )
        left_images_origins.append(new_origins)

    left_warped_images = list(reversed(left_warped_images))
    left_images_origins = list(reversed(left_images_origins))

    a += 1


# RIGHT IMAGES
print "RIGHT IMAGES:"

# Arbitrary index
a = 0

for i in right:
    # Show the name of the image
    print sys.argv[i]

    # Open it and append it to the right images list
    right_images.append(cv2.imread(sys.argv[i]))

    # If closest to the middle image, warp into that perspective
    # Otherwise, warp into the perspective of the last image
    # that was warped
    if a is 0:
        h = ps.homography(middle_image, right_images[a])
    else:
        h = ps.homography(right_warped_images[a - 1], right_images[a])
    warped, origins = ps.warp_image(right_images[a], h)
    right_warped_images.append(warped)

    # If closest to the middle image, use normal origins
    # Otherwise, use the origins + the origins of the
    # last image that was warped
    if a is 0:
        right_images_origins.append(origins)
    else:
        prev_origins = right_images_origins[a - 1]
        new_origins = (
            origins[0] + prev_origins[0],
            origins[1] + prev_origins[1]
        )
        right_images_origins.append(new_origins)

    a += 1

# PUT IT ALL TOGETHER
print "CREATING MOSAIC!"

# Warped images
all_warped_images = []
all_warped_images.extend(left_warped_images)
middle_image = cv2.cvtColor(middle_image, cv2.COLOR_BGR2BGRA)
all_warped_images.append(middle_image)
all_warped_images.extend(right_warped_images)

# Origins
all_origins = []
all_origins.extend(left_images_origins)
all_origins.append((0, 0))
all_origins.extend(right_images_origins)

res = ps.create_mosaic(all_warped_images, all_origins)

cv2.imwrite(output_mosaic_name,
            res)
