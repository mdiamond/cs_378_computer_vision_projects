"""
This script tests our program in the most
simple possible way.
Three hardcoded images are used to create a simple
panorama.
"""

import cv2
import numpy as np
import pano_stitcher as ps

# Open all of the images
b1 = cv2.imread('my_panos/src/part1.jpg')
b2 = cv2.imread('my_panos/src/part2.jpg')
b3 = cv2.imread('my_panos/src/part3.jpg')

# Find a homography to warp image 1
# onto image 2, warp it
b1_homog = ps.homography(b2, b1)
b1_warped, b1_origins = ps.warp_image(b1, b1_homog)
print 'origins for warped b1:', b1_origins

# Set b2 to be at the origin
b2_origins = (0, 0)

# Find a homography to warp image 3
# onto image 2, warp it
b3_homog = ps.homography(b2, b3)
b3_warped, b3_origins = ps.warp_image(b3, b3_homog)
print 'origins for warped b3:', b3_origins

# Convert b2 to a 4-channel image
b2 = cv2.cvtColor(b2, cv2.COLOR_BGR2BGRA)

# Create the mosaic, write it out
pano = ps.create_mosaic(
    [b1_warped, b2, b3_warped], [b1_origins, b2_origins, b3_origins])
cv2.imwrite('my_panos/pano_test.png', pano)
