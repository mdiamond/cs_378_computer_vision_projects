import cv2
import numpy as np
import pano_stitcher as ps

# Load source images
p1 = cv2.imread('my_panos/src/part1.jpg')
p2 = cv2.imread('my_panos/src/part2.jpg')
p3 = cv2.imread('my_panos/src/part3.jpg')

# Warp first image by the homography mapping
# the first image to the second image
p1_homography = ps.homography(p2, p1)
p1_warped, p1_origin = ps.warp_image(p1, p1_homography)

# Warp third image by the homography mapping
# the third image to the second image
p3_homography = ps.homography(p2, p3)
p3_warped, p3_origin = ps.warp_image(p3, p3_homography)

# Add alpha channel to second image
blue, green, red = cv2.split(p2)
alpha = np.zeros(green.shape, dtype=np.uint8)
alpha.fill(255)
p2 = cv2.merge([blue, green, red, alpha])


# Composite warped images and image in target plane
pano = ps.create_mosaic(
    [p1_warped, p2, p3_warped], [p1_origin, (0, 0), p3_origin])

cv2.imwrite('my_panos/pano.jpg', pano)
