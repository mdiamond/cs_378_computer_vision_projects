import cv2
import numpy as np
import pano_stitcher as ps

b1 = ps.b1
b2 = ps.b2
b3 = ps.b3

b1_homog = ps.homography(b2, b1)
b1_warped, b1_origins = ps.warp_image(b1, b1_homog)
print 'origins for warped b1:', b1_origins

cv2.imwrite('b1w.png', b1_warped)

b2_origins = (0, 0)

b3_homog = ps.homography(b2, b3)
b3_warped, b3_origins = ps.warp_image(b3, b3_homog)
print 'origins for warped b3:', b3_origins
cv2.imwrite('b3w.png', b3_warped)

blue, green, red = cv2.split(b2)
alpha = np.zeros(green.shape, dtype=np.uint8)
alpha.fill(255)
b2 = cv2.merge([blue, green, red, alpha])

pano = ps.create_mosaic(
    [b1_warped, b2, b3_warped], [b1_origins, b2_origins, b3_origins])
cv2.imwrite('pano_test.png', pano)
