import cv2
import pano_stitcher

b1 = pano_stitcher.b1
b2 = pano_stitcher.b2
homography = pano_stitcher.homography(b1, b2)
res, upper_left = pano_stitcher.warp_image(b2, homography)
