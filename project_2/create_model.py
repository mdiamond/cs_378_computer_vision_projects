import cv2
import numpy as np
import stereo
import sys


if len(sys.argv) < 3:
    print "MUST PASS IN AT LEAST 2 IMAGES TO RECTIFY"
    exit()

sys.argv.remove(sys.argv[0])

output_filename = sys.argv[len(sys.argv) - 1]
sys.argv.remove(sys.argv[len(sys.argv) - 1])

image_left = cv2.imread(sys.argv[0])
image_right = cv2.imread(sys.argv[1])

disparity_image = stereo.disparity_map(image_left, image_right)

pc = stereo.point_cloud(disparity_image, image_left, 10)

with open(output_filename, 'w') as f:
    f.write(pc)

with open('disparity_image.png', 'w') as f:
    f.write(disparity_image)
