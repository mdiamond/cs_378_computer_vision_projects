import cv2
import numpy as np
import stereo
import sys


# Check the number of arguments
if len(sys.argv) < 3:
    print "MUST PASS IN AT LEAST 2 IMAGES TO RECTIFY"
    exit()

# Dispose of the first argument (script name)
sys.argv.remove(sys.argv[0])

# Use the last argument as the output filename
# for the PLY file, then remove it from the
# list of arguments
output_filename = sys.argv[len(sys.argv) - 1]
sys.argv.remove(sys.argv[len(sys.argv) - 1])

# Read in the two images, using the only remaining
# arguments as the filenames of the left and right images
image_left = cv2.imread(sys.argv[0])
image_right = cv2.imread(sys.argv[1])

# Calculate the disparity image
disparity_image = stereo.disparity_map(image_left, image_right)

# Construct a point cloud
pc = stereo.point_cloud(disparity_image, image_left, 10)

# Write out the point cloud
with open(output_filename, 'w') as f:
    f.write(pc)

# Write out the disparity image
with open('disparity_image.png', 'w') as f:
    f.write(disparity_image)
