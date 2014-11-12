"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy


def _track_ball(video):
    """
    Helper for all track_ball functions

    Loops through each frame of the video, isolates the moving object
    finds a rectangle that bounds the object, and returns a list of
    the coordinates of the corners of this rectangle for each frame
    """
    result = []

    ret, frame = video.read()

    while ret is True:
        # Take frame from video

        # Set up the Region of Interest for the object we want to track
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi,
                           numpy.array((0., 60., 32.)),
                           numpy.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

        # Convert from BGR to HSV color-space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Calculate the back projection of the histogram of the ball
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Applies a fixed-level threshold to each array element
        ret, thresh = cv2.threshold(dst, 127, 255, 0)

        # contours is a list of all the contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Gets the XYWH of each of the faces detected and appends them
        x, y, w, h = cv2.boundingRect(contours[0])
        result.append((x, y, x + w, y + h))

        ret, frame = video.read()

    return result


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.
    """
    return _track_ball(video)


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    return _track_ball(video)


def track_ball_3(video):
    """
    As track_ball_1, but for ball_3.mov.

    Requires going through the video once to find the size of the moving
    object, then again to track its location
    """
    result = []
    num_frames = 0
    fgbg = cv2.BackgroundSubtractorMOG()
    x, y, w, h = 0, 0, 0, 0
    avg_w, avg_h = 0, 0

    ret, frame = video.read()

    while ret is True:
        num_frames = num_frames + 1
        sub = fgbg.apply(frame)

        kernel = numpy.ones((5, 5), numpy.uint8)
        dilation = cv2.dilate(sub, kernel, iterations=1)
        ret, thresh = cv2.threshold(dilation, 127, 255, 0)

        # contours is a list of all the contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if (contours):
            x, y, w, h = cv2.boundingRect(contours[0])
            avg_w = avg_w + w
            avg_h = avg_h + h

        ret, frame = video.read()

    avg_w = avg_w / num_frames
    avg_h = avg_h / num_frames

    # Reset the video
    video.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 0)

    # take first frame of the video
    ret, frame = video.read()

    # setup initial location of window to track an object of the
    # size determined previously
    track_window = (avg_w, avg_w, avg_h, avg_h)
    x, y, w, h = track_window

    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       numpy.array((0., 60., 32.)),
                       numpy.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination criteria,
    # either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while ret is True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # draw it on image
        x, y, w, h = track_window
        result.append((x, y, x + w, y + h))

        ret, frame = video.read()

    return result


def track_ball_4(video):
    """As track_ball_1, but for ball_4.mov."""
    return _track_ball(video)


def track_face(video):
    """As track_ball_1, but for face.mov."""
    result = []

    # Detects face in a video stream
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Reads the video every frame and tries to detect a face
    if(video.isOpened()):
        ret, frame = video.read()
        while ret is True:

            # Converts stream color to grey
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Equalizes the histogram to get a better gray image
            gray = cv2.equalizeHist(gray)

            # Detects faces of different sizes using
            # cv2.CASCADE_SCALE_IMAGE
            # Minimum neighbors a candidate needs so it can saved is 6
            # minSize is (70, 70) to prevent artifacts
            faces = cascade.detectMultiScale(gray,
                                             scaleFactor=1.2,
                                             minNeighbors=6,
                                             minSize=(70, 70),
                                             flags =
                                             cv2.CASCADE_SCALE_IMAGE)

            # Gets the XYWH of each of the faces detected and appends them
            for (x, y, w, h) in faces:
                result.append(((x, y, x + w, y + h)))

            ret, frame = video.read()

    return result
