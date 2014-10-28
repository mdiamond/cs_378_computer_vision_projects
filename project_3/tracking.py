"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.
    """
    result = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.BackgroundSubtractorMOG()
    if(video.isOpened()):
        hasFrame = True
        while(hasFrame):
            ret, frame = video.read()
            if(ret != 0):
                fgmask = fgbg.apply(frame)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, hierarchy = cv2.findContours(thresh,
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                result.append((x, y, x + w, y + h))
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                hasFrame = False
    cv2.destroyAllWindows()
    return result


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    result = []

    # take first frame of the video
    ret, frame = video.read()
    
    # setup initial location of window
    r, h, c, w = 65, 65, 65, 65  # simply hardcoded the values
    track_window = (c, r, w, h)
    x, y, w, h = track_window
    result.append((x, y, x + w, y + h))

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       numpy.array((0., 60., 32.)),
                       numpy.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while(1):
        ret, frame = video.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # draw it on image
            x, y, w, h = track_window
            result.append((x, y, x + w, y + h))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('img2', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'): # Escape key
                break

        else:
            break
    cv2.destroyAllWindows()
    return result


def track_ball_3(video):
    """As track_ball_1, but for ball_2.mov."""
    result = []

    # take first frame of the video
    ret, frame = video.read()
    
    # setup initial location of window
    r, h, c, w = 65, 65, 65, 65  # simply hardcoded the values
    track_window = (c, r, w, h)
    x, y, w, h = track_window
    result.append((x, y, x + w, y + h))

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       numpy.array((0., 60., 32.)),
                       numpy.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while(1):
        ret, frame = video.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # draw it on image
            x, y, w, h = track_window
            result.append((x, y, x + w, y + h))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('img2', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'): # Escape key
                break

        else:
            break
    cv2.destroyAllWindows()
    return result


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    result = []

    # take first frame of the video
    ret, frame = video.read()
    
    # setup initial location of window
    r, h, c, w = 65, 65, 65, 65  # simply hardcoded the values
    track_window = (c, r, w, h)
    x, y, w, h = track_window
    result.append((x, y, x + w, y + h))

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi,
                       numpy.array((0., 60., 32.)),
                       numpy.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while(1):
        ret, frame = video.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # draw it on image
            x, y, w, h = track_window
            result.append((x, y, x + w, y + h))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('img2', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'): # Escape key
                break

        else:
            break
    cv2.destroyAllWindows()
    return result


def track_face(video):
    """As track_ball_1, but for face.mov."""
    result = []
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if(video.isOpened()):
        hasFrame = True
        while(hasFrame):
            ret, frame = video.read()
            if(ret != 0):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                faces = cascade.detectMultiScale(gray,
                                                 scaleFactor=1.2,
                                                 minNeighbors=6,
                                                 minSize=(70, 70),
                                                 flags =
                                                 cv2.CASCADE_SCALE_IMAGE)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    result.append(((x, y, x + w, y + h)))

                cv2.imshow('Video', frame)

                if (cv2.waitKey(1) & 0xff) == 27:
                    break

            else:
                hasFrame = False

    cv2.destroyAllWindows()
    return result
