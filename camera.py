from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
from playsound import playsound
import time
import dlib
import _thread
import cv2


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    # dist.euclidean computes the Euclidean distance between two 1-D arrays.

    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    eye_aspect_ratio = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return eye_aspect_ratio

# frames the eye must be below the threshold
EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 30

# initialize the frame counter to count how much the driver's eyes will be closed
COUNTER = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor that I got from Google as a dataset to use in this project
print("[+] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("[+] facial landmark predictor loaded successfully ✓")
video_stram = VideoStream(src=0).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
    
    # after counting 100 frames while the driver's eyes 
    # are closed the alert gets played to wake them up
    if COUNTER == 80:
        playsound('alert.wav')
        # and the counter goes to the half to to make the
        # time shorter for checking if the driver is still sleeping
        COUNTER = 40

    # grab the frames , resize them, and convert them to 
    # grayscale channels for better detection and recognetion
    frame = video_stram.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = detector(gray, 0)

    # loop over all faces which got detected
    for face in faces:

        # determine the facial landmarks for the face region, then
        shape = predictor(gray, face)
        
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[left_eye_start:left_eye_end]
        rightEye = shape[right_eye_start:right_eye_end]
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEyeAspectRatio + rightEyeAspectRatio)

        # compute the convex hull for the left and right eye, then
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # visualize each of the eyes
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the frame counter to
        # calculate this frame as an eye closed moment
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Eye: {}".format("close"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, "E.A.R.: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            COUNTER = COUNTER + 1


        # otherwise, the eye aspect ratio is not below the blink
        # threshold we restart the counter because this is a sign
        # that the driver is awake
        else:
            cv2.putText(frame, "Eye: {}".format("open"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, "E.A.R.: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            COUNTER = 0

    # show the frame as pop-up output video stream
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop 
    # and to quit from the application
    if key == ord("q"):
        break

# cleaning up the windows we opened
cv2.destroyAllWindows()
video_stram.stop()