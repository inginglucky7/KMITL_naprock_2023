#import section#

import mediapipe as mp
import sys
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#video capture use openCV
cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistics:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistics.process(image)
        print(results.face_landmarks)

        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv.imshow('Test', image)
        if cv.waitKey(10) and 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()