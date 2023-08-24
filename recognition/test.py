import mediapipe as mp
import cv2 as cv
import csv
import numpy as np
import os
from mediapipe import solutions
from mediapipe.python.solutions.pose import PoseLandmark

cap = cv.VideoCapture(0)

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_holistic = solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(color=(80,110,10),thickness=1, circle_radius=1)
class_name = "Sit"

numcoords = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as holistic:
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print("Empty Camera frame")
                break

            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            frame.flags.writeable = True
            image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style()
            )

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style()
            )

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style()
            )

        except():
            pass

        if results.face_landmarks:
            numcoords += len(results.face_landmarks.landmark)

        if results.pose_landmarks:
            numcoords += len(results.pose_landmarks.landmark)

        if results.right_hand_landmarks:
            numcoords += len(results.right_hand_landmarks.landmark)

        if results.left_hand_landmarks:
            numcoords += len(results.left_hand_landmarks.landmark)

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

        except:
            pass

        cv.imshow("Frame", cv.flip(frame, 1))
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

# Release the capture and close all windows
cap.release()
cv.destroyAllWindows()

# Calculate header based on the total numcoords
header = ['class'] + [f'x{i}' for i in range(1, numcoords + 1)] + \
         [f'y{i}' for i in range(1, numcoords + 1)] + \
         [f'z{i}' for i in range(1, numcoords + 1)] + \
         [f'v{i}' for i in range(1, numcoords + 1)]

# Write the collected data to a CSV file
csv_filename = 'coords.csv'
with open(csv_filename, mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)
