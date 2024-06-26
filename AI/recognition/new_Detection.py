import cv2
import mediapipe as mp
import cv2 as cv
import csv
import numpy as np
import os
import pickle
import pandas as pd
from mediapipe import solutions
from mediapipe.python.solutions.pose import PoseLandmark

cap = cv.VideoCapture("D:\\naprock_classified\\KMITL_naprock_2023\\website_flask\\static\\files\\VID_20230927_182621.mp4")
r_hand_row = []
l_hand_row = []
pose_row = []
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_holistic = solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(color=(80,110,10),thickness=0.5, circle_radius=0.5)

df = pd.read_csv('D:/naprock_classified/KMITL_naprock_2023/AI/recognition/coords.csv')
numcoords = 0
test_x = df.values

avg_probs = {}

threshold = 0.8

with open('D:/naprock_classified/KMITL_naprock_2023/ensemble_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

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

            # mp_drawing.draw_landmarks(
            #     frame,
            #     results.face_landmarks,
            #     mp_holistic.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style()
            # )

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

        except():
            pass

        # if results.face_landmarks:
        #     numcoords += len(results.face_landmarks.landmark)
        #     face = results.face_landmarks.landmark
        #     face_row = list(
        #         np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
        
        try:
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            if results.right_hand_landmarks:
                r_hand = results.right_hand_landmarks.landmark
                r_hand_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in r_hand]).flatten())

            if results.left_hand_landmarks:
                l_hand = results.left_hand_landmarks.landmark
                l_hand_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in l_hand]).flatten())
            else:
                pose = None

            if pose is not None:
                row = pose_row + r_hand_row + l_hand_row
                X_data = pd.DataFrame([row])
                # print(X_data.values)
                body_language_class = model.predict(X_data.values)[0]
                body_language_prob = model.predict_proba(X_data)[0]
                print(body_language_class, np.argmax(body_language_prob), np.max(body_language_prob))
                
                if body_language_class not in avg_probs:
                    avg_probs[body_language_class] = body_language_prob
                else:
                    avg_probs[body_language_class] += body_language_prob

            # Grab ear coords
                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640, 480]).astype(int))

                cv.rectangle(frame,
                              (coords[0], coords[1] + 5),
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                              (245, 117, 16), -1)
                cv.putText(frame, body_language_class, coords,
                            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

                # Get status box
                cv.rectangle(frame, (0, 0), (250, 60), (245, 117, 16), -1)

                # Display Class
                cv.putText(frame, 'CLASS'
                            , (95, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(frame, body_language_class.split(' ')[0]
                            , (90, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

                # Display

                cv2.putText(frame, 'PROB'
                            , (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                            , (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                
        except Exception as e:
            print(e)
            pass

        cv.imshow("Frame", frame)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

num_frames = len(avg_probs)
for class_name in avg_probs:
    avg_probs[class_name] /= num_frames
    
print("Avg : ")
for class_name, avg_prob in avg_probs.items():
    print(f"{class_name} : {avg_prob}")