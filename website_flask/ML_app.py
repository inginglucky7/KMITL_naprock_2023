import sys
import cv2
import mediapipe as mp
import cv2 as cv
import numpy as np
import pickle
import pandas as pd
import json
import os
import ffmpeg
import random
from flask import session
from PIL import Image, ImageFont, ImageDraw
from mediapipe import solutions
from PIL import Image
from mediapipe.python.solutions.pose import PoseLandmark
from moviepy.editor import VideoFileClip
sys.path.append("/naprock_classified/KMITL_naprock_2023/")

cap = cv.VideoCapture(0)
r_hand_row = []
l_hand_row = []
pose_row = []
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_holistic = solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(color=(80,110,10),thickness=0.5, circle_radius=0.5)

# detection open file and var zone
df = pd.read_csv('D:\\naprock_classified\\KMITL_naprock_2023\\AI\\recognition\\coords.csv')
numcoords = 0
test_x = df.values

avg_probs = {}
threshold_great = 80
threshold_good = 30
threshold_OK = 5

with open('D:/naprock_classified/KMITL_naprock_2023/ensemble_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

def prob_export(video):
    body_language_class = ""
    body_language_prob = ""
    avg_probs = {}
    text_score = ""
    cap_vid = cv.VideoCapture(video)
    frame_count = 0
    frames_with_landmarks = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap_vid.isOpened():
            try:
                ret, frame = cap_vid.read()
                if not ret:
                    print("Empty Camera!")
                    break
                image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                frame.flags.writeable = True
                image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

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
            
            if results.pose_landmarks and results.right_hand_landmarks and results.left_hand_landmarks:
                frames_with_landmarks.append((frame_count, frame))
            
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

                    # if body_language_class not in avg_probs:
                    #     avg_probs[body_language_class] = np.max(body_language_prob)
                    # else:
                    #     avg_probs[body_language_class] += np.max(body_language_prob)
                    
                    max_prob = np.max(body_language_prob)
                    
                    if body_language_class not in avg_probs and body_language_class != "Feet":
                        avg_probs[body_language_class] = (max_prob, text_score)
                    elif avg_probs[body_language_class][0] + max_prob / 7 >= threshold_great:
                        text_score = "GREAT"
                        avg_probs[body_language_class] = (avg_probs[body_language_class][0] + max_prob / 7, text_score)
                    elif threshold_good <= avg_probs[body_language_class][0] + max_prob / 7 < threshold_great:
                        text_score = "GOOD"
                        avg_probs[body_language_class] = (avg_probs[body_language_class][0] + max_prob / 7, text_score)
                    elif threshold_OK <= avg_probs[body_language_class][0] + max_prob / 7 < threshold_good:
                        text_score = "OK"
                        avg_probs[body_language_class] = (avg_probs[body_language_class][0] + max_prob / 7, text_score)
                    else:
                        text_score = "OK"
                        avg_probs[body_language_class] = (avg_probs[body_language_class][0] + max_prob / 7, text_score)
                    
                # Grab ear coords หาโคออของหูเพื่อเอาไปแปะ Not used
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
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    cap_vid.release()
    cv.destroyAllWindows()
    if frames_with_landmarks:
        path = "static/files"
        random_frame, selected_frame = random.choice(frames_with_landmarks)
        filename = f"selected_frame.jpg"
        file_path = os.path.join(path, filename)
        cv.imwrite(file_path, selected_frame)
        print(file_path)
    return avg_probs, file_path
    

def get_video_metadata(video_path):
    metadata = ffmpeg.probe(video_path)
    format_metadata = metadata['format']
    clip_duration = json.dumps(format_metadata["duration"])
    creation_time = json.dumps(format_metadata["tags"]["creation_time"])
    return clip_duration, creation_time

def generate_image(filename, duration, create_date, probs, frame_path):
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 16)
    draw.text((50, 50), f"Filename: {filename}", fill='black', font=font)
    draw.text((50, 75), f"Duration: {duration}", fill='black', font=font)
    draw.text((50, 100), f"Create Date: {create_date}", fill='black', font=font)
    draw.text((50, 125), "Probabilities:", fill='black', font=font)
    y_offset = 150
    
    for key, value in probs.items():
        draw.text((50, y_offset), f"{key}: {value[1]}", fill='black', font=font)
        y_offset += 25
        
    frame_img = Image.open(frame_path)
    frame_position = (50, y_offset)
    frame_img.thumbnail((300, 100)) 
    img.paste(frame_img, frame_position)
    return img