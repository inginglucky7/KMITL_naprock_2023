import pickle
import os
import cv2
import sys
from mediapipe import solutions
from flask import Flask, render_template, Response, request, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import FileAllowed, FileRequired, FileField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from moviepy.editor import VideoFileClip
from ML_app import prob_export
sys.path.append("/naprock_classified/KMITL_naprock_2023/")

# In case path is missing
# cur_path = os.path.dirname(__file__)
# model_path = os.path.relpath('..\\AI\\Model\\ensemble_classifier.pkl', cur_path)

app = Flask(__name__)
app.config["FLASK_APP"] = "app.py"
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/files"

model = pickle.load(open('..\\AI\\Model\\ensemble_classifier.pkl', 'rb'))

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired(), FileAllowed(['mp4', 'mkv'], 'Video Only!')])
    submit = SubmitField("Upload File")

# In case want to resize Vid Size (Can use ffmpeg with this to change bitrate but takes time to write new Vid)
def pre_process(video):
    new_width = 1280
    new_height = 720
    resize_video = video.resize((new_width, new_height))
    return resize_video
    
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")

@app.route('/upload', methods = ["GET", "POST"])
def video():
    form = UploadFileForm()
    if(form.validate_on_submit()):
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config["UPLOAD_FOLDER"], secure_filename(file.filename)))
        filename = secure_filename(file.filename)
        file_path = os.path.abspath(filename)
        # video_file = VideoFileClip(file_path) ----> Read pre_process()
        video_cap = cv2.VideoCapture(file_path)
        return render_template('videoPage.html', filename=filename)
    return render_template('upload.html', form=form)

@app.route('/predict', methods = ["POST"])
def running():
    req = request.get_json(force=True)
    data = req["data"]
    response = {'response' : 'data receive, ' + data + '!'}
    return jsonify(response)

if(__name__ == "__main__"):
    app.run(debug=True)
