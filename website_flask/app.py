import os
import sys
import json
import ast
import io
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import FileAllowed, FileField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from ML_app import prob_export, get_video_metadata, generate_image
sys.path.append("/naprock_classified/KMITL_naprock_2023/")

# In case path is missing
# cur_path = os.path.dirname(__file__)
# model_path = os.path.relpath('..\\AI\\Model\\ensemble_classifier.pkl', cur_path)

app = Flask(__name__)
app.config["FLASK_APP"] = "app.py"
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/files"

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired(), FileAllowed(['mp4', 'mkv'], 'Video Only!')])
    submit = SubmitField("Upload File")

@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    passed_filename = request.args.get('passed_filename')
    passed_duration = request.args.get('passed_duration')
    passed_create_date = request.args.get('passed_create_date')
    passed_probs_str = request.args.get('passed_probs', '{}')
    passed_probs = ast.literal_eval(passed_probs_str) if isinstance(passed_probs_str, str) else passed_probs_str
    return render_template('videoPage.html', filename=passed_filename, duration=passed_duration, create_date=passed_create_date, probs=passed_probs)

@app.route('/upload', methods = ["GET", "POST"])
def video():
    form = UploadFileForm()
    if(form.validate_on_submit()):
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config["UPLOAD_FOLDER"], secure_filename(file.filename)))
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        duration, create_date = get_video_metadata(file_path)
        probs_data, frame_pic = prob_export(file_path)
        session['passed_filename'] = filename
        session['passed_duration'] = duration
        session['passed_create_date'] = create_date
        session['passed_probs'] = probs_data
        session['passed_frame_pic'] = frame_pic
        # video_file = VideoFileClip(file_path) ----> Read pre_process()
        # return render_template('videoPage.html', filename=filename, duration=duration, create_date=create_date, probs_data=probs_data)
        return redirect(url_for('video_feed', passed_filename=filename, passed_duration=duration, passed_create_date=create_date, passed_probs=probs_data))
    return render_template('upload.html', form=form)

@app.route('/img_export', methods=['POST'])
def img_export():
    filename = session.get('passed_filename')
    duration = session.get('passed_duration')
    create_date = session.get('passed_create_date')
    probs_data = session.get('passed_probs')
    frame_pic = session.get('passed_frame_pic')
    if filename and duration and create_date and probs_data:
        img = generate_image(filename, duration, create_date, probs_data, frame_pic)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'exported_image.jpg')
        img.save(img_path)
        return send_file(img_path, mimetype='image/jpeg', as_attachment=True)
    else:
        return jsonify({'error': 'Data not found in session'}), 404

if(__name__ == "__main__"):
    app.run(debug=True)
