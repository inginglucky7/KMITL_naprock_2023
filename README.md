# Mediapipe Scikit-Learn Pose Classification
****************************************************************
This is a simple pose classification model that uses Google's Mediapipe and Scikit-Learn to train AI and draws landmarks
on human bodies to make predictions of pose and gestures for medical purposes.

Our goal is to make pose classification for physical therapy in limited situations and areas. This is for doctor who wants to gives the patient
a physical therapy test before meeting the doctor.

## Contents
- **_[Installation](#installation)_**<br/>
- **_[Google-Colab](#google-colab)_**
***
## Installation

This is packaged that you need to install if you want to test on the windows.
```cs
pip install mediapipe
pip install scikit-learn==1.3.0
pip install xgboost==2.00
pip install imgaug
pip install opencv-python
pip install numpy
```

you can run these following files to create data for training models.
- [writedowncsv.py](https://github.com/inginglucky7/KMITL_naprock_2023/blob/main/recognition/writedowncsv.py) to create a CSV file for stores data of landmarks.
- Then you must select files formatted that you want to write to the CSV file.
  - [createExDataFromPicture.py](https://github.com/inginglucky7/KMITL_naprock_2023/blob/main/recognition/createExDataFromPicture.py) creates a dataset from picture.
  - [createExDataFromVOD.py](https://github.com/inginglucky7/KMITL_naprock_2023/blob/main/recognition/createExDataFromVOD.py) creates a dataset from video.
- Run the [Pipeline.py](https://github.com/inginglucky7/KMITL_naprock_2023/blob/main/classified_func/Pipeline.py) for creates a model that we'll use for detecting the body. <br/>
And you can rename the ```.pkl``` file to other name that you want. <br/>
```cs
with open('YOUR_MODEL_NAME.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
```
- Then you can simply run the [detection.py](https://github.com/inginglucky7/KMITL_naprock_2023/blob/main/recognition/detection.py). That'll open your camera webcam and record your body and estimate the probability of the pose that you fit the model.
You can change the ```.pkl``` file to the same name of the ```YOUR_MODEL_NAME.pkl``` file in the section below.
```cs
with open('YOUR_MODEL_NAME.pkl', 'rb') as f:
    model = pickle.load(f)
```
- if you want to change camera please change it this section.
```cap = cv.VideoCapture(YOUR_CAMERA)``` (```0``` is first camera capture) for now this can use only one camera for one detection session.

### Example Result
<img alt="Butterfly_Hug" src="https://cdn.discordapp.com/attachments/1031263201763016704/1156171157825916979/Screenshot_2023-09-26_163050.jpg?ex=6513ffcc&is=6512ae4c&hm=1a783820ea3bf7668d93daeba9a1829ec34017fcfc699aa80947d92e3423be60&"/>
<img alt="Thighs" src="https://cdn.discordapp.com/attachments/1031263201763016704/1156171158065008640/Screenshot_2023-09-26_163130.jpg?ex=6513ffcc&is=6512ae4c&hm=a5420ce12f3eca195e7e1184e2ca28b9e909324790da6e3bd4e7db723c0f40e8&"/>

********************************

## Google Colab
**Detecting from user uploaded video and then processing the probabilities of all poses.**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CYWjo0ZJ6BGQCOEu_FhnnO0NaiT74JdF?usp=sharing])
****
### Remarks
This project is for education and training purposes between **[Faculty of Information-technology](https://www.it.kmitl.ac.th/en/)** and **[Faculty of Medicine](https://www.facebook.com/medkmitl/)** of **[King Mongkut's Institute of Technology Ladkrabang](https://www.kmitl.ac.th/)** 
