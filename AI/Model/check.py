import pickle
import pathlib
path_vid = "D:\\naprock_classified\\KMITL_naprock_2023\\website_flask\\static\\files\\VID_20231013_021137.mp4"

def check_model():    
    with open("D:/naprock_classified/KMITL_naprock_2023/ensemble_classifier.pkl", "rb") as f:
        model = pickle.load(f)
        print(type(model))
        print(hasattr(model, "predict"))
    return model

def check_model2():    
    with open("D:/naprock_classified/KMITL_naprock_2023/AI/Model/body_language_kFold_xgb.pkl", "rb") as f:
        model = pickle.load(f)
        print(type(model))
        print(hasattr(model, "predict"))
    return model

def predict():
    pathlib.Path("VID_20231013_021137.mp4").parent.resolve()
    return

predict()