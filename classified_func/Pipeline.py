import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

# Load your dataset (change the file path accordingly)
df = pd.read_csv('../recognition/coords.csv')

# Separate features (x) and target (y)
class_label_mapping = {
    'Arm': 0,
    'Calves': 1,
    'Feet': 2,
    'Fist': 3,
    'HDown': 4,
    'Thighs': 5
}

x = df.drop('class', axis=1)
y = df['class']
y = y.map(class_label_mapping)

# Define preprocessing steps
numeric_features = x.columns  # Assuming all columns are numeric
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)])

# Define classifiers
classifiers = [
    ('logistic', LogisticRegression(random_state=1234)),
    ('ridge', RidgeClassifier(random_state=1234)),
    ('xgboost', XGBClassifier(random_state=1234)),
    ('randomforest', RandomForestClassifier(random_state=1234))
]

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifiers)
])

param_dist = {
    'classifier__logistic__C': [0.001, 0.01, 0.1, 1.0],
    'classifier__ridge__alpha': [0.1, 1.0, 10.0],
    'classifier__xgboost__n_estimators': np.arange(50, 401, 50),
    'classifier__xgboost__max_depth': np.arange(3, 7),
    'classifier__randomforest__n_estimators': np.arange(50, 401, 50),
    'classifier__randomforest__max_depth': np.arange(3, 7),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=cv, scoring="accuracy", n_jobs=-1, random_state=1234)

random_search.fit(x, y)

best_pipeline = random_search.best_estimator_

cv_accuracy = cross_val_score(best_pipeline, x, y, cv=cv, scoring='accuracy')

print("Cross-Validation Accuracy:", np.mean(cv_accuracy))

with open('body_language_kFold_pipeline.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)