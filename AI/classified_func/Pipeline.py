import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("D:/naprock_classified/KMITL_naprock_2023/AI/recognition/coords.csv")

x = df.drop('class', axis=1)
y = df['class']

logistic_classifier = LogisticRegression(random_state=1234, max_iter=1000, solver="lbfgs")
ridge_classifier = RidgeClassifier(random_state=1234, max_iter=1000)
rf_classifier = RandomForestClassifier(random_state=1234)
xgb_classifier = XGBClassifier(random_state=1234)

estimators = [('logistic', logistic_classifier), ('ridge', ridge_classifier), ('randomforest', rf_classifier), ('xgboost', xgb_classifier)]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier(random_state=1234))

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', StandardScaler()),
    ('classifier', stacking_classifier)
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

with open('ensemble_classifier.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
