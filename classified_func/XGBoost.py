import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset (change the file path accordingly)
df = pd.read_csv('../recognition/coords.csv')

# Separate features (x) and target (y)
x = df.drop('class', axis=1)
y = df['class']

# Initialize the XGBoost classifier
xgb = XGBClassifier(random_state=1234)

# Define a parameter distribution for hyperparameter tuning
param_dist = {
    'n_estimators': np.arange(50, 401, 50),  # Range of estimators
    'max_depth': np.arange(3, 7),  # Range of max_depth
    'learning_rate': [0.001, 0.01, 0.1, 0.2],  # Learning rate options
    'subsample': [0.6, 0.7, 0.8, 0.9],  # Subsample fraction
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Fraction of features used by trees
}

# Initialize StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# Initialize RandomizedSearchCV with XGBoost, parameter distribution, and cross-validation
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=10, cv=cv, scoring="accuracy", n_jobs=-1, random_state=1234)

# Fit RandomizedSearchCV to find the best hyperparameters
try:

    random_search.fit(x, y)

except Exception as e:
    print(e)
# Get the best XGBoost estimator from RandomizedSearchCV
best_xgb = random_search.best_estimator_

# Calculate cross-validation accuracy
cv_accuracy = cross_val_score(best_xgb, x, y, cv=cv, scoring='accuracy')

# Print the mean cross-validation accuracy
print("Cross-Validation Accuracy:", np.mean(cv_accuracy))

# Save the best XGBoost model to a file
with open('body_language_kFold_xgb.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)
