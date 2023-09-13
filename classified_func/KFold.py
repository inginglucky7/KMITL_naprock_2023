import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint
import pickle

df = pd.read_csv('../recognition/coords.csv')

x = df.drop('class', axis=1)
y = df['class']

rf = RandomForestClassifier(random_state=1234,
                            min_samples_leaf=5,
                            max_depth=10,
                            min_samples_split=10,
                            max_features='sqrt'
                            )

param_dist = {
    'n_estimators': randint(50, 400),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 8)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

#Gridsearch
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=cv, scoring="accuracy", n_jobs=-1, random_state=1234)

# grid_search.fit(x, y)
random_search.fit(x, y)

best_rf = random_search.best_estimator_

cv_accuracy = cross_val_score(best_rf, x, y, cv=cv, scoring='accuracy')
print("Cross-Validation Accuracy:", np.mean(cv_accuracy))

feature_importances = best_rf.feature_importances_
threshold = 0.01
selected_features = x.columns[feature_importances > threshold]
x_selected = x[selected_features]

with open('body_language_kFold.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
