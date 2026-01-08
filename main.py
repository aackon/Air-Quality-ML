import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data_set = pd.read_csv("updated_pollution_dataset.csv")

# Reclassify air quality
data_set['Air Quality'] = data_set['Air Quality'].replace({
    'Good': 'Good',        # Keep as Good
    'Moderate': 'Good',    # Reclassify Moderate as Good
    'Poor': 'Hazardous',   # Keep as Hazardous
    'Hazardous': 'Hazardous'
})

# One-hot encode the target variable
encoded_data = pd.get_dummies(data_set['Air Quality'], prefix='Air Quality').astype(int)
encoded_data = encoded_data.drop("Air Quality_Hazardous", axis=1)  # Drop one target variable for dimension reduction
data_set = data_set.drop(['Air Quality'], axis=1)

# Define features and target
X = data_set
y = encoded_data

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Store results for comparison
results = {}

# Logistic Regression with Cross-Validation
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_params = {'C': [0.01, 0.1, 1, 10, 100]}
log_cv = GridSearchCV(log_model, log_params, cv=5)
log_cv.fit(X_train, y_train)
y_pred_log = log_cv.predict(X_test)
results['Logistic Regression'] = accuracy_score(y_test, y_pred_log)
print("Best Logistic Regression Parameters:", log_cv.best_params_)

# Decision Tree with Cross-Validation
dt_model = DecisionTreeClassifier(random_state=42)
dt_params = {'max_depth': [3, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
dt_cv = GridSearchCV(dt_model, dt_params, cv=5)
dt_cv.fit(X_train, y_train)
y_pred_dt = dt_cv.predict(X_test)
results['Decision Tree'] = accuracy_score(y_test, y_pred_dt)
print("Best Decision Tree Parameters:", dt_cv.best_params_)

# Random Forest with Cross-Validation
rf_model = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}
rf_cv = GridSearchCV(rf_model, rf_params, cv=5)
rf_cv.fit(X_train, y_train)
y_pred_rf = rf_cv.predict(X_test)
results['Random Forest'] = accuracy_score(y_test, y_pred_rf)
print("Best Random Forest Parameters:", rf_cv.best_params_)

# K-Nearest Neighbors with Cross-Validation
knn_model = KNeighborsClassifier()
knn_params = {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']}
knn_cv = GridSearchCV(knn_model, knn_params, cv=5)
knn_cv.fit(X_train, y_train)
y_pred_knn = knn_cv.predict(X_test)
results['KNN'] = accuracy_score(y_test, y_pred_knn)
print("Best KNN Parameters:", knn_cv.best_params_)

# Display results
print("\nModel Performance Comparison:")
for model, acc in results.items():
    print(f"{model}: Accuracy = {acc:.4f}")

# Detailed Classification Reports
print("\nDetailed Classification Reports:")
models = {
    "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "KNN": y_pred_knn
}

for model_name, predictions in models.items():
    print(f"\n{model_name}:")
    print(classification_report(y_test, predictions))
