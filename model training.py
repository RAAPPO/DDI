import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the features and target variable
X = pd.read_csv("G:/drug/src/features.csv")
y = pd.read_csv("G:/drug/src/target.csv").squeeze()

logging.info("Loaded features and target variable.")

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info("Split data into training and testing sets with stratification.")

# Reduce the dataset size for initial testing (optional)
# Uncomment the following lines to use a smaller subset of data
X_train, y_train = X_train.sample(frac=0.1, random_state=42), y_train.sample(frac=0.1, random_state=42)
X_test, y_test = X_test.sample(frac=0.1, random_state=42), y_test.sample(frac=0.1, random_state=42)
logging.info("Reduced dataset size for initial testing.")

# Initialize Random Forest model and its hyperparameters for tuning
model = RandomForestClassifier(random_state=42, n_jobs=-1)  # Use all available cores
params = {
    'n_estimators': [100, 150],  # Narrowed down for faster tuning
    'max_depth': [None, 10],  # Narrowed down for faster tuning
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Train model with hyperparameter tuning
logging.info("Training RandomForestClassifier with hyperparameter tuning.")
grid_search = GridSearchCV(model, params, cv=3, scoring='f1', n_jobs=-1)  # Use parallel processing
grid_search.fit(X_train, y_train.to_numpy())

best_model = grid_search.best_estimator_
logging.info(f"Best parameters for RandomForestClassifier: {grid_search.best_params_}")

# Save the best model
joblib.dump(best_model, "G:/drug/src/RandomForestClassifier_model.pkl")
logging.info("Saved best RandomForestClassifier model.")

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Store the metrics
evaluation_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc_score": roc_auc
}

# Save the evaluation metrics to a CSV file
evaluation_metrics_df = pd.DataFrame([evaluation_metrics], index=["RandomForestClassifier"])
evaluation_metrics_df.to_csv("G:/drug/src/RandomForestClassifier_evaluation_metrics.csv", index=True)

logging.info("Model training with hyperparameter tuning completed and model saved to disk.")
logging.info(f"Evaluation metrics: {evaluation_metrics}")
print("Model training with hyperparameter tuning completed and evaluation metrics saved.")
