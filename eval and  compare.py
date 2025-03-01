import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the features and target variable for the test dataset
X_test = pd.read_csv("G:/drug/src/features_test.csv")
y_test = pd.read_csv("G:/drug/src/target_test.csv").squeeze()

logging.info("Loaded features and target variable for the test dataset.")

# Load the training data to get the feature names and statistics
X_train = pd.read_csv("G:/drug/src/features.csv")
train_stats = X_train.describe().transpose()

# Align the test data columns with the training data columns
X_test = X_test.reindex(columns=X_train.columns)

# Fill the missing values in the test data with the mean values from the training data
for column in X_train.columns:
    if column not in X_test.columns or X_test[column].isnull().all():
        X_test[column] = train_stats.loc[column, 'mean']

# Handle remaining missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_test = imputer.fit_transform(X_test)

# Convert to numpy array for model compatibility
logging.info("Handled missing values in the test dataset.")

# List of models to evaluate
model_files = ["G:/drug/src/RandomForestClassifier_model.pkl", "G:/drug/src/SVC_model.pkl", "G:/drug/src/LogisticRegression_model.pkl"]
model_names = ["RandomForestClassifier", "SVC", "LogisticRegression"]

# Initialize a dictionary to store evaluation metrics
evaluation_metrics = {}

# Evaluate each model
for model_file, model_name in zip(model_files, model_names):
    logging.info(f"Evaluating {model_name}.")
    
    # Load the model
    model = joblib.load(model_file)
    
    # Make predictions on the test dataset
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Store the metrics
    evaluation_metrics[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc_score": roc_auc
    }
    
    logging.info(f"Evaluation metrics for {model_name}: {evaluation_metrics[model_name]}")

# Save the evaluation metrics to a CSV file
evaluation_metrics_df = pd.DataFrame(evaluation_metrics).T
evaluation_metrics_df.to_csv("G:/drug/src/evaluation_metrics.csv", index=True)

# Determine the best model based on F1-score
best_model_name = evaluation_metrics_df['f1_score'].idxmax()
best_model_metrics = evaluation_metrics_df.loc[best_model_name]

logging.info(f"The best model based on F1-score is {best_model_name} with the following metrics: {best_model_metrics}")

# Save the best model name and its metrics
best_model_info = pd.DataFrame({
    "Best Model": [best_model_name],
    "Metrics": [best_model_metrics.to_dict()]
})
best_model_info.to_csv("G:/drug/src/best_model_info.csv", index=False)

logging.info("Evaluation and comparison completed. Best model identified and saved.")
print("Evaluation and comparison completed. Best model identified and saved.")
