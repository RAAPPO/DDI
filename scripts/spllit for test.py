import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
X = pd.read_csv("G:/drug/src/features.csv")
y = pd.read_csv("G:/drug/src/target.csv").squeeze()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the test set
X_test.to_csv("G:/drug/src/features_test.csv", index=False)
y_test.to_csv("G:/drug/src/target_test.csv", index=False)

print("Dataset split into training and testing sets, and test set saved.")
