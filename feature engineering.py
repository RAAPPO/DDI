import pandas as pd
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the preprocessed data in chunks to handle memory constraints
chunksize = 100000  # Adjust the chunk size based on available memory
drugs_df = pd.read_csv("G:/drug/src/preprocessed_drugs.csv")
interactions_chunks = pd.read_csv("G:/drug/src/preprocessed_interactions.csv", chunksize=chunksize)

logging.info("Loaded preprocessed drugs data.")
logging.info("Processing interactions data in chunks.")

# Extract features for the model
# We will use a subset of features from both drugs involved in the interaction

# Features from the drug initiating the interaction
drug_features = [
    "average_mass", "monoisotopic_mass", "logP", "logS", 
    "Water Solubility", "Molecular Weight", "Monoisotopic Weight", 
    "Polar Surface Area (PSA)", "Refractivity", "Polarizability", 
    "Rotatable Bond Count", "H Bond Acceptor Count", "H Bond Donor Count", 
    "pKa (strongest acidic)", "pKa (strongest basic)", "Physiological Charge", 
    "Number of Rings", "Bioavailability", "state"
]

# Additional features to include if available
additional_features = [
    "drug_class", "mechanism_of_action", "target_proteins", "interaction_descriptors"
]

# Create feature columns for both drugs involved in the interaction
features_drug = [f"{feature}_drug" for feature in drug_features]
features_interaction = [f"{feature}_interaction" for feature in drug_features]

# Check for additional features and add them if available
for feature in additional_features:
    if feature in drugs_df.columns:
        features_drug.append(f"{feature}_drug")
        features_interaction.append(f"{feature}_interaction")

# Combine all feature columns
all_features = features_drug + features_interaction

# Initialize an empty DataFrame to store the processed chunks
features_df_list = []
target_df_list = []

# Variable to track the presence of both classes in the target variable
has_positive_class = False
has_negative_class = False

for chunk in interactions_chunks:
    logging.info("Processing a new chunk of interactions data.")

    # Check if the required columns are in the chunk
    missing_features = [feature for feature in all_features if feature not in chunk.columns]
    if missing_features:
        logging.warning(f"Missing features in the chunk: {missing_features}. Continuing without these features.")
        available_features = [feature for feature in all_features if feature in chunk.columns]
        X_chunk = chunk[available_features]
    else:
        X_chunk = chunk[all_features]
    
    y_chunk = chunk["interaction_description"].apply(lambda x: 1 if x else 0)
    
    # Update the class presence trackers
    if 1 in y_chunk.values:
        has_positive_class = True
    if 0 in y_chunk.values:
        has_negative_class = True

    # Check the distribution of the target variable for the current chunk
    target_distribution_chunk = y_chunk.value_counts()
    logging.info(f"Target variable distribution for the current chunk:\n{target_distribution_chunk}")

    # Append the processed chunk to the list
    features_df_list.append(X_chunk)
    target_df_list.append(y_chunk)

    logging.info("Processed and appended the current chunk of interactions data.")

# Concatenate all processed chunks into a single DataFrame
X = pd.concat(features_df_list, ignore_index=True)
y = pd.concat(target_df_list, ignore_index=True)

logging.info("Concatenated all processed chunks into a single DataFrame.")

# Check the distribution of the target variable after concatenating all chunks
target_distribution = y.value_counts()
logging.info(f"Target variable distribution after concatenating all chunks:\n{target_distribution}")

# Ensure the target variable has both classes
if not (has_positive_class and has_negative_class):
    logging.warning("The target variable lacks negative samples. Generating synthetic negative samples.")
    
    # Generating synthetic negative samples
    num_negative_samples = X.shape[0] // 2  # Generate as many negative samples as positive samples
    synthetic_negatives = X.sample(n=num_negative_samples, replace=True, random_state=42)
    synthetic_negatives_target = pd.Series(np.zeros(num_negative_samples), name='interaction_description')
    
    # Append synthetic negative samples to the dataset
    X = pd.concat([X, synthetic_negatives], ignore_index=True)
    y = pd.concat([y, synthetic_negatives_target], ignore_index=True)
    
    logging.info(f"Generated {num_negative_samples} synthetic negative samples.")

# Handle missing values
X = X.fillna(X.mean())
logging.info("Handled missing values in features data.")

# Save the features and target variable
X.to_csv("G:/drug/src/features.csv", index=False)
y.to_csv("G:/drug/src/target.csv", index=False)

logging.info("Feature engineering completed and saved to CSV files.")
print("Feature engineering completed successfully.")
