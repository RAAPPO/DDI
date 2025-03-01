import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
drugs_df = pd.read_csv("G:/drug/src/drugs.csv")
interactions_df = pd.read_csv("G:/drug/src/interactions.csv")

logging.info("Loaded drugs and interactions data.")

# Preprocess drugs data
# Handle missing values by filling them with empty strings
drugs_df.fillna("", inplace=True)

# Convert numerical columns to appropriate data types
numerical_columns = [
    "average_mass", "monoisotopic_mass", "logP", "logS", 
    "Water Solubility", "Molecular Weight", "Monoisotopic Weight", 
    "Polar Surface Area (PSA)", "Refractivity", "Polarizability", 
    "Rotatable Bond Count", "H Bond Acceptor Count", "H Bond Donor Count", 
    "pKa (strongest acidic)", "pKa (strongest basic)", "Physiological Charge", 
    "Number of Rings", "Bioavailability"
]
for col in numerical_columns:
    if col in drugs_df.columns:
        drugs_df[col] = pd.to_numeric(drugs_df[col], errors='coerce')

# Normalize numerical columns
drugs_df[numerical_columns] = drugs_df[numerical_columns].apply(lambda x: (x - x.mean()) / x.std())

logging.info("Handled missing values, converted data types, and normalized numerical columns in drugs data.")

# Encode categorical columns
categorical_columns = ["state"]
for col in categorical_columns:
    if col in drugs_df.columns:
        drugs_df[col] = drugs_df[col].astype("category").cat.codes

logging.info("Encoded categorical columns in drugs data.")

# Select relevant columns from the drugs DataFrame
relevant_columns = ["drugbank_id"] + numerical_columns + categorical_columns
drugs_df_relevant = drugs_df[relevant_columns]

# Merge interactions with drug features
interactions_merged = interactions_df.merge(drugs_df_relevant, left_on="drugbank_id", right_on="drugbank_id", how="left")
interactions_merged = interactions_merged.merge(drugs_df_relevant, left_on="interaction_id", right_on="drugbank_id", how="left", suffixes=("_drug", "_interaction"))

# Drop any columns that are no longer necessary or duplicated
interactions_merged.drop(columns=["drugbank_id_interaction"], inplace=True)

logging.info("Merged interactions with drug features.")

# Save the preprocessed data
drugs_df.to_csv("G:/drug/src/preprocessed_drugs.csv", index=False)
interactions_merged.to_csv("G:/drug/src/preprocessed_interactions.csv", index=False)

logging.info("Data preprocessing completed and saved to CSV files.")
