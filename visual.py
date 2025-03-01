import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Load the evaluation metrics for all models
random_forest_metrics = pd.read_csv("G:/drug/src/RandomForestClassifier_evaluation_metrics.csv", index_col=0)
svc_metrics = pd.read_csv("G:/drug/src/SVC_evaluation_metrics.csv", index_col=0)
logistic_regression_metrics = pd.read_csv("G:/drug/src/LogisticRegression_evaluation_metrics.csv", index_col=0)

# Combine all metrics into a single DataFrame
evaluation_metrics_df = pd.concat([random_forest_metrics, svc_metrics, logistic_regression_metrics])

# List of metrics to plot
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc_score"]

# Stacked Bar Chart
def plot_stacked_bar_chart(df, metrics, title):
    df[metrics].plot(kind='barh', stacked=True, figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Metric Value')
    plt.ylabel('Models')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"G:/drug/src/{title.replace(' ', '_')}.png")
    plt.close()

# Radar Chart
def plot_radar_chart(df, metrics, labels, title):
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    stats = df[metrics].values
    
    angles += angles[:1]
    stats = np.concatenate((stats, stats[:, :1]), axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, row in enumerate(stats):
        ax.plot(angles, row, label=labels[idx])
        ax.fill(angles, row, alpha=0.25)
    
    plt.xticks(angles[:-1], metrics)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"G:/drug/src/{title.replace(' ', '_')}_radar.png")
    plt.close()

# Plot Stacked Bar Chart for Evaluation Metrics
plot_stacked_bar_chart(evaluation_metrics_df, metrics_to_plot, 'Comparison of Evaluation Metrics Across Models')

# Plot Radar Chart for Evaluation Metrics
plot_radar_chart(evaluation_metrics_df, metrics_to_plot, evaluation_metrics_df.index, 'Radar Chart of Evaluation Metrics')

# Load the preprocessed drug properties data
preprocessed_data = pd.read_csv("G:/drug/src/preprocessed_drugs.csv")  # Replace with actual preprocessed data path

# Map feature names to drug properties
property_mapping = {
    'average_mass': 'Average Mass',
    'monoisotopic_mass': 'Monoisotopic Mass',
    'logP': 'LogP',
    'logS': 'LogS',
    'molecular_weight': 'Molecular Weight',
    'polar_surface_area': 'Polar Surface Area',
    'num_rotable_bonds': 'Number of Rotatable Bonds',
    'num_h_donors': 'Number of Hydrogen Bond Donors',
    'num_h_acceptors': 'Number of Hydrogen Bond Acceptors',
    # Add more mappings as needed
}

# Rename columns based on property_mapping
preprocessed_data.rename(columns=property_mapping, inplace=True)

# Select only numeric columns for variance calculation
numeric_data = preprocessed_data.select_dtypes(include='number')

# Get importance of drug properties based on statistical measures (e.g., variance)
property_importances = numeric_data.var().sort_values(ascending=False)

# Create a DataFrame for importance
importance_df = property_importances.reset_index()
importance_df.columns = ['Property', 'Importance']

# Plot importance of drug properties
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Property', data=importance_df)
plt.title('Importance of Drug Properties in Predicting DDIs')
plt.xlabel('Importance (Variance)')
plt.ylabel('Drug Property')
plt.tight_layout()
plt.savefig("G:/drug/src/drug_property_importance.png")
plt.close()

# Save the drug property importances to a CSV file
importance_df.to_csv("G:/drug/src/drug_property_importances.csv", index=False)

# Plotting the correlation heatmap for the preprocessed data
plt.figure(figsize=(12, 10))
corr = numeric_data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix Heatmap of Drug Properties')
plt.tight_layout()
plt.savefig("G:/drug/src/correlation_heatmap.png")
plt.close()

print("Visualizations for evaluation metrics, drug property importance, and correlation heatmap completed and saved as images.")
