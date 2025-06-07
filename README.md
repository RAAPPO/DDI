# 💊 Predicting Drug-Drug Interactions Using Machine Learning

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)  
[![Preprint](https://img.shields.io/badge/Read-Preprint-orange)](https://www.researchsquare.com/article/rs-6137371/v1)  
[![LinkedIn](https://img.shields.io/badge/Author-RAAPPO-blue)](https://www.linkedin.com/in/raappo/)

---

## 🚀 Project Overview

This repository contains code and data to **predict harmful Drug-Drug Interactions (DDIs)** leveraging machine learning models trained on features from the DrugBank database.

DDIs are a significant cause of adverse drug reactions (ADRs), often leading to severe health risks or hospitalization. This project aims to proactively identify these interactions to improve patient safety and aid pharmacovigilance.

---

## 🎯 Key Features

- **Dataset:** DrugBank v5.1.1 (drug properties and interactions)  
- **Models:** Logistic Regression, Support Vector Classifier (SVC), Random Forest  
- **Best Model:** Logistic Regression (F1-score ~0.80)  
- **Data Processing:** Parsing, cleaning, feature engineering, negative sample generation  
- **Evaluation:** Comprehensive metrics - Accuracy, Precision, Recall, F1, ROC AUC  
- **Visualizations:** Correlation heatmaps, feature importance plots  

---

## 🗂 Repository Structure

```bash
DDI-Prediction-ML/
├── data/                       # Raw and processed datasets
│   ├── drugs.csv               # Drug information dataset
│   ├── drugbank.xsd            # XML Schema for DrugBank data
│   ├── drug_property_importances.csv
│   ├── combined_evaluation_metrics.csv
│   └── README.md               # Description of datasets and sources
├── scripts/                    # Python scripts for pipeline steps
│   ├── parse_drugbank.py       # Parsing raw DrugBank XML data
│   ├── preprocess.py           # Data cleaning and preparation
│   ├── feature_engineering.py  # Feature extraction and engineering
│   ├── model_training.py       # General model training pipeline
│   ├── logistic_reg.py         # Logistic Regression training
│   ├── svc_training.py         # SVC model training
│   ├── eval_and_compare.py     # Evaluation and comparison of models
│   ├── split_for_test.py       # Data splitting and test set creation
│   └── visual.py               # Visualization of results and feature importance
├── visuals/                    # Plots and images for analysis & presentation
│   ├── correlation_heatmap.png
│   ├── drug_property_importance.png
│   └── Radar_Chart_of_Evaluation_Metrics_radar.png
├── results/                    # Model outputs, logs, saved models
│   ├── best_model_info.csv
│   ├── LogisticRegression_model.pkl
│   ├── RandomForestClassifier_model.pkl
│   ├── SVC_model.pkl
│   ├── final.txt       
├── LICENSE                     # CC BY 4.0 License file
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Ignored files and folders
```

---

## 🧠 Problem Statement

Drug-Drug Interactions (DDIs) occur when multiple drugs interact, altering their effects and potentially causing adverse reactions. Timely prediction and identification of DDIs can save lives, reduce healthcare costs, and inform safer prescribing practices.

---

## 🧪 Data Processing Pipeline

1. **Parse** DrugBank XML data (`scripts/parse_drugbank.py`)  
2. **Preprocess** and clean the data (`scripts/preprocess.py`)  
3. **Feature Engineering** including balancing and sample generation (`scripts/feature_engineering.py`)  
4. **Model Training** for Logistic Regression, SVC, etc. (`scripts/model_training.py`, `scripts/logistic_reg.py`, `scripts/svc_training.py`)  
5. **Evaluation & Comparison** (`scripts/eval_and_compare.py`)  
6. **Visualization** of results (`scripts/visual.py`)  

---

## 🤖 Model Performance Summary

| Model                    | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression       | 0.666    | 0.666     | 1.00   | 0.799    | 0.500   |
| Support Vector Classifier | 0.653    | 0.653     | 1.00   | 0.790    | 0.494   |
| Random Forest            | 0.664    | 0.664     | 1.00   | 0.798    | 0.500   |

---

## 📊 Visual Insights

- **Correlation Heatmap:** Understanding feature relationships (`visuals/correlation_heatmap.png`)  
- **Feature Importance:** Top features influencing DDI prediction (`visuals/drug_property_importance.png`)  

---

## 📈 Future Directions

- Extend to the full DrugBank dataset for improved generalizability  
- Incorporate advanced pharmacokinetic/dynamic features  
- Experiment with deep learning models such as Graph Neural Networks  
- Integrate biological mechanisms underlying DDIs  

---

## 📚 Citation

Please cite our work if you use this repository:

> Aditya V J, et al. Predicting Drug-Drug Interactions Using Machine Learning. *ResearchSquare* 2025. DOI: [10.21203/rs.3.rs-6137371/v1](https://doi.org/10.21203/rs.3.rs-6137371/v1)

---

## 🔖 License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## 👨‍💻 About the Author

**Aditya V J**  
- B.Tech CSE @ Christ University, Bangalore  
- AI & Healthcare Researcher  
- [LinkedIn Profile](https://www.linkedin.com/in/raappo/)  
- [Published Research](https://doi.org/10.21203/rs.3.rs-6137371/v1)

---

## ⭐ Support & Collaboration

If this project helps you or you want to collaborate:  
- ⭐ Star the repo  
- 📩 Connect on LinkedIn  
- 🤝 Reach out for AI, healthcare, or defense technology collaborations  


