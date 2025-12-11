# ML-Financial_Fraud_Detection

**Credit Card Fraud Detection**

Machine Learning Project | Imbalanced Classification

This project builds a complete machine learning pipeline to detect fraudulent credit card transactions using the creditcard.csv dataset. The notebook includes data loading, preprocessing, handling imbalanced data, model training, and performance evaluation.


**Project Highlights**

✔ Load and explore the credit card fraud dataset
✔ Handle missing values
✔ Deal with class imbalance (fraud vs non-fraud)
✔ Feature scaling for numerical features
✔ Train ML models (Logistic Regression, Random Forest, XGBoost, etc.)
✔ Evaluate performance using advanced metrics
✔ Visualize insights and fraud distribution

**Machine Learning Pipeline**
1. Data Loading

Uses:

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


Loads creditcard.csv and checks basic structure using:

df.head()

df.info()

df.isnull().sum()

2. Data Cleaning

Drop or fill missing values

Ensure correct types

Remove invalid rows if needed

3. Exploratory Data Analysis (EDA)

Fraud class distribution (df['Class'].value_counts())

Statistical summaries

Check correlations

Visualize imbalance

**Handling Class Imbalance**

Fraud cases are ~0.17% of the dataset. Techniques used may include:

Undersampling / Oversampling

SMOTE

Class weights

Anomaly detection style models

4. Preprocessing

Scale numerical features using StandardScaler

Train-test split

Prepare data for ML models

5. Model Training

Models may include:

Logistic Regression

Random Forest

XGBoost

Decision Tree

Gradient Boosting
(Your notebook imports XGBoost explicitly)

6. Evaluation Metrics

Fraud detection requires metrics beyond accuracy:

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

**Requirements**

Add this to requirements.txt:

numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn

**Results & Insights**

Fraud detection is challenging due to extreme imbalance

Models help distinguish rare fraudulent transactions

XGBoost and Random Forest often provide superior recall

Important for banking, finance, and digital payments security
