# Fraud Guard

Fraud Guard is a Python-based application designed to detect fraudulent transactions in credit card data. It utilizes machine learning algorithms to analyze transaction patterns and identify anomalies that may indicate fraud.

## Features

- **Data Loading**: Load the credit card transaction dataset from a CSV file.
- **Data Preprocessing**: Handle missing values and scale features for better model performance.
- **Model Training**: Implement various classification algorithms, including Logistic Regression, Random Forest, and Decision Tree.
- **Oversampling**: Utilize SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset and improve model accuracy.
- **Model Evaluation**: Generate performance metrics including accuracy, confusion matrix, and ROC-AUC score.
- **Save Model**: Save the trained model for future use with `joblib`.

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset includes features that are the result of a PCA transformation, which are anonymized for security purposes.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib tqdm
