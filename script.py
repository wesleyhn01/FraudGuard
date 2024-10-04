import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import time

# Comment out matplotlib and seaborn imports if visualization is causing issues
# import matplotlib.pyplot as plt
# import seaborn as sns

start_time = time.time()

print('Fraud Guard has started running...')

# Load the dataset
try:
    print("Loading dataset...")
    df = pd.read_csv('creditcard.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit()

# Remove duplicate rows
print("Removing duplicate rows...")
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Rows removed: {initial_rows - df.shape[0]}")

# Standardize 'Amount' and 'Time' columns
print("Standardizing 'Amount' and 'Time' columns...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

def train_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classifiers on the given data.
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(random_state=42)
    }

    for name, model in classifiers.items():
        print(f"\n================ {name} ================\n")
        with tqdm(total=100, desc=f"Training {name}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            model.fit(X_train, y_train)
            pbar.update(50)
            y_pred = model.predict(X_test)
            pbar.update(50)
        
        # Print evaluation metrics
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
        
        # Calculate and print ROC-AUC score
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"ROC-AUC Score: {roc_auc}\n")

# Train and evaluate models on imbalanced data
print("Training on imbalanced data:")
train_model(X_train, X_test, y_train, y_test)

# Undersample normal transactions
print("Undersampling normal transactions...")
normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]
normal_sample = normal.sample(n=fraud.shape[0])
new_df = pd.concat([normal_sample, fraud], ignore_index=True)

X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Train and evaluate models on undersampled data
print("Training on undersampled data:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_model(X_train, X_test, y_train, y_test)

# Apply SMOTE for oversampling
print("Applying SMOTE oversampling:")
with tqdm(total=100, desc="SMOTE", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    pbar.update(100)

# Train and evaluate models on oversampled data
print("Training on oversampled data (SMOTE):")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
train_model(X_train, X_test, y_train, y_test)

# Train final model (Random Forest) on oversampled data
print("Training final Random Forest model:")
with tqdm(total=100, desc="Final Model Training", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    pbar.update(100)

# Save the trained model
try:
    print("Saving the model...")
    joblib.dump(model, "credit_card_model.pkl")
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")
    exit()

# Load the model and make a prediction
try:
    print("Loading the model...")
    model = joblib.load("credit_card_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please ensure the model was saved correctly.")
    exit()
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Specific transaction data for prediction
transaction = [[-1.2063166480452974, -0.653464067093327, 1.15579454161356, 1.4398458100309, 
                -0.0483979577939286, -0.257954764175468, -0.763320426103366, 0.339229688923037, 
                -0.768705965846787, -0.115541693321453, -0.20021646873148, -0.650926246487322, 
                -0.735778340137806, -1.3940656101548, 0.447721760826057, 0.98477163074674, 
                0.271223077633162, -0.251055900420813, -0.165413052650476, 0.0091942158033362, 
                -0.160498072645411, 0.518041029678345, -0.970619090556498, 0.104889604203672, 
                0.307935462300307, -0.222502578722938, 0.0825004649897294, 0.291624326333603, 
                0.125488524044667, -0.3442213776454372]]

print("Making prediction on sample transaction...")
pred = model.predict(transaction)
pred_proba = model.predict_proba(transaction)

print("\nPrediction for the specific transaction:")
if pred[0] == 0:
    print("Normal Transaction")
    confidence = pred_proba[0][0] * 100
else:
    print("Fraud Transaction")
    confidence = pred_proba[0][1] * 100

print(f"Confidence: {confidence:.2f}%")

# Display feature importance
print("\nFeature Importance:")
feature_importance = model.feature_importances_
features = X.columns
feature_importance_dict = dict(zip(features, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

print('FraudGuard has finished running!')
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")