import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix

# Load the dataset
data = pd.read_csv('ObesityDataSet.csv')

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Train and evaluate models
results = {}
confusion_matrices = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else None
    
    results[model_name] = {
        "Accuracy": accuracy,
        "Recall": recall,
        "F1 Score": f1,
        "Precision": precision,
        "ROC-AUC": roc_auc
    }
    
    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

# Streamlit app
st.title("Obesity Level Prediction")
st.markdown("### Enter the details below to predict obesity level")

# Display model performance metrics
st.markdown("### Model Performance Metrics")
for model_name, metrics in results.items():
    st.write(f"#### {model_name}")
    st.write(f"Accuracy: {metrics['Accuracy']:.4f}")
    st.write(f"Recall: {metrics['Recall']:.4f}")
    st.write(f"F1 Score: {metrics['F1 Score']:.4f}")
    st.write(f"Precision: {metrics['Precision']:.4f}")
    st.write(f"ROC-AUC: {metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] is not None else "ROC-AUC: Not available")
    st.write("---")

    # Display confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrices[model_name], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)
