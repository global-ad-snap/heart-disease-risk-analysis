# heart_disease_prediction.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# 📁 Ensure visuals folder exists
os.makedirs("visuals", exist_ok=True)

# 🔹 Load and clean data
def load_and_clean_data(filepath):
    """
    Loads CSV data, strips column names, and replaces ambiguous values with NaN.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace(['?', '', 'unknown', 999, -1], np.nan, inplace=True)
    return df

# 🔹 Preprocess data
def preprocess_data(df):
    """
    Imputes missing values, scales numeric features, and encodes categorical variables.
    """
    numeric_columns = ['trestbps', 'chol', 'thalachh', 'fbs', 'exang', 'ca']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    df = pd.get_dummies(df, columns=['restecg', 'slope', 'thal'], drop_first=True)
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# 🔹 Split data
def split_data(df):
    """
    Splits data into training and testing sets.
    """
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train models
def train_models(X_train, y_train):
    """
    Trains multiple classifiers and returns a dictionary of fitted models.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True), # enable predict_proba for ROC–AUC
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier()
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

# 🔹 Evaluate models
def evaluate_models(models, X_test, y_test):
    """
    Evaluates models and returns a sorted DataFrame of metrics.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        # ROC–AUC requires probability estimates
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            auc = np.nan
        
        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"visuals/confusion_matrix_{name.replace(' ', '_')}.png", bbox_inches='tight')
        plt.close()

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': auc
        })
    return pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)

# 🔹 Plot visuals
def plot_target_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2')
    plt.title('Target Class Distribution')
    plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("visuals/target_distribution.png", bbox_inches='tight')
    plt.show()

def plot_feature_distributions(df, features=['age', 'chol', 'oldpeak']):
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.savefig(f"visuals/distribution_{feature}.png", bbox_inches='tight')
        plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("visuals/correlation_heatmap.png", bbox_inches='tight')
    plt.show()

def plot_model_comparison(results_df, save_path="visuals/model_comparison_metrics.png", save=True):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
    fig.suptitle('Model Comparison Metrics', fontsize=16, y=1.02)  # move title higher
    
    for ax, metric in zip(axes, metrics):
        sns.barplot(x=results_df[metric], y=results_df['Model'], palette='viridis', ax=ax)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel('')
        ax.set_xlim(0, 1)  # keep scale consistent
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 🔹 Interpret model
def interpret_model(model, X_test):
    """
    Generates feature importance and SHAP summary plots.
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=X_test.columns)
        plt.figure(figsize=(10,6))
        sns.barplot(x=importance.values, y=importance.index)
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("visuals/feature_importance_lightgbm.png", bbox_inches='tight') 
        plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values_to_use = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(shap_values_to_use, X_test)
    plt.savefig("visuals/shap_summary_lightgbm.png", bbox_inches='tight')
    plt.show()

    try:
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values_to_use[1], X_test.iloc[1])
    except Exception as e:
        print("⚠️ SHAP force plot failed:", e)

# 🔹 Main execution
if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "raw_merged_heart_dataset.csv")
    df = load_and_clean_data(DATA_PATH)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)
    print(results_df)

    # Generate visuals
    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_model_comparison(results_df)

    best_model = models['LightGBM']
    interpret_model(best_model, X_test)















