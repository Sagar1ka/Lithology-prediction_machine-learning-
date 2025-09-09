# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:29:32 2025

@author: roysa
"""

# Python code to run in the notebook environment.
# It will attempt to load 'well_logs.csv' from /mnt/data (or current working dir).
# If that file is not present, it will create a synthetic dataset that mimics the real logs
# (Gamma-ray, Density, Resistivity) and lithology classes, and then train models:
# SVM, Decision Tree, Random Forest, MLP, and XGBoost (if available).
# The script performs preprocessing, stratified train/test split, GridSearchCV for small param grids,
# evaluation (classification report, confusion matrix), cross-validation scores, feature importance for tree models,
# and saves best models to disk.
# This code is intended as a ready-to-run template for your real well log CSV.
# If you have a CSV, ensure it contains columns: 'GR' (gamma ray), 'Density', 'Resistivity', and 'Lithology'.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt

# Helper to display dataframe in this environment if available
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_df_fn = display_dataframe_to_user
except Exception:
    display_df_fn = None

# Try importing XGBoost; if unavailable, skip XGB
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception as e:
    print("XGBoost not available in this environment; XGB model will be skipped. Error:", e)
    xgb_available = False

DATA_PATH = "well_logs.csv"  # change if your file is elsewhere

def generate_synthetic_data(n_boreholes=25, samples_per_bh=200, random_state=42):
    rng = np.random.RandomState(random_state)
    n = n_boreholes * samples_per_bh
    # Synthetic log ranges (approximate)
    GR = rng.normal(loc=80, scale=30, size=n)  # gamma ray
    Density = rng.normal(loc=2.35, scale=0.15, size=n)  # g/cc
    Resistivity = np.exp(rng.normal(loc=1.5, scale=1.0, size=n))  # ohm-m lognormal
    # Create synthetic lithology classes: Sand, Clay, Gravel, Shale, Limestone, Water-bearing Sand
    classes = ['Sand', 'Clay', 'Gravel', 'Shale', 'Limestone', 'WaterSand']
    # Make class assignment depend on logs
    probs = np.zeros((n, len(classes)))
    # Example heuristics for synthetic labels
    probs[:, 0] = (GR < 70) & (Density < 2.5) & (Resistivity > 5)  # Sand
    probs[:, 1] = (GR > 90) & (Density < 2.45) & (Resistivity < 5)  # Clay
    probs[:, 2] = (Density > 2.45) & (Resistivity > 10)  # Gravel
    probs[:, 3] = (GR > 110) & (Resistivity < 2)  # Shale
    probs[:, 4] = (Density > 2.6) & (Resistivity < 8)  # Limestone
    probs[:, 5] = (Resistivity < 3) & (GR < 80)  # Water-bearing sand (low resistivity + low GR)
    # If none of the heuristics fired, random assign weighted by GR
    labels = []
    for i in range(n):
        idxs = np.where(probs[i] > 0)[0]
        if len(idxs) == 0:
            labels.append(rng.choice(classes, p=[0.25,0.2,0.15,0.15,0.15,0.1]))
        else:
            labels.append(classes[rng.choice(idxs)])
    bh_ids = np.repeat(np.arange(1, n_boreholes+1), samples_per_bh)
    df = pd.DataFrame({
        'Borehole_ID': bh_ids,
        'GR': GR,
        'Density': Density,
        'Resistivity': Resistivity,
        'Lithology': labels
    })
    # Clip physically impossible values
    df['GR'] = df['GR'].clip(0, 300)
    df['Density'] = df['Density'].clip(1.8, 3.2)
    return df

# Load or generate dataset
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from {DATA_PATH}, rows:", len(df))
else:
    print(f"File '{DATA_PATH}' not found. Generating synthetic demo dataset.")
    df = generate_synthetic_data(n_boreholes=25, samples_per_bh=200)

# Quick look at the data
if display_df_fn is not None:
    display_df_fn("Sample well logs (head)", df.head(10))
else:
    print(df.head())

# Ensure required columns exist
required_cols = {'GR', 'Density', 'Resistivity', 'Lithology'}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"Input data must contain columns: {required_cols}. Found: {df.columns.tolist()}")

# Preprocessing
X = df[['GR', 'Density', 'Resistivity']].copy()
y = df['Lithology'].astype(str).copy()
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_
print("Classes:", class_names)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Define models and parameter grids for GridSearchCV (small grids for demo)
models_and_grids = {
    'SVM': (SVC(probability=True, random_state=42), {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__kernel': ['rbf', 'linear']
    }),
    'DecisionTree': (DecisionTreeClassifier(random_state=42), {
        'clf__max_depth': [4, 8, None],
        'clf__min_samples_split': [2, 5, 10]
    }),
    'RandomForest': (RandomForestClassifier(random_state=42, n_jobs=-1), {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [6, 12, None]
    }),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {
        'clf__hidden_layer_sizes': [(50,), (100,)], 
        'clf__alpha': [0.0001, 0.001]
    })
}

if xgb_available:
    models_and_grids['XGBoost'] = (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [3, 6]
    })

results = {}
best_models = {}

# Use StratifiedKFold for CV inside GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (estimator, param_grid) in models_and_grids.items():
    print(f"\nTraining {name} ...")
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='f1_macro', verbose=0)
    gs.fit(X_train, y_train)
    print(f"Best params for {name}: {gs.best_params_}")
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{name} Test Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    # Cross-validated score on whole train for the chosen metric
    cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"{name} CV F1-macro scores: {cv_scores}\nMean: {cv_scores.mean():.4f} Std: {cv_scores.std():.4f}")
    results[name] = {
        'best_params': gs.best_params_,
        'test_accuracy': acc,
        'test_f1_macro': f1,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
        'confusion_matrix': cm
    }
    best_models[name] = best
    # Save model
    model_filename = f"best_model_{name}.joblib"
    joblib.dump(best, model_filename)
    print(f"Saved best {name} model to {model_filename}")

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout()
    plt.show()

    # Feature importance for tree-based models
    if name in ('DecisionTree', 'RandomForest', 'XGBoost') and hasattr(best.named_steps['clf'], 'feature_importances_'):
        importances = best.named_steps['clf'].feature_importances_
        fi_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
        print("Feature importances:\n", fi_df.to_string(index=False))
        # Plot feature importances
        plt.figure(figsize=(6,3))
        plt.bar(fi_df['feature'], fi_df['importance'])
        plt.title(f"{name} Feature Importances")
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

# Summarize results in a DataFrame
summary_rows = []
for name, res in results.items():
    summary_rows.append({
        'Model': name,
        'Test Accuracy': res['test_accuracy'],
        'Test F1-macro': res['test_f1_macro']
    })
summary_df = pd.DataFrame(summary_rows).sort_values('Test F1-macro', ascending=False)
if display_df_fn is not None:
    display_df_fn("Model performance summary", summary_df)
else:
    print("\nModel performance summary:\n", summary_df)

print("\nCompleted training all models. Best models saved as joblib files in working directory.")
