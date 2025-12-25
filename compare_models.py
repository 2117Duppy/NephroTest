"""
NEPHROTEST - Model Comparison: Logistic Regression vs Decision Tree vs Random Forest
=====================================================================================
Trains and compares three models with confusion matrix visualizations.
Uses the SAME preprocessing as evaluate_model.py for consistency.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = 'kidney_disease.csv'
TARGET_COLUMN = 'classification'

print("=" * 70)
print("NEPHROTEST - Model Comparison")
print("=" * 70)
print()

# ============================================================================
# Load and preprocess dataset (SAME as evaluate_model.py)
# ============================================================================
print("[1/4] Loading and preprocessing dataset...")

df_raw = pd.read_csv(DATASET_PATH)

# Extract and clean target
y_raw = df_raw[TARGET_COLUMN].str.strip()
y = y_raw.map({'ckd': 0, 'notckd': 1})  # 0=CKD, 1=Not CKD

valid_indices = y.notna()
df_raw = df_raw[valid_indices].reset_index(drop=True)
y = y[valid_indices].reset_index(drop=True)

print(f"✓ Dataset loaded: {len(y)} samples")
print(f"  CKD (0): {sum(y==0)}, Not CKD (1): {sum(y==1)}")

# ============================================================================
# Load trained model to get expected feature order
# ============================================================================
print("\nLoading trained Random Forest to get feature order...")
with open('ckd_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
expected_features = list(rf_model.feature_names_in_)
print(f"✓ Loaded model with {len(expected_features)} expected features")

# ============================================================================
# Transform features (SAME preprocessing as evaluate_model.py)
# ============================================================================

def safe_float(val, default):
    """Convert value to float, handling missing markers."""
    if pd.isna(val):
        return default
    val_str = str(val).strip()
    if val_str in ['?', '', 'nan', 'NaN']:
        return default
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return default

def safe_str(val, default=''):
    """Convert value to string, handling missing values."""
    if pd.isna(val):
        return default
    return str(val).strip()

features_list = []

for idx in range(len(df_raw)):
    row = df_raw.iloc[idx]
    
    mapping = {
        "age": safe_float(row['age'], 45.0),
        "blood_pressure": safe_float(row['bp'], 80.0),
        "specific_gravity": safe_float(row['sg'], 1.015),
        "albumin": safe_float(row['al'], 0.0),
        "sugar": safe_float(row['su'], 0.0),
        "red_blood_cells": 1 if safe_str(row['rbc']) == "normal" else 0,
        "pus_cell": 1 if safe_str(row['pc']) == "normal" else 0,
        "pus_cell_clumps": 1 if safe_str(row['pcc']) == "present" else 0,
        "bacteria": 1 if safe_str(row['ba']) == "present" else 0,
        "blood_glucose_random": safe_float(row['bgr'], 120.0),
        "blood_urea": safe_float(row['bu'], 30.0),
        "serum_creatinine": safe_float(row['sc'], 1.0),
        "sodium": safe_float(row['sod'], 140.0),
        "potassium": safe_float(row['pot'], 4.0),
        "haemoglobin": safe_float(row['hemo'], 14.0),
        "packed_cell_volume": safe_float(row['pcv'], 40.0),
        "white_blood_cell_count": safe_float(row['wc'], 7000.0),
        "red_blood_cell_count": safe_float(row['rc'], 4.5),
        "hypertension": 1 if safe_str(row['htn']) == "yes" else 0,
        "diabetes_mellitus": 1 if safe_str(row['dm']) == "yes" else 0,
        "coronary_artery_disease": 1 if safe_str(row['cad']) == "yes" else 0,
        "appetite": 1 if safe_str(row['appet']) == "good" else 0,
        "peda_edema": 1 if safe_str(row['pe']) == "yes" else 0,
        "aanemia": 1 if safe_str(row['ane']) == "yes" else 0,
    }
    
    feature_row = [mapping[feat] for feat in expected_features]
    features_list.append(feature_row)

X = pd.DataFrame(features_list, columns=expected_features)
print(f"✓ Features transformed: {X.shape[1]} features × {X.shape[0]} samples")

# ============================================================================
# Train three models
# ============================================================================
print("\n[2/4] Training models...")

# Logistic Regression
print("  Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X, y)
y_pred_lr = log_reg.predict(X)

# Decision Tree
print("  Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X, y)
y_pred_dt = dt_model.predict(X)

# Random Forest (already loaded)
print("  Using trained Random Forest...")
y_pred_rf = rf_model.predict(X)

print("✓ All models ready")

# ============================================================================
# Calculate metrics for all models
# ============================================================================
print("\n[3/4] Computing metrics...")

models_data = {
    'Logistic Regression': y_pred_lr,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf
}

results = {}

for model_name, y_pred in models_data.items():
    cm = confusion_matrix(y, y_pred, labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y, y_pred) * 100
    prec = precision_score(y, y_pred, pos_label=0, zero_division=0) * 100
    rec = recall_score(y, y_pred, pos_label=0, zero_division=0) * 100
    f1 = f1_score(y, y_pred, pos_label=0, zero_division=0) * 100
    
    results[model_name] = {
        'cm': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%")

# ============================================================================
# Generate confusion matrix plots
# ============================================================================
print("\n[4/4] Generating confusion matrix plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, data) in enumerate(results.items()):
    cm = data['cm']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Not CKD', 'CKD'],
                yticklabels=['Not CKD', 'CKD'],
                annot_kws={"size": 14, "weight": "bold"},
                ax=axes[idx])
    
    axes[idx].set_title(f'{model_name}\nAccuracy: {data["accuracy"]:.2f}%',
                       fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')

plt.suptitle('NEPHROTEST - Model Comparison: Confusion Matrices', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_confusion_matrices.png")
plt.show()

# ============================================================================
# Generate individual plots for each model
# ============================================================================

for model_name, data in results.items():
    cm = data['cm']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=True,
                xticklabels=['Not CKD', 'CKD'],
                yticklabels=['Not CKD', 'CKD'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title(f'NEPHROTEST - {model_name}\nConfusion Matrix', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=14, fontweight='bold')
    
    # Add metrics text
    metrics_text = (f'Accuracy: {data["accuracy"]:.2f}%\n'
                   f'Precision: {data["precision"]:.2f}%\n'
                   f'Recall: {data["recall"]:.2f}%\n'
                   f'F1-Score: {data["f1"]:.2f}%')
    
    plt.text(2.3, 0.5, metrics_text, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')
    
    plt.tight_layout()
    
    # Save with appropriate filename
    filename = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()

# ============================================================================
# Print Summary Table
# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 70)
print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)

for model_name, data in results.items():
    print(f"{model_name:<25} {data['accuracy']:>10.2f}% {data['precision']:>10.2f}% "
          f"{data['recall']:>10.2f}% {data['f1']:>10.2f}%")

print("\n" + "=" * 70)
print("✓ Model comparison complete!")
print("=" * 70)
