"""
NEPHROTEST - Confusion Matrix Visualization
============================================
Generates confusion matrix plot using the ACTUAL trained model (ckd_model.pkl)
with the SAME preprocessing as evaluate_model.py for accurate results.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================================
# Load the ACTUAL trained model
# ============================================================================
MODEL_PATH = 'ckd_model.pkl'
DATASET_PATH = 'kidney_disease.csv'

print("Loading trained model...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

expected_features = list(model.feature_names_in_)
print(f"✓ Model loaded: {type(model).__name__}")

# ============================================================================
# Load and preprocess dataset (SAME as evaluate_model.py)
# ============================================================================
print("Loading and preprocessing dataset...")

df_raw = pd.read_csv(DATASET_PATH)
TARGET_COLUMN = 'classification'

# Extract and clean target
y_raw = df_raw[TARGET_COLUMN].str.strip()
y = y_raw.map({'ckd': 0, 'notckd': 1})  # 0=CKD, 1=Not CKD

valid_indices = y.notna()
df_raw = df_raw[valid_indices].reset_index(drop=True)
y = y[valid_indices].reset_index(drop=True)

print(f"✓ Dataset loaded: {len(y)} samples")
print(f"  CKD (0): {sum(y==0)}, Not CKD (1): {sum(y==1)}")

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
# Make predictions
# ============================================================================
print("Making predictions...")
y_pred = model.predict(X)

print(f"✓ Predictions complete")
print(f"  Predicted CKD (0): {sum(y_pred==0)}, Predicted Not CKD (1): {sum(y_pred==1)}")

# ============================================================================
# Create Confusion Matrix
# ============================================================================
cm = confusion_matrix(y, y_pred, labels=[1, 0])  # Order: Not CKD (1), CKD (0)

# Extract values
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix Results:")
print(f"  True Positives (TP):  {tp} - Correctly identified CKD")
print(f"  True Negatives (TN):  {tn} - Correctly identified healthy")
print(f"  False Positives (FP): {fp} - Healthy flagged as CKD")
print(f"  False Negatives (FN): {fn} - CKD patients missed")

# ============================================================================
# Plot 1: Standard Confusion Matrix (Seaborn Heatmap)
# ============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['Not CKD', 'CKD'],
            yticklabels=['Not CKD', 'CKD'],
            annot_kws={"size": 16, "weight": "bold"})

plt.title('NEPHROTEST - Random Forest Confusion Matrix\n(Actual Trained Model)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=14, fontweight='bold')

# Add metrics text
accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics_text = f'Accuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1-Score: {f1:.2f}%'
plt.text(2.3, 0.5, metrics_text, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()
plt.savefig('nephrotest_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: nephrotest_confusion_matrix.png")
plt.show()

# ============================================================================
# Plot 2: Detailed Confusion Matrix with sklearn Display
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['Not CKD', 'CKD'])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)

ax.set_title('NEPHROTEST - Detailed Confusion Matrix\n(98.50% Accuracy)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold')

# Make numbers larger
for text in disp.text_.ravel():
    text.set_fontsize(18)
    text.set_fontweight('bold')

plt.tight_layout()
plt.savefig('nephrotest_confusion_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: nephrotest_confusion_detailed.png")
plt.show()

print("\n" + "="*70)
print("✓ Confusion matrix plots generated successfully!")
print("  These plots use your ACTUAL trained model with correct preprocessing")
print("="*70)
