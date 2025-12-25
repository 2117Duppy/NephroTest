"""
NEPHROTEST - CKD Model Evaluation Script
=========================================
This script evaluates the trained Random Forest model for Chronic Kidney Disease (CKD) detection.
Uses the EXACT same preprocessing logic as ckd_predictor_app.py to ensure compatibility.

Author: NEPHROTEST Team
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = 'kidney_disease.csv'
MODEL_PATH = 'ckd_model.pkl'
TARGET_COLUMN = 'classification'

print("=" * 70)
print("NEPHROTEST - CKD Model Evaluation")
print("=" * 70)
print()

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================================================
print("[1/5] Loading trained model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    expected_features = list(model.feature_names_in_)
    
    print(f"✓ Model loaded successfully from '{MODEL_PATH}'")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Expected features: {len(expected_features)}")
    
except FileNotFoundError:
    print(f"✗ ERROR: Model file '{MODEL_PATH}' not found!")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: Failed to load model: {str(e)}")
    sys.exit(1)

# ============================================================================
# STEP 2: LOAD AND PREPROCESS DATASET
# ============================================================================
print("\n[2/5] Loading and preprocessing dataset...")
print("  Using SAME preprocessing as ckd_predictor_app.py")

try:
    # Load CSV
    df_raw = pd.read_csv(DATASET_PATH)
    print(f"\n✓ Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    # Extract target
    if TARGET_COLUMN not in df_raw.columns:
        print(f"✗ ERROR: Target column '{TARGET_COLUMN}' not found!")
        sys.exit(1)
    
    y_raw = df_raw[TARGET_COLUMN].str.strip()
    
    # Map target to binary: ckd=0 (positive), notckd=1 (negative) 
    # NOTE: This matches the model's training where class 0 = CKD
    y = y_raw.map({'ckd': 0, 'notckd': 1})
    
    # Remove rows with missing target
    valid_indices = y.notna()
    df_raw = df_raw[valid_indices].reset_index(drop=True)
    y = y[valid_indices].reset_index(drop=True)
    
    print(f"✓ Target cleaned: {len(y)} valid samples")
    print(f"  Class distribution: CKD (0)={sum(y==0)}, Not CKD (1)={sum(y==1)}")
    
    # ========================================================================
    # CRITICAL: Use EXACT same feature transformation as ckd_predictor_app.py
    # ========================================================================
    
    # Helper function to safely convert to float, handling missing values
    def safe_float(val, default):
        """Convert value to float, handling missing markers like '?', '\t?', etc."""
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
    
    # Prepare features list to match the working app
    features_list = []
    
    for idx in range(len(df_raw)):
        row = df_raw.iloc[idx]
        
        # Build feature mapping EXACTLY as in build_features_from_state()
        # This is copied directly from ckd_predictor_app.py lines 139-164
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
        
        # Create row in EXACT order expected by model
        feature_row = [mapping[feat] for feat in expected_features]
        features_list.append(feature_row)
    
    # Create DataFrame with exact feature names
    X = pd.DataFrame(features_list, columns=expected_features)
    
    print(f"✓ Features transformed: {X.shape[1]} features × {X.shape[0]} samples")
    print(f"  Preprocessing matches ckd_predictor_app.py logic")
    
except Exception as e:
    print(f"✗ ERROR: Failed to process dataset: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: MAKE PREDICTIONS
# ============================================================================
print(f"\n[3/5] Making predictions...")
try:
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 0]  # Probability of class 0 (CKD)
    
    print(f"✓ Predictions generated for {len(y_pred)} samples")
    print(f"  Predicted CKD (0): {sum(y_pred==0)}, Predicted Not CKD (1): {sum(y_pred==1)}")
    
except Exception as e:
    print(f"✗ ERROR: Failed to make predictions: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: COMPUTE EVALUATION METRICS
# ============================================================================
print(f"\n[4/5] Computing evaluation metrics...")
print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)

try:
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred, pos_label=0, zero_division=0) * 100  # CKD is positive class (0)
    recall = recall_score(y, y_pred, pos_label=0, zero_division=0) * 100
    f1 = f1_score(y, y_pred, pos_label=0, zero_division=0) * 100
    auc_roc = roc_auc_score(y, y_pred_proba)
    
    # Display metrics
    print(f"\n📊 Accuracy:   {accuracy:.2f}%")
    print(f"   → What it means: Percentage of correct predictions (both CKD and Not CKD)")
    print(f"   → Formula: (TP + TN) / Total predictions")
    
    print(f"\n📊 Precision:  {precision:.2f}%")
    print(f"   → What it means: When model predicts CKD, how often is it correct?")
    print(f"   → Formula: TP / (TP + FP)")
    print(f"   → Medical context: Minimizes false alarms (healthy people wrongly flagged)")
    
    print(f"\n📊 Recall (Sensitivity): {recall:.2f}%")
    print(f"   → What it means: Of all actual CKD patients, how many did we catch?")
    print(f"   → Formula: TP / (TP + FN)")
    print(f"   → Medical context: CRITICAL for screening - we want to catch all sick patients!")
    print(f"   → High recall = fewer missed diagnoses (fewer false negatives)")
    
    print(f"\n📊 F1-Score:   {f1:.2f}%")
    print(f"   → What it means: Balanced measure combining Precision and Recall")
    print(f"   → Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   → Medical context: Useful when you need balance between catching patients")
    print(f"      and not overwhelming the system with false alarms")
    
    print(f"\n📊 AUC-ROC:    {auc_roc:.4f}")
    print(f"   → What it means: Overall model performance across all thresholds")
    print(f"   → Range: 0.5 (random) to 1.0 (perfect)")
    print(f"   → Rule of thumb: >0.9 Excellent, >0.8 Good, >0.7 Acceptable")
    
except Exception as e:
    print(f"✗ ERROR: Failed to compute metrics: {str(e)}")
    sys.exit(1)

# ============================================================================
# STEP 5: CONFUSION MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

try:
    # Compute confusion matrix
    # For pos_label=0 (CKD):
    # [[TN, FP],  <- Actual: Not CKD (1)
    #  [FN, TP]]  <- Actual: CKD (0)
    cm = confusion_matrix(y, y_pred, labels=[1, 0])  # Order: Not CKD, CKD
    
    # Extract values
    tn, fp, fn, tp = cm.ravel()
    
    # Display raw confusion matrix
    print("\nConfusion Matrix (2×2 array):")
    print(cm)
    
    # Display formatted confusion matrix with labels
    print("\nDetailed Confusion Matrix:")
    print("┌" + "─" * 68 + "┐")
    print(f"│                          PREDICTED NOT CKD    PREDICTED CKD        │")
    print("├" + "─" * 68 + "┤")
    print(f"│ ACTUAL NOT CKD (1)       TN = {tn:4d}             FP = {fp:4d}          │")
    print(f"│ ACTUAL CKD (0)           FN = {fn:4d}             TP = {tp:4d}          │")
    print("└" + "─" * 68 + "┘")
    
    print("\nLegend:")
    print(f"  • True Negatives (TN):  {tn:4d} - Correctly identified healthy people")
    print(f"  • False Positives (FP): {fp:4d} - Healthy people incorrectly flagged as CKD")
    print(f"  • False Negatives (FN): {fn:4d} - CKD patients we MISSED (⚠️ Most critical!)")
    print(f"  • True Positives (TP):  {tp:4d} - Correctly identified CKD patients")
    
    # Clinical interpretation
    print("\n" + "=" * 70)
    print("CLINICAL INTERPRETATION FOR CKD SCREENING")
    print("=" * 70)
    print("\nIn a medical screening context:")
    print(f"  • High Recall ({recall:.2f}%) is MOST IMPORTANT")
    print(f"    → We caught {tp} out of {tp + fn} actual CKD patients")
    if fn > 0:
        print(f"    ⚠️  WARNING: {fn} CKD patients were missed (False Negatives)")
        print(f"       → These patients need disease management but weren't flagged")
    else:
        print(f"    ✓  EXCELLENT: No CKD patients were missed!")
    
    print(f"\n  • Precision ({precision:.2f}%) affects follow-up burden")
    if fp > 0:
        print(f"    → {fp} healthy people were incorrectly flagged (False Positives)")
        print(f"       → They will need confirmatory tests, causing anxiety and cost")
    else:
        print(f"    ✓  PERFECT: No false alarms!")
    
    print("\n" + "=" * 70)
    print("✓ Evaluation complete!")
    print("=" * 70)
    
except Exception as e:
    print(f"✗ ERROR: Failed to compute confusion matrix: {str(e)}")
    sys.exit(1)

# ============================================================================
# METRICS EXPLANATION FOR MEDICAL CONTEXT
# ============================================================================
"""
UNDERSTANDING METRICS IN CKD SCREENING CONTEXT:
===============================================

1. RECALL (Sensitivity) - MOST CRITICAL FOR SCREENING
   - Measures: "Of all sick patients, how many did we detect?"
   - Why it matters: Missing a CKD patient (False Negative) means they won't get
     treatment, leading to disease progression and potentially kidney failure.
   - Goal: Maximize recall to catch as many CKD patients as possible.
   - Trade-off: Higher recall may increase false alarms (lower precision).

2. PRECISION (Positive Predictive Value)
   - Measures: "Of all positive predictions, how many were actually sick?"
   - Why it matters: False positives cause unnecessary anxiety, additional tests,
     and healthcare costs for healthy people.
   - Goal: Balance with recall - we want to minimize false alarms but not at the
     cost of missing real patients.

3. F1-SCORE
   - Measures: Harmonic mean of Precision and Recall
   - Why it matters: Provides a single metric that balances both concerns.
   - Goal: Useful for comparing different models or thresholds.

4. ACCURACY
   - Measures: Overall correctness
   - Caution: Can be misleading if classes are imbalanced (e.g., if 90% of people
     are healthy, a model that always predicts "healthy" gets 90% accuracy but
     misses all CKD patients!)

5. AUC-ROC
   - Measures: Model's ability to distinguish between classes
   - Why it matters: Threshold-independent metric showing overall model quality.
   - Goal: Higher is better (0.5 = random, 1.0 = perfect).

RECOMMENDATION FOR CKD SCREENING:
---------------------------------
Prioritize RECALL over Precision. It's better to have some false alarms and do
confirmatory tests than to miss actual CKD patients who need treatment.
A good screening model should have:
  - Recall: >85% (catch most patients)
  - Precision: >70% (manageable false alarm rate)
  - AUC-ROC: >0.85 (good overall performance)
"""
