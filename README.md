# 🩺 NephroTest - CKD Detection System

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning-powered **Chronic Kidney Disease (CKD) detection system** with an interactive web application for clinical screening and model evaluation.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

NephroTest is a comprehensive CKD detection system that uses machine learning to predict chronic kidney disease based on 24 clinical and laboratory parameters. The project includes:

- **Random Forest classifier** trained on the UCI CKD dataset (400 samples)
- **Interactive Streamlit web app** with polished UI for patient screening
- **Model evaluation tools** with detailed performance metrics
- **Comparison framework** for benchmarking multiple ML models

## ✨ Features

### 🌐 Web Application (`ckd_predictor_app.py`)
- **User-friendly interface** with single-column input form
- **24 clinical parameters** including age, blood pressure, lab values, and medical history
- **Real-time predictions** with probability scores
- **Interactive visualizations** including:
  - Zoomable PCA and t-SNE projections
  - Feature importance analysis
  - Patient-specific SHAP explanations
- **Patient history tracking** with CSV export functionality
- **Professional UI** with responsive design and glassmorphism effects

### 📊 Model Evaluation (`evaluate_model.py`)
- Comprehensive metrics: **Accuracy, Precision, Recall, F1-Score, AUC-ROC**
- Medical-context explanations for each metric
- Confusion matrix with clinical interpretation
- Evaluation on full dataset with consistent preprocessing

### 🔬 Model Comparison (`compare_models.py`)
- Benchmarks **3 ML algorithms**: Logistic Regression, Decision Tree, Random Forest
- Side-by-side confusion matrices
- Performance comparison table
- Individual model evaluation plots

### 📈 Visualization Tools
- `confusion_matrix_plot.py` - Train and visualize confusion matrices
- `plot_confusion_matrix.py` - Advanced confusion matrix plotting
- `inspect_model.py` - Model introspection utility

## 🛠 Technology Stack

- **Python 3.9+**
- **Machine Learning**: scikit-learn, XGBoost
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, SHAP
- **Model Persistence**: joblib, pickle

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ckd-detection-system.git
cd ckd-detection-system
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python inspect_model.py
```

Expected output:
```
Model expects the following feature names:
['age', 'blood_pressure', 'specific_gravity', ...]
```

## 🚀 Usage

### Running the Web Application

```bash
streamlit run ckd_predictor_app.py
```

The app will open at `http://localhost:8501`

**How to use:**
1. Enter patient clinical data in the input form
2. Click **"🔍 Analyze Patient"** to generate predictions
3. View results including:
   - CKD risk assessment (Low/High probability)
   - Confidence score
   - Interactive visualizations
   - Feature importance ranking
4. Export patient history via **"Download Patient History (CSV)"**

### Evaluating the Model

```bash
python evaluate_model.py
```

Output includes:
```
MODEL PERFORMANCE METRICS
======================================================================
📊 Accuracy:   98.50%
📊 Precision:  98.03%
📊 Recall:     99.60%
📊 F1-Score:   98.81%
📊 AUC-ROC:    0.9980

CONFUSION MATRIX
  True Positives (TP): 249  <- Correctly identified CKD patients
  False Negatives (FN): 1   <- Missed CKD patients (most critical!)
  ...
```

### Comparing Multiple Models

```bash
python compare_models.py
```

Generates:
- `model_comparison_confusion_matrices.png` - Side-by-side comparison
- Individual confusion matrix plots for each model
- Performance metrics table

### Visualizing Confusion Matrices

```bash
python confusion_matrix_plot.py
```

Trains 3 models and saves confusion matrix plots.

## 📁 Project Structure

```
ckd-detection-system/
│
├── ckd_model.pkl                          # Trained Random Forest model
├── kidney_disease.csv                     # UCI CKD dataset (400 samples)
│
├── ckd_predictor_app.py                   # Streamlit web application
├── evaluate_model.py                      # Model evaluation script
├── compare_models.py                      # Multi-model comparison
├── confusion_matrix_plot.py               # Confusion matrix visualization
├── plot_confusion_matrix.py               # Advanced plotting utilities
├── inspect_model.py                       # Model inspection tool
│
├── CKD_Model_Training_Updated.ipynb       # Jupyter notebook for training
├── patient_history_sample.csv             # Sample patient history export
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── LICENSE                                # MIT License
└── .gitignore                             # Git ignore rules
```

## 📈 Model Performance

### Random Forest Classifier (Primary Model)

| Metric | Score | Clinical Significance |
|--------|-------|----------------------|
| **Accuracy** | 99.75% | Overall correctness across all predictions |
| **Precision** | 100.00% | When we predict CKD, we're always right (no false alarms) |
| **Recall** | 99.60% | We catch 99.6% of actual CKD patients |
| **F1-Score** | 99.80% | Balanced measure of Precision and Recall |
| **AUC-ROC** | 0.9980 | Excellent discrimination ability |

**Confusion Matrix:**
```
                Predicted
              Not CKD    CKD
Actual  
Not CKD      150        0      ← Perfect specificity
CKD            1      249      ← Only 1 false negative
```

**Medical Context:**
- **High Recall (99.6%)** is critical for CKD screening - we only miss 1 out of 250 patients
- **Perfect Precision (100%)** means no false alarms - reduces unnecessary tests and patient anxiety
- Overall excellent performance suitable for clinical decision support

## 📊 Dataset

**Source:** [UCI Machine Learning Repository - Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Chronic+Kidney+Disease)

**Details:**
- **Samples:** 400 (250 CKD, 150 Not CKD)
- **Features:** 24 clinical attributes
  - **Numerical (11):** age, blood pressure, specific gravity, albumin, sugar, blood glucose, blood urea, creatinine, sodium, potassium, hemoglobin, packed cell volume, white blood cell count, red blood cell count
  - **Categorical (13):** red blood cells, pus cells, pus cell clumps, bacteria, hypertension, diabetes, coronary artery disease, appetite, pedal edema, anemia

**Preprocessing:**
- Missing values handled with domain-specific defaults
- Categorical encoding (0/1 binary)
- Feature names normalized for consistency

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [UCI CKD Dataset](https://archive.ics.uci.edu/ml/datasets/Chronic+Kidney+Disease)
- [Chronic Kidney Disease - MedlinePlus](https://medlineplus.gov/chronickidneydisease.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 👨‍💻 Author

**Prakhar Srivastav**

---

⭐ **Star this repository if you found it helpful!**

> **Disclaimer:** This is an educational project for demonstrating machine learning applications in healthcare. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
