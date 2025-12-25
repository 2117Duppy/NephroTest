import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------
df = pd.read_csv("kidney_disease.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Replace "?" with NaN
df = df.replace("?", np.nan)

# Convert numerical columns to numeric
num_cols = [
    'age','bp','sg','al','su','bgr','bu','sc','sod','pot',
    'hemo','pcv','wc','rc'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode categorical columns
cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
for col in cat_cols:
    df[col] = df[col].astype(str).str.lower()
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target separately
df['classification'] = df['classification'].astype(str).str.strip().str.lower()
df['classification'] = LabelEncoder().fit_transform(df['classification'])

# ----------------------------------------------------
# Handle Missing Values (Fill NaN)
# ----------------------------------------------------
# Fill numerical columns with median
for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode (most frequent value)
for col in cat_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)

# Drop 'id' column if it exists (not needed for training)
if 'id' in df.columns:
    df = df.drop('id', axis=1)

print(f"✓ Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Missing values remaining: {df.isnull().sum().sum()}")

# ----------------------------------------------------
# 2. Split Data
# ----------------------------------------------------
X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 3. Train Models
# ----------------------------------------------------
log_reg = LogisticRegression(max_iter=200)
dtree = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

log_reg.fit(X_train, y_train)
dtree.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ----------------------------------------------------
# 4. Make Predictions
# ----------------------------------------------------
log_pred = log_reg.predict(X_test)
dt_pred = dtree.predict(X_test)
rf_pred = rf_model.predict(X_test)

# ----------------------------------------------------
# Helper function for Confusion Matrix Plot
# ----------------------------------------------------
def plot_conf_matrix(cm, title, filename):
    plt.figure(figsize=(6,4))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", cbar=False)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# ----------------------------------------------------
# 5. Confusion Matrices
# ----------------------------------------------------
cm_log = confusion_matrix(y_test, log_pred)
cm_dt = confusion_matrix(y_test, dt_pred)
cm_rf = confusion_matrix(y_test, rf_pred)

plot_conf_matrix(cm_log, "Logistic Regression - Confusion Matrix", "log_reg_confusion.png")
plot_conf_matrix(cm_dt, "Decision Tree - Confusion Matrix", "decision_tree_confusion.png")
plot_conf_matrix(cm_rf, "Random Forest - Confusion Matrix", "rf_confusion.png")

print("Confusion Matrices Saved Successfully!")
