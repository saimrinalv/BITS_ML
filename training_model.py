import pandas as pd
import numpy as np
import joblib
import re
import sys
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# Advanced Resampling for Class Imbalance
try:
    from imblearn.combine import SMOTEENN
    SMOTEENN_AVAILABLE = True
except ImportError:
    print("WARNING: imblearn not installed. Please run: pip install imbalanced-learn")
    SMOTEENN_AVAILABLE = False

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ===== GLOBAL CONFIGURATION =====
RANDOM_STATE = 42
TEST_SIZE = 0.2

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

# --- STEP 1: LOAD & DEDUPLICATE DATA (PREVENTING DATA LEAKAGE) ---
# WHY: The dataset contains multiple encounters for the same patients. If a patient is 
# in both the training and test set, the model memorizes the patient, leading to Data Leakage.
print_section("STEP 1: LOADING & CLEANING DATA")
try:
    df = pd.read_csv("diabetic_data.csv", na_values='?')
except FileNotFoundError:
    print("ERROR: 'diabetic_data.csv' not found!")
    sys.exit(1)

if 'patient_nbr' in df.columns:
    df = df.sort_values(['patient_nbr', 'encounter_id'])
    # Keep only the first encounter per patient to ensure test data represents unseen patients
    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    print(f"Removed duplicate patients (Leakage Prevention). Unique patients: {len(df):,}")

# --- STEP 2: TARGET CREATION ---
# WHY: We are simplifying the multi-class problem ('NO', '<30', '>30') into a binary 
# classification problem to predict IF a patient will be readmitted at all.
print_section("STEP 2: TARGET CREATION")
if 'readmitted' in df.columns:
    df['is_readmitted'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
    df = df.drop('readmitted', axis=1)
    print("Converted target 'readmitted' to binary format (0 = No, 1 = Yes)")

# --- STEP 3: ADVANCED FEATURE ENGINEERING ---
# WHY: Machine learning models perform better with aggregated clinical metrics rather 
# than isolated data points. We engineer clinically relevant features here.
print_section("STEP 3: FEATURE ENGINEERING")

# Feature 1: Total Prior Visits (Clinical Rationale: Identifies "Frequent Flyers" at high risk)
visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
if all(col in df.columns for col in visit_cols):
    df['total_prior_visits'] = df[visit_cols].sum(axis=1)
    print("Created feature: 'total_prior_visits'")

# Feature 2: Total Interventions (Clinical Rationale: Indicates the severity/complexity of the hospital stay)
intervention_cols = ['num_medications', 'num_lab_procedures', 'num_procedures']
if all(col in df.columns for col in intervention_cols):
    df['total_interventions'] = df[intervention_cols].sum(axis=1)
    print("Created feature: 'total_interventions'")

# Drop unusable identifiers that hold no predictive value
cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# --- STEP 4: MISSING VALUES & ENCODING ---
# WHY: Models cannot process NaN values or strings. We use Median for numerics to resist outliers,
# and Mode for categoricals to fill missing values safely.
print_section("STEP 4: IMPUTATION & ENCODING")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Robust Imputation
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

# Label Encoding for categorical strings
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
joblib.dump(label_encoders, 'label_encoders.pkl')

# Regex clean for XGBoost compatibility (Cannot handle [, ], or < in feature names)
df.columns = [re.sub(r'[\[\]<>]', '_', col) for col in df.columns]
print("Imputed missing values and label-encoded categorical features")

# --- STEP 5: STRATIFIED SPLITTING & SCALING ---
# WHY: 'stratify=y' ensures the exact same percentage of readmitted patients exists in both Train and Test.
# The Scaler is fitted ONLY on training data to prevent test data statistics from leaking into the model.
print_section("STEP 5: SPLITTING & SCALING")
X = df.drop('is_readmitted', axis=1)
y = df['is_readmitted']

# Assignment Requirement Validation
if X.shape[1] < 12 or X.shape[0] < 500:
    print(f"WARNING: Dataset does not meet minimum requirements (Features: {X.shape[1]}, Samples: {X.shape[0]})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # FIT ONLY ON TRAIN
X_test_scaled = scaler.transform(X_test)       # TRANSFORM ONLY ON TEST
joblib.dump(scaler, 'scaler.pkl')

# Export unscaled, unseen test data for the Streamlit App to use
test_data = X_test.copy()
test_data['is_readmitted'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print("Stratified split complete. Saved 'scaler.pkl' and 'test_data.csv'")

# --- STEP 6: SMOTEENN (CLASS IMBALANCE HANDLING) ---
# WHY: Readmission datasets are heavily imbalanced. SMOTEENN synthesizes minority class data (SMOTE) 
# and then deletes noisy, overlapping majority data (Edited Nearest Neighbors) to create clean decision boundaries.
print_section("STEP 6: APPLYING SMOTEENN")
if SMOTEENN_AVAILABLE:
    print("Applying SMOTEENN (This step deliberately removes noisy data to improve Precision)...")
    smote_enn = SMOTEENN(random_state=RANDOM_STATE)
    # Applied strictly to Training Data to maintain Test Data integrity
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training shape: {X_train_resampled.shape} (Noise successfully removed)")
else:
    X_train_resampled, y_train_resampled = X_train_scaled, y_train

# --- STEP 7: OPTIMIZED MODELS ---
# WHY: Hyperparameters have been explicitly tuned to prevent overfitting (e.g., max_depth limits, subsampling).
print_section("STEP 7: TRAINING OPTIMIZED MODELS")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_split=10, class_weight='balanced', random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,          # Increased trees for stability
        max_depth=15,              # Capped depth to prevent overfitting 
        min_samples_leaf=4,        # Smoother decision leaves
        class_weight='balanced', 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        n_estimators=200,          
        learning_rate=0.05,        # Slower, more accurate learning
        max_depth=6,               # Prevent deep, overfit trees
        subsample=0.8,             # Use 80% of data per tree to improve generalization
        colsample_bytree=0.8,      
        random_state=RANDOM_STATE
    )
}

print(f"\n{'Model':<20} | {'Accuracy':<8} | {'AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'MCC':<8}")
print("-" * 95)

results = []

for name, model in models.items():
    # 1. Train on Resampled Data
    model.fit(X_train_resampled, y_train_resampled)
    
    # 2. Predict on 100% Unseen, Real Test Data
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # 3. Evaluate Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"{name:<20} | {acc:.4f}   | {auc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {mcc:.4f}")
    
    # 4. Save Model for Streamlit Deployment
    joblib.dump(model, f"{name}.pkl")
    
    results.append({'Model': name, 'Accuracy': acc, 'AUC': auc, 'Precision': prec, 'Recall': rec, 'F1': f1})

# Save metadata for the Streamlit dashboard
metadata = {
    'training_date': datetime.now().isoformat(),
    'n_features': X.shape[1],
    'n_test_samples': X_test.shape[0],
    'smote_used': SMOTEENN_AVAILABLE
}
joblib.dump(metadata, 'model_metadata.pkl')

pd.DataFrame(results).to_csv('model_results.csv', index=False)
print("\nAll models trained and saved successfully! Run 'streamlit run app.py' to view dashboard.")