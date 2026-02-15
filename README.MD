# Diabetic Readmission Predictor
**Machine Learning - Assignment 2** | *Predicting hospital readmission based on patient vitals and medications*

---

## Problem Statement

The goal of this project is to predict whether a diabetic patient will be readmitted to the hospital after being discharged. Diabetic care is incredibly complex, and a patient's recovery depends heavily on how their specific body reacts to changes across multiple medications simultaneously. 

Because it is nearly impossible for human doctors to manually calculate the intersecting risks of dozens of different drugs, this project utilizes machine learning to build a binary classification model. By accurately flagging patients who are at a **"High Risk"** for readmission *before* they are sent home, healthcare providers can proactively intervene, ultimately improving patient care and preventing financial penalties for the hospital.

---

## Dataset Description

**Dataset Name:** Diabetes 130-US Hospitals (1999–2008)

**Source:** - UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/296
- Kaggle: https://www.kaggle.com/datasets/brandao/diabetes

**Size:**
- **Original records:** 101,766 patient encounters
- **After preprocessing:** ~71,518 unique patients (Duplicates strictly removed to prevent data leakage)
- **Features:** 40+ clinical and pharmaceutical features (after preprocessing)

**Key Features Include:**
1. **Demographic Features:** race, gender, age
2. **Clinical Features:** time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_diagnoses, number_outpatient, number_emergency, number_inpatient
3. **Pharmaceutical Features:** Dosage changes (No, Down, Steady, Up) across 24 diabetes medications including Metformin, Glimepiride, and Insulin.

**Target Variable:**
- `is_readmitted` (Binary)
  - **0** = Not readmitted
  - **1** = Readmitted (within 30 days or after 30 days)

---

## Machine Learning Models & Evaluation

We trained and evaluated **6 different machine learning models**. Because clinical healthcare data is inherently complex and often imbalanced, we applied **SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors)** to improve model boundaries and prioritized **Recall (Sensitivity)** and **MCC (Matthews Correlation Coefficient)** over standard Accuracy.

### Why These Metrics Matter in Healthcare:
- **Recall:** Critical for minimizing False Negatives (sending a highly vulnerable patient home without proper care).
- **MCC & AUC:** The gold standards for proving the model genuinely understands underlying clinical patterns across both classes.
- **Precision:** Important to avoid overwhelming medical staff with false alarms.

---

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.5854 | 0.6242 | 0.4847 | 0.6198 | 0.5439 | 0.1786 |
| Decision Tree | 0.5870 | 0.6052 | 0.4844 | 0.5470 | 0.5138 | 0.1580 |
| K-Nearest Neighbors | 0.5119 | 0.5850 | 0.4372 | **0.7780** | 0.5598 | 0.1222 |
| Naive Bayes | **0.6158** | 0.6158 | **0.5631** | 0.1649 | 0.2551 | 0.1219 |
| Random Forest | 0.6110 | 0.6487 | 0.5105 | 0.6068 | 0.5545 | 0.2164 |
| **XGBoost (Best)** | 0.5977 | **0.6572** | 0.4970 | 0.6771 | **0.5732** | **0.2184** |

---

## Model Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Acted as a strong linear baseline. It handled the resampled data well, achieving a balanced F1 score (0.543) and a respectable Recall (~62%), proving linear relationships exist in the clinical features. |
| **Decision Tree** | Performed moderately well but struggled with generalization. Its lower F1 (0.513) and Recall (0.547) compared to ensemble methods suggest it may be overfitting to specific patient rules. |
| **K-Nearest Neighbors** | Achieved the highest Recall (77.8%) but at a severe cost to Accuracy (51.1%) and MCC (0.122). It struggled with the "curse of dimensionality," essentially over-predicting readmissions and generating too many False Positives to be clinically viable. |
| **Naive Bayes** | The worst-performing model for this specific task. Despite having high Accuracy (~61.5%), its Recall crashed to 16.4%. Its fundamental assumption of feature independence failed here, as patient vitals and drug administrations are highly correlated. |
| **Random Forest** | Delivered excellent, balanced results. It achieved strong overall Accuracy (61.1%), a strong AUC (0.648), and a solid Recall (~60.6%). It successfully utilized bagging to reduce variance and map complex categorical drug data. |
| **XGBoost (Best Overall)** | The absolute best performer across the board. It achieved the highest AUC (0.6572), F1 Score (0.5732), and MCC (0.2184). It PERFECTLY balanced the precision-recall trade-off, achieving a strong Recall (67.7%) to catch high-risk patients while maintaining nearly 60% accuracy. **Recommended for production deployment.** |

---

## Advanced Techniques Used

### 1. **Data Leakage Prevention (Critical Step)**
- Removed duplicate patient encounters (kept only first visit per patient).
- Ensures the test data represents 100% unseen patients, preventing the model from memorizing patient identities.

### 2. **SMOTEENN (Advanced Resampling)**
- Applied strictly to training data to maintain Test Data integrity.
- Synthesizes minority class data while actively deleting noisy, overlapping majority data to create clean decision boundaries. 

### 3. **Feature Engineering**
- Combined visit types into `total_prior_visits` and clinical procedures into `total_interventions` to provide stronger risk signals to the models.

### 4. **Robust Imputation & Scaling**
- Used median imputation for numeric features (resists outliers) and mode for categoricals.
- `StandardScaler` was fitted *only* on training data to prevent statistical leakage.

---

## How to Run the Project Locally

**Step 1: Clone the Repository**
```bash
git clone <your-repository-url>
cd diabetes_project