# 🏦 Bank Loan Approval Prediction using Machine Learning

This project is an end-to-end Machine Learning system that predicts whether a bank loan will be **Approved** or **Rejected** based on an applicant’s financial and personal details. The goal is to automate and improve the accuracy of loan approval decisions using data-driven methods.

The entire pipeline includes:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training
* Model evaluation
* Saving the best trained model

The project is built completely using **Python and Scikit-learn**.

---

## 📁 Dataset Information

* **Source:** Kaggle
* **Total Records:** 4,269
* **Total Columns:** 13
* **Target Column:** `loan_status` (Approved / Rejected)

### Main Features Used:

* Income per annum
* Loan amount
* Loan term
* CIBIL credit score
* Residential, commercial, luxury & bank asset values
* Number of dependents
* Education
* Self-employed status

---

## ⚙️ How the Project Works

### 1. Data Loading & Cleaning

* Dataset is loaded from `train.csv`
* Column names are cleaned and standardized
* Target labels are normalized into:

  * `approved` → 1
  * `rejected` → 0
* Missing values are handled using:

  * Median for numeric features
  * Most frequent value for categorical features

---

### 2. Exploratory Data Analysis (EDA)

The following outputs are generated automatically and saved in the `output/` folder:

* Education vs Loan Status plot
* Income vs Loan Status distribution plot
* Correlation analysis with loan approval

A summary file `eda_summary.txt` is also created.

Key Insight:

* **CIBIL score has the strongest influence on loan approval decisions.**

---

### 3. Feature Engineering

New features created:

* **TotalAssets** = Sum of all asset values
* **Loan-to-Asset Ratio** = Loan amount / Total assets

These features improve model understanding of financial risk.

---

### 4. Train–Test Split

* Dataset is split into:

  * **80% Training Data**
  * **20% Testing Data**
* Stratified splitting is applied to preserve class balance.

---

## 🤖 Models Used

Two Machine Learning models are trained and compared:

### 1. Logistic Regression

Used as a baseline linear classification model.

### 2. Random Forest Classifier

Used as the final model due to its superior performance.

---

## 📊 Model Performance

| Model               | Accuracy   | F1 Score   | Precision  | Recall     |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 91.33%     | 93.11%     | 92.08%     | 94.16%     |
| ✅ Random Forest     | **98.83%** | **99.06%** | **98.87%** | **99.24%** |

✅ **Random Forest is selected as the final model.**

---

## 💾 Output Files Generated

After running the program, the following files are saved inside the `output/` folder:

* `education_vs_loan_status.png`
* `income_distribution_by_loan_status.png`
* `eda_summary.txt`
* `model_comparison.csv`
* `feature_importance_rf.png`
* `feature_importances.csv`
* `best_model.pkl` ✅ (Final trained model)

---

## ▶️ How to Run This Project

### Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 2: Place Dataset

Put `train.csv` in either:

* The root folder
  OR
* Inside a folder named `data/`

### Step 3: Run the Program

```bash
python loan_pipeline.py
```

All outputs will be generated automatically.

---

## ✅ Final Result

This system successfully predicts loan approval with **nearly 99% accuracy** using the Random Forest classifier. It is suitable for academic demonstrations, portfolio projects, and real-world banking automation concepts.

---

## 👤 Author

**Sagnik Majumdar**

---

