import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['figure.dpi'] = 150
OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches='tight')
    print("Saved:", path)

def safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def normalize_target(s):
    if pd.isna(s):
        return np.nan
    s2 = str(s).strip().lower()
    if s2 in ('approved', 'approve', 'yes', 'y', '1', 'true', 'accepted'):
        return 'approved'
    if s2 in ('rejected', 'reject', 'no', 'n', '0', 'false', 'declined'):
        return 'rejected'
    return s2  

def main():
    try:
        DATA_PATH = os.path.join("data", "train.csv")
        if not os.path.exists(DATA_PATH):
            DATA_PATH = "train.csv"
        df = safe_read(DATA_PATH)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Head:\n", df.head().to_string(index=False, max_rows=5))

        if 'loan_status' not in df.columns:
            print("Error: target column 'loan_status' not found in dataset.")
            sys.exit(1)

        df['loan_status'] = df['loan_status'].apply(normalize_target)
        print("Unique loan_status values (post-normalize):", df['loan_status'].unique().tolist())
        missing_before = df['loan_status'].isna().sum()
        if missing_before > 0:
            print(f"Found {missing_before} rows with missing loan_status. These will be removed before training.")
            df = df[~df['loan_status'].isna()].copy()

        classes = df['loan_status'].unique()
        if len(classes) < 2:
            print("Error: After cleaning, target has fewer than 2 classes. Cannot train. Unique values:", classes)
            sys.exit(1)

        try:
            if 'education' in df.columns:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.countplot(data=df, x='education', hue='loan_status', ax=ax)
                ax.set_title("Education vs Loan Status")
                savefig(fig, "education_vs_loan_status.png")
                plt.close(fig)
            else:
                print("Column 'education' not present — skipping education plot.")
        except Exception as e:
            print("Warning: failed to create education plot:", e)

        income_col = None
        for cand in ['income_annum', 'applicantincome', 'applicant_income', 'income']:
            if cand in df.columns:
                income_col = cand
                break
        try:
            if income_col is not None:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.histplot(data=df, x=income_col, bins=40, kde=False, hue='loan_status', multiple="stack", ax=ax)
                ax.set_title(f"Income distribution by Loan Status ({income_col})")
                savefig(fig, "income_distribution_by_loan_status.png")
                plt.close(fig)
            else:
                print("No income-like column found (tried common names) — skipping income plot.")
        except Exception as e:
            print("Warning: failed to create income plot:", e)

        summary_lines = []
        summary_lines.append("Quick EDA Findings:")
        missing = df.isnull().sum()
        summary_lines.append("Missing values per column:\n" + missing.to_string())

        tmp = df.copy()
        tmp['loan_status_num'] = tmp['loan_status'].map({'approved':1, 'rejected':0})
        num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_status_num' in num_cols:
            num_cols.remove('loan_status_num')
            corrs = tmp[num_cols + ['loan_status_num']].corr()['loan_status_num'].drop('loan_status_num').sort_values(key=abs, ascending=False)
            summary_lines.append("\nNumeric features correlation with Loan_Status (abs sorted):\n" + corrs.to_string())
        with open(os.path.join(OUTDIR, "eda_summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))
        print("Saved: EDA summary to", os.path.join(OUTDIR, "eda_summary.txt"))

        numeric_candidates = [
            'income_annum','applicantincome','coapplicantincome','loan_amount','loan_term','cibil_score',
            'residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'
        ]
        num_features = [c for c in numeric_candidates if c in df.columns]

        cat_candidates = ['no_of_dependents','dependents','education','self_employed','gender','married','property_area']
        cat_features = [c for c in cat_candidates if c in df.columns]

        if len(num_features) == 0 and len(cat_features) == 0:
            print("Error: Could not find any expected numeric or categorical feature columns in dataset.")
            sys.exit(1)

        features = num_features + cat_features
        print("Using features:", features)

        df_model = df[features + ['loan_status']].copy()

        if num_features:
            num_imputer = SimpleImputer(strategy='median')
            df_model[num_features] = num_imputer.fit_transform(df_model[num_features])
        if cat_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_model[cat_features] = cat_imputer.fit_transform(df_model[cat_features])

        le_dict = {}
        for col in cat_features:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le

        df_model['loan_status_num'] = df_model['loan_status'].map({'approved':1, 'rejected':0})
        if df_model['loan_status_num'].isna().any():
            bad = df_model['loan_status_num'].isna().sum()
            print(f"Warning: {bad} target rows could not be mapped to 'approved'/'rejected' and will be removed.")
            df_model = df_model[~df_model['loan_status_num'].isna()].copy()
        y = df_model['loan_status_num'].astype(int)
        X = df_model.drop(columns=['loan_status','loan_status_num'])

        asset_cols = [c for c in ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'] if c in X.columns]
        if asset_cols:
            X['TotalAssets'] = X[asset_cols].sum(axis=1)
        if 'loan_amount' in X.columns and 'TotalAssets' in X.columns:
            X['loan_to_asset_ratio'] = X['loan_amount'] / (X['TotalAssets'] + 1)

        print("Final X shape:", X.shape, "y shape:", y.shape)
        if y.isna().any():
            print("Error: target still contains NaN after cleaning. Exiting.")
            sys.exit(1)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            print("Warning: stratify failed (possibly low class counts). Using random split without stratify.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Train:", X_train.shape, "Test:", X_test.shape)

        numeric_for_scaling = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        scaler = StandardScaler()
        X_train_lr = X_train.copy()
        X_test_lr = X_test.copy()
        if numeric_for_scaling:
            X_train_lr[numeric_for_scaling] = scaler.fit_transform(X_train_lr[numeric_for_scaling])
            X_test_lr[numeric_for_scaling] = scaler.transform(X_test_lr[numeric_for_scaling])

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_lr, y_train)
        y_pred_lr = lr.predict(X_test_lr)

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        def eval_print(name, y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            print(f"Model: {name}")
            print("Accuracy:", round(acc,4), "F1:", round(f1,4), "Precision:", round(prec,4), "Recall:", round(rec,4))
            print(classification_report(y_true, y_pred, digits=4))
            cm = confusion_matrix(y_true, y_pred)
            return dict(name=name, accuracy=acc, f1=f1, precision=prec, recall=rec, cm=cm)

        res_lr = eval_print("Logistic Regression", y_test, y_pred_lr)
        res_rf = eval_print("Random Forest", y_test, y_pred_rf)

        results_df = pd.DataFrame([res_lr, res_rf])
        results_df = results_df[['name','accuracy','f1','precision','recall']]
        results_df.to_csv(os.path.join(OUTDIR, "model_comparison.csv"), index=False)
        print("Saved model comparison CSV.")

        chosen_metric = "f1"
        best_model_name = results_df.sort_values(by=chosen_metric, ascending=False).iloc[0]['name']
        print("Best model by", chosen_metric, ":", best_model_name)

        if best_model_name == "Random Forest":
            joblib.dump(rf, os.path.join(OUTDIR, "best_model.pkl"))
        else:
            joblib.dump(lr, os.path.join(OUTDIR, "best_model.pkl"))
        print("Saved best model to", os.path.join(OUTDIR, "best_model.pkl"))

        if hasattr(rf, "feature_importances_"):
            fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,4))
            fi.plot(kind='bar', ax=ax)
            ax.set_title("Feature importances (Random Forest)")
            savefig(fig, "feature_importance_rf.png")
            plt.close(fig)
            fi.to_csv(os.path.join(OUTDIR,"feature_importances.csv"))

        print("All done. Check the 'output' folder.")

    except Exception as e:
        print("Fatal error:", str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
