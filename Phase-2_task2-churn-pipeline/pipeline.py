import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load & Clean
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Target: Churn → 0/1
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
# ColumnTransformer applies different steps to different column types
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Pipelines
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# GridSearchCV Tuning
# Note: parameter names follow pattern: stepname__paramname
lr_params = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["lbfgs", "liblinear"]
}

rf_params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
}

print("Tuning Logistic Regression...")
lr_cv = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring="f1", n_jobs=-1)
lr_cv.fit(X_train, y_train)

print("Tuning Random Forest...")
rf_cv = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_cv.fit(X_train, y_train)

# Evaluate
for name, model in [("Logistic Regression", lr_cv), ("Random Forest", rf_cv)]:
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print(f"  Best Params: {model.best_params_}")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# Export
# Save entire pipeline (preprocessor + model) — load later with joblib.load()
joblib.dump(lr_cv.best_estimator_, "models/lr_churn_pipeline.pkl")
joblib.dump(rf_cv.best_estimator_, "models/rf_churn_pipeline.pkl")
print("\nPipelines saved to models/")