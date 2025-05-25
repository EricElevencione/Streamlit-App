import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# Encode target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify categorical and numerical features
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Save the column names after transformation
encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
all_columns = np.concatenate([num_features, encoded_columns])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC AUC:", roc_auc)

# Save metrics to dictionary
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

metrics = {
    "report": report,
    "roc_auc": roc_auc,
    "confusion_matrix": conf_matrix.tolist()
}

# Save files
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(preprocessor.named_transformers_['num'], f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(preprocessor.named_transformers_['cat'], f)

with open("columns.pkl", "wb") as f:
    pickle.dump(all_columns.tolist(), f)

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("✅ All files saved: xgboost_model.pkl, scaler.pkl, encoder.pkl, columns.pkl")
