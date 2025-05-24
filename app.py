import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
                   names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

st.title("Pima Diabetes Classifier")
st.markdown("""
This app demonstrates core **classification algorithms** on the Pima Indians Diabetes dataset.

Compare classifiers and visualize two key models: **Logistic Regression** and **Decision Tree**.
""")

# Sidebar user input
st.sidebar.header("Input Parameters for Prediction")
def user_input():
    inputs = {}
    for col in data.columns[:-1]:
        inputs[col] = st.sidebar.number_input(col, value=float(data[col].median()))
    return pd.DataFrame([inputs])

user_df = user_input()

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 1. OneR Classifier ---
def one_r(X_train, y_train, X_test):
    best_acc = 0
    best_feature = None
    best_rules = {}
    for col in X_train.columns:
        rules = X_train[col].round().astype(int).groupby(X_train[col].round().astype(int)).agg(lambda x: y_train.loc[x.index].mode()[0])
        preds = X_test[col].round().astype(int).map(rules).fillna(0).astype(int)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_feature = col
            best_rules = rules
    final_preds = X_test[best_feature].round().astype(int).map(best_rules).fillna(0).astype(int)
    return final_preds, best_feature, best_rules

# --- 2. Other models ---
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=0)
}

results = {}

# OneR Model
oner_preds, best_feat, rules = one_r(X_train, y_train, X_test)
results["OneR"] = (accuracy_score(y_test, oner_preds), confusion_matrix(y_test, oner_preds), roc_auc_score(y_test, oner_preds))

# Train other models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = (accuracy_score(y_test, preds), confusion_matrix(y_test, preds), roc_auc_score(y_test, preds))

# Display metrics
st.subheader("📊 Model Performance Comparison")
# Convert Confusion Matrices to strings for display
formatted_results = {
    model: [acc, str(cm), roc]
    for model, (acc, cm, roc) in results.items()
}
df_results = pd.DataFrame.from_dict(formatted_results, orient="index", columns=["Accuracy", "Confusion Matrix", "ROC AUC"])
st.dataframe(df_results)

# --- ROC Curves ---
st.subheader("🔍 ROC Curves")
fig, ax = plt.subplots()
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=name)
probs = oner_preds
fpr, tpr, _ = roc_curve(y_test, probs)
ax.plot(fpr, tpr, label="OneR")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# --- Logistic Regression Explanation ---
st.subheader("📌 Logistic Regression Explanation with LIME")
# Create LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['No Diabetes', 'Diabetes'],
    mode='classification'
)

# Get explanation for the user input
exp = explainer.explain_instance(
    user_df.values[0],
    models["Logistic Regression"].predict_proba,
    num_features=8
)

# Create a figure for the explanation
fig2, ax = plt.subplots(figsize=(10, 6))
exp.as_pyplot_figure()
st.pyplot(fig2)

# Show prediction
st.write("**Prediction:**", int(models["Logistic Regression"].predict(user_df)[0]))

# --- Decision Tree Visualization ---
st.subheader("🌳 Decision Tree Visualization")
fig3, ax2 = plt.subplots(figsize=(12, 6))
plot_tree(models["Decision Tree"], feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], filled=True, ax=ax2)
st.pyplot(fig3)

# --- Predict using Logistic and Tree ---
st.subheader("🤖 Prediction using Logistic & Tree")
st.write("**Input Data:**")
st.dataframe(user_df)
log_pred = models["Logistic Regression"].predict(user_df)[0]
tree_pred = models["Decision Tree"].predict(user_df)[0]
st.success(f"Logistic Regression Prediction: {'Diabetic' if log_pred else 'Non-Diabetic'}")
st.success(f"Decision Tree Prediction: {'Diabetic' if tree_pred else 'Non-Diabetic'}")

st.caption("Made for Data Mining Class - Classification Module")
