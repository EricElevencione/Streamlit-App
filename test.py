import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

# Title and description
st.title("Customer Churn Analysis Dashboard")
st.markdown("""
This dashboard helps analyze customer churn patterns and predict potential churners.
The analysis is based on the Telco Customer Churn dataset.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

# Load the data
try:
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Churn Prediction"])
    
    if page == "Data Overview":
        st.header("Dataset Overview")
        
        # Display basic information
        st.subheader("Basic Information")
        st.write(f"Number of customers: {len(df)}")
        st.write(f"Number of features: {len(df.columns)}")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Display data types and missing values
        st.subheader("Data Types and Missing Values")
        data_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(data_info)
        
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        # Churn Distribution
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, title='Churn Distribution')
        st.plotly_chart(fig)
        
        # Numerical Features Analysis
        st.subheader("Numerical Features Analysis")
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Convert TotalCharges to numeric, handling any non-numeric values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        for col in numerical_cols:
            fig = px.histogram(df, x=col, color='Churn', 
                             title=f'{col} Distribution by Churn Status',
                             barmode='overlay')
            st.plotly_chart(fig)
        
        # Categorical Features Analysis
        st.subheader("Categorical Features Analysis")
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
        
        for col in categorical_cols:
            fig = px.bar(df.groupby([col, 'Churn']).size().reset_index(name='count'),
                        x=col, y='count', color='Churn',
                        title=f'{col} Distribution by Churn Status')
            st.plotly_chart(fig)
            
    elif page == "Churn Prediction":
        st.header("Churn Prediction")
        
        # Data preprocessing
        def preprocess_data(df):
            # Convert categorical variables to dummy variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            
            # Handle missing values
            df_processed = df_processed.fillna(df_processed.mean())
            
            return df_processed
        
        # Prepare data for modeling
        df_processed = preprocess_data(df)
        X = df_processed.drop('Churn_Yes', axis=1)
        y = df_processed['Churn_Yes']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        
        st.subheader("Model Performance")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), x='Feature', y='Importance',
                    title='Top 10 Most Important Features')
        st.plotly_chart(fig)
        
        # Prediction interface
        st.subheader("Predict Churn for New Customer")
        
        # Create input fields for key features
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
            contract = st.selectbox("Contract", df['Contract'].unique())
            
        with col2:
            internet_service = st.selectbox("Internet Service", df['InternetService'].unique())
            payment_method = st.selectbox("Payment Method", df['PaymentMethod'].unique())
            
        if st.button("Predict Churn"):
            # Create a sample input
            sample_input = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'Contract': [contract],
                'InternetService': [internet_service],
                'PaymentMethod': [payment_method]
            })
            
            # Preprocess the input
            sample_processed = preprocess_data(sample_input)
            
            # Make prediction
            prediction = model.predict_proba(sample_processed)[0]
            
            # Display results
            st.write("Churn Probability:", f"{prediction[1]:.2%}")
            
            if prediction[1] > 0.5:
                st.error("High risk of churn!")
            else:
                st.success("Low risk of churn!")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please make sure the dataset file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory as this script.")




