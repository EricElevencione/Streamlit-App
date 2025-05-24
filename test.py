import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# Data Preparation Functions
def prepare_data(df, is_prediction=False):
    """
    Comprehensive data preparation function that handles all preprocessing steps
    """
    # Create a copy to avoid modifying the original data
    df_processed = df.copy()
    
    if not is_prediction:
        # 1. Handle Missing Values
        st.subheader("1. Missing Values Analysis")
        missing_values = df_processed.isnull().sum()
        st.write("Missing values per column:")
        st.write(missing_values[missing_values > 0])
    
    # Convert TotalCharges to numeric and handle missing values
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].mean(), inplace=True)
    
    # 2. Feature Selection
    if not is_prediction:
        st.subheader("2. Feature Selection")
    # Drop customerID as it's not useful for prediction
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
    
    # 3. Feature Engineering
    if not is_prediction:
        st.subheader("3. Feature Engineering")
    
    # Ensure numerical columns are float type
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(float)
    
    # Create tenure groups
    df_processed['tenure_group'] = pd.qcut(df_processed['tenure'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create monthly charges groups
    df_processed['monthly_charges_group'] = pd.qcut(df_processed['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Calculate average monthly charges per service
    # Handle division by zero by using monthly charges when tenure is 0
    df_processed['avg_monthly_charges'] = np.where(
        df_processed['tenure'] > 0,
        df_processed['TotalCharges'] / df_processed['tenure'],
        df_processed['MonthlyCharges']
    )
    df_processed['avg_monthly_charges'] = df_processed['avg_monthly_charges'].astype(float)
    df_processed['avg_monthly_charges'].fillna(df_processed['MonthlyCharges'], inplace=True)
    
    # 4. Categorical Variable Encoding
    if not is_prediction:
        st.subheader("4. Categorical Variable Encoding")
    
    # Get categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Create dummy variables for categorical features
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # 5. Feature Scaling
    if not is_prediction:
        st.subheader("5. Feature Scaling")
    
    # Identify numerical columns for scaling
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges']
    numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
    
    # Scale numerical features
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    return df_processed

# Load the data
try:
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Data Preparation", "Exploratory Analysis", "Churn Prediction"])
    
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
        
    elif page == "Data Preparation":
        st.header("Data Preparation Process")
        
        # Apply data preparation
        df_processed = prepare_data(df)
        
        # Display processed data information
        st.subheader("Processed Data Overview")
        st.write("Shape of processed data:", df_processed.shape)
        
        # Display correlation matrix
        st.subheader("Feature Correlations")
        correlation_matrix = df_processed.corr()
        fig = px.imshow(correlation_matrix,
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig)
        
        # Display feature importance
        st.subheader("Feature Importance Analysis")
        X = df_processed.drop('Churn_Yes', axis=1)
        y = df_processed['Churn_Yes']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10),
                    x='Feature',
                    y='Importance',
                    title='Top 10 Most Important Features')
        st.plotly_chart(fig)
        
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        # Churn Distribution
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values,
                    names=churn_counts.index,
                    title='Churn Distribution')
        st.plotly_chart(fig)
        
        # Numerical Features Analysis
        st.subheader("Numerical Features Analysis")
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        for col in numerical_cols:
            fig = px.histogram(df,
                             x=col,
                             color='Churn',
                             title=f'{col} Distribution by Churn Status',
                             barmode='overlay')
            st.plotly_chart(fig)
        
        # Categorical Features Analysis
        st.subheader("Categorical Features Analysis")
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
        
        for col in categorical_cols:
            fig = px.bar(df.groupby([col, 'Churn']).size().reset_index(name='count'),
                        x=col,
                        y='count',
                        color='Churn',
                        title=f'{col} Distribution by Churn Status')
            st.plotly_chart(fig)
            
    elif page == "Churn Prediction":
        st.header("Churn Prediction")
        
        try:
            # Prepare data for modeling
            df_processed = prepare_data(df)
            
            # Ensure all columns are numeric
            for col in df_processed.columns:
                if col != 'Churn_Yes':  # Skip the target variable
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
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
            
            fig = px.bar(feature_importance.head(10),
                        x='Feature',
                        y='Importance',
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
                try:
                    # Create a sample input with all necessary columns
                    sample_input = pd.DataFrame({
                        'tenure': [float(tenure)],
                        'MonthlyCharges': [float(monthly_charges)],
                        'TotalCharges': [float(monthly_charges * tenure)],  # Estimate total charges
                        'Contract': [contract],
                        'InternetService': [internet_service],
                        'PaymentMethod': [payment_method]
                    })
                    
                    # Preprocess the input
                    sample_processed = prepare_data(sample_input, is_prediction=True)
                    
                    # Ensure all columns from training data are present
                    for col in X.columns:
                        if col not in sample_processed.columns:
                            sample_processed[col] = 0
                    
                    # Reorder columns to match training data
                    sample_processed = sample_processed[X.columns]
                    
                    # Make prediction
                    prediction = model.predict_proba(sample_processed)[0]
                    
                    # Display results
                    st.write("Churn Probability:", f"{prediction[1]:.2%}")
                    
                    if prediction[1] > 0.5:
                        st.error("High risk of churn!")
                    else:
                        st.success("Low risk of churn!")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please check your input values and try again.")
                    
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            st.info("Please make sure the dataset file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory as this script.")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please make sure the dataset file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory as this script.")




