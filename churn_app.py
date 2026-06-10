import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(r'D:\VS_CODE\Customer_churn_project\Telco-Customer-Churn.csv')

# =========================
# CLEAN DATA (MOST IMPORTANT FIX)
# =========================

# Remove ID column
df = df.drop('customerID', axis=1)

# Target encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Gender encoding
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# Convert ALL Yes/No columns automatically
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encoding for remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

# =========================
# SPLIT DATA
# =========================
X = df.drop('Churn', axis=1).values
y = df['Churn'].values

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# =========================
# ANN MODEL
# =========================
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train,
        batch_size=32,
        epochs=15,
        validation_data=(X_test, y_test))

# =========================
# STREAMLIT UI
# =========================
st.title("Customer Churn Prediction App")

st.sidebar.header("Input Features")

credit_score = st.sidebar.number_input("Credit Score", 0)
geography = st.sidebar.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.sidebar.selectbox("Gender", ("Female", "Male"))
age = st.sidebar.number_input("Age", 0)
tenure = st.sidebar.number_input("Tenure", 0)
balance = st.sidebar.number_input("Balance", 0.0)
num_of_products = st.sidebar.number_input("Number of Products", 1, 4)
has_cr_card = st.sidebar.selectbox("Has Credit Card", (0, 1))
is_active_member = st.sidebar.selectbox("Is Active Member", (0, 1))
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0)

# =========================
# USER INPUT PROCESSING
# =========================

user_data = pd.DataFrame([[
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary
]], columns=[
    'CreditScore','Geography','Gender','Age','Tenure',
    'Balance','NumOfProducts','HasCrCard',
    'IsActiveMember','EstimatedSalary'
])

# Convert gender
user_data['Gender'] = user_data['Gender'].map({'Female': 0, 'Male': 1})

# One-hot encoding (same as training)
user_data = pd.get_dummies(user_data, drop_first=True)

# Align columns with training data
user_data = user_data.reindex(columns=df.drop('Churn', axis=1).columns, fill_value=0)

# Scale input
user_data = sc.transform(user_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    prediction = ann.predict(user_data)
    result = "Churn" if prediction[0][0] > 0.5 else "No Churn"

    st.success(f"Prediction: {result}")

    # Accuracy
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)