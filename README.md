# Customer Churn Prediction ML

## 📌 Project Overview

Customer Churn Prediction ML is a machine learning project designed to predict whether a customer is likely to leave a telecom service provider. By analyzing customer demographics, account information, and service usage patterns, the model helps businesses identify at-risk customers and improve retention strategies.

The project uses data preprocessing, feature engineering, feature scaling, and an Artificial Neural Network (ANN) built with TensorFlow/Keras to perform churn prediction. An interactive Streamlit application allows users to make real-time predictions.

---

## 🚀 Features

* Customer churn prediction using Artificial Neural Networks (ANN)
* Data preprocessing and feature engineering
* Automatic categorical variable encoding
* Feature scaling using StandardScaler
* Interactive Streamlit web interface
* Real-time customer churn prediction
* Model performance evaluation using Accuracy Score and Confusion Matrix

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* TensorFlow / Keras
* Scikit-Learn
* Streamlit

---

## 📂 Dataset

The project uses the Telco Customer Churn Dataset containing customer information such as:

* Gender
* Senior Citizen Status
* Partner
* Dependents
* Tenure
* Internet Service
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges
* Churn Status

---

## 🔄 Project Workflow

### 1. Data Collection

Load customer churn dataset from CSV format.

### 2. Data Preprocessing

* Remove unnecessary columns
* Handle categorical variables
* Encode target labels
* Apply one-hot encoding
* Scale features using StandardScaler

### 3. Model Development

Build and train an Artificial Neural Network using TensorFlow/Keras.

### 4. Model Evaluation

Evaluate model performance using:

* Accuracy Score
* Confusion Matrix

### 5. Deployment

Deploy the trained model through a Streamlit web application.

---


## 📊 Results

The model predicts whether a customer is likely to churn or remain with the company. This prediction can help organizations implement proactive customer retention strategies.

---



### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
customer-churn-prediction-ml/
│
├── app.py
├── Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
└── model/
```

---

## 🎯 Future Enhancements

* Hyperparameter tuning
* Model persistence using Pickle/Joblib
* Cloud deployment
* Advanced visualization dashboard
* Comparison with other ML algorithms such as Random Forest and XGBoost

---

## 📚 Learning Outcomes

* Machine Learning Workflow
* Deep Learning with TensorFlow/Keras
* Data Preprocessing Techniques
* Feature Engineering
* Streamlit Application Development
* Model Evaluation and Performance Analysis

---

## 👩‍💻 Author

**Swati Jadhav**

Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer

