🩺 Diabetes Prediction using Machine Learning
Pima Indians Diabetes Dataset
📌 Project Overview

This project aims to predict whether a patient has diabetes based on diagnostic medical measurements using Machine Learning.

The dataset used is the Pima Indians Diabetes Dataset, which contains medical records of female patients of Pima Indian heritage.

The project demonstrates a complete ML workflow:

Data Cleaning

Exploratory Data Analysis (EDA)

Feature Scaling

Model Training

Model Evaluation

Handling Imbalanced Data (SMOTE)

Hyperparameter Tuning

📊 Dataset Information

Total Records: 768

Features: 8

Target Variable: Outcome

0 → No Diabetes

1 → Diabetes

Features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

🧹 Data Preprocessing

Replaced unrealistic zero values in:

Glucose

BloodPressure

SkinThickness

Insulin

BMI

Used median to handle skewness and outliers.

Applied Train-Test Split (80-20).

Applied Feature Scaling (StandardScaler / RobustScaler).

Handled class imbalance using SMOTE.

📈 Exploratory Data Analysis

Key Findings:

Glucose is the most important feature.

Diabetic patients have significantly higher median glucose.

Insulin shows strong outliers.

Dataset is slightly imbalanced (500 vs 268).

🤖 Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Cross Validation

GridSearchCV for hyperparameter tuning

📊 Model Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC-AUC Score

In medical datasets, Recall is prioritized to reduce false negatives.

🏆 Best Performing Model

Random Forest achieved the best overall balance between:

Accuracy

Recall

Stability

🛠 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Imbalanced-learn

🚀 How to Run This Project

Clone the repository:

git clone https://github.com/your-username/diabetes-prediction.git


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook DB.ipynb

📌 Future Improvements

Implement XGBoost

Deploy using Flask / FastAPI

Build a Web App interface

Use larger dataset for better generalization

👨‍💻 Author

Sanyam Sharma
Machine Learning Enthusiast
