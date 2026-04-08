# Loan Credit Risk Prediction System

A comprehensive Machine Learning web application that predicts loan eligibility using banking marketing data. The system uses a Random Forest Classifier and provides predictions through a Flask web interface with a modern dashboard.

## Live Demo

https://loan-credit-rist-prediction-system-4.onrender.com

## Features

• Loan eligibility prediction using Machine Learning
• Random Forest Classifier with imbalanced data handling
• Real-time prediction using Flask API
• Feature importance visualization
• Model evaluation metrics (Accuracy, Precision, Recall, F1 Score)
• Modern responsive frontend interface
• Cloud deployment on Render

## Tech Stack

Python
Flask
Scikit-learn
Pandas
NumPy
HTML
CSS
JavaScript
Render (Cloud Hosting)
Joblib

## Machine Learning Pipeline

Data Preprocessing → Feature Scaling → Model Training → Model Evaluation → Model Serialization → Flask API Deployment

## How to Run Locally

### 1. Clone repository

git clone https://github.com/sonuratha31/LOAN-CREDIT-RIST-PREDICTION-SYSTEM.git

cd LOAN-CREDIT-RIST-PREDICTION-SYSTEM

### 2. Install dependencies

pip install -r requirements.txt

### 3. Train the model

python train.py

(This creates the models folder containing trained model and scaler.)

### 4. Run the application

python app.py

## Project Structure

LOAN-CREDIT-RIST-PREDICTION-SYSTEM
│
├── data/              Dataset files
├── models/            Saved ML models
├── static/            CSS and JavaScript files
├── templates/         HTML files
├── app.py             Flask backend
├── train.py           Model training pipeline
├── requirements.txt   Dependencies
└── README.md

## Model Details

Algorithm Used: Random Forest Classifier

Evaluation Metrics:
• Accuracy
• Precision
• Recall
• F1 Score
• Confusion Matrix
• Cross Validation Stability

## Future Improvements

• Add multiple ML model comparison
• Add user login system
• Add prediction history database
• Docker container deployment
• REST API documentation

## Author

Sonu Ratha

B.Tech Computer Science (AI & ML)
Aspiring Machine Learning Engineer

GitHub: https://github.com/sonuratha31
