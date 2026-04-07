# Loan Detection AI - ML Project

A comprehensive machine learning solution for predicting loan eligibility based on banking marketing data.

## Features
- **Exploratory Data Analysis**: Pre-processed dataset (already one-hot encoded).
- **Machine Learning**: Random Forest Classifier with imbalanced class handling.
- **Flask API**: High-performance backend for real-time predictions.
- **Premium Frontend**: Modern dark-themed dashboard with Inter font and CSS animations.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn flask flask-cors joblib
   ```

2. **Train the Model**:
   ```bash
   python train.py
   ```
   *Note: This generates the `models/` directory with `loan_model.joblib` and `scaler.joblib`.*

3. **Start the Web App**:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Project Structure
- `data/`: Contains the original CSV dataset.
- `models/`: Serialized ML model and scaler.
- `static/`: Frontend assets (CSS, JS).
- `templates/`: HTML structure.
- `app.py`: Backend server logic.
- `train.py`: Model training pipeline.
