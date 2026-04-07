from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load artifacts
model = joblib.load('models/loan_model.joblib')
scaler = joblib.load('models/scaler.joblib')
feature_names = joblib.load('models/feature_names.joblib')

# Load eval metrics (may not exist yet, so handle it)
def get_eval_metrics():
    try:
        return joblib.load('models/eval_metrics.joblib')
    except:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received data: {data}")
        
        # Prepare an empty feature row
        row = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Numeric Features (already scaled in our model trainer)
        numeric_cols = ['age', 'campaign', 'pdays', 'previous']
        for col in numeric_cols:
            row.at[0, col] = float(data.get(col, 0))
            
        # Binary / Simple Features
        binary_cols = ['no_previous_contact', 'not_working']
        for col in binary_cols:
            row.at[0, col] = 1 if data.get(col) == 'yes' else 0
            
        # Categorical Features (One-Hot Encoded in our dataset)
        groups = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        for group in groups:
            val = data.get(group)
            if val:
                full_col = f"{group}_{val}"
                if full_col in feature_names:
                    row.at[0, full_col] = 1
        
        # Scale numeric features
        row[numeric_cols] = scaler.transform(row[numeric_cols])
        
        # Make prediction
        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0][1]
        
        result = {
            'loan_approved': int(prediction),
            'probability': float(probability),
            'status': 'Eligible' if prediction == 1 else 'Ineligible',
            'message': 'Congratulations! You are likely to be approved for a loan.' if prediction == 1 else 'Sorry, you are currently not likely to be approved.'
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/stats')
def stats():
    try:
        # Get Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10
        top_features = [feature_names[i] for i in indices]
        top_importances = [float(importances[i]) for i in indices]

        # Dataset stats (counts)
        dataset_path = 'data/loan_detection.csv'
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            counts = df['Loan_Status_label'].value_counts().to_dict()
        else:
            counts = {0: 36548, 1: 4640}

        metrics = get_eval_metrics()
        
        return jsonify({
            'top_features': top_features[::-1],
            'top_importances': top_importances[::-1],
            'distribution': {
                'labels': ['Rejected (0)', 'Approved (1)'],
                'values': [int(counts.get(0, 0)), int(counts.get(1, 0))]
            },
            'metrics': metrics
        })
    except Exception as e:
        print(f"Error in stats: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
