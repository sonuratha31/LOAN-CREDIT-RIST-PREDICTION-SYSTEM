import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Load data
print("Loading data for advanced stats...")
df = pd.read_csv('data/loan_detection.csv')
target = 'Loan_Status_label'

# 1. Calculate Correlation Matrix (Top 12 most correlated with target)
print("Calculating Correlation Matrix...")
corr_matrix = df.corr()
target_corr = corr_matrix[target].abs().sort_values(ascending=False)
top_corr_features = target_corr.index[:12].tolist()
small_corr = corr_matrix.loc[top_corr_features, top_corr_features].round(2)

# Features and Target
X = df.drop(columns=[target])
y = df[target]

# Preprocessing
numeric_cols = ['age', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 2. Cross-Validation Stability Check
print("Performing 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), X_scaled, y, cv=skf, scoring='f1')

# Final Train/Test Split for detailed evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
eval_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std(),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'correlation_data': {
        'columns': top_corr_features,
        'values': small_corr.values.tolist()
    }
}

# Save artifacts
print("Saving advanced evaluation artifacts...")
joblib.dump(model, 'models/loan_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(list(X.columns), 'models/feature_names.joblib')
joblib.dump(eval_metrics, 'models/eval_metrics.joblib')

print("Advanced stability and correlation artifacts saved!")
