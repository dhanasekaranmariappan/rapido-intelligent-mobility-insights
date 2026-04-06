import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models', 'cancellation_risk_model.pkl')

FEATURES = [
    'customer_age', 'customer_signup_days_ago', 'total_bookings',
    'completed_rides', 'cancelled_rides', 'incomplete_rides',
    'cancellation_rate', 'avg_customer_rating', 'loyalty_score', 'high_risk_flag',
]

TARGET = 'customer_cancel_flag'

def train(df):
    print('\n===== Model 3: Customer cancellation risk [Binary Classification] =====')

    df_clean = df.dropna(subset=FEATURES + [TARGET])
    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    print(f' Training on {len(X):,} rows')
    print(f' Class distribution: {y.value_counts().to_string()}')

    # Split 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    print(' Training...')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f' Accuracy : {acc:.4f}')
    print(f' AUC-ROC  : {auc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    # Feature importance

    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(' Feature importance: ')
    print(fi.to_string())

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURES}, f)
    print(f' Model saved -> saved_models/cancellation_risk_model.pkl')

    return {
        'model': model,
        'accuracy': acc,
        'auc': auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': fi,
    }

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)
    
def predict_risk(input_dict):
    obj = load_model()
    df_inp = pd.DataFrame([input_dict])[obj['features']]
    prob = obj['model'].predict_proba(df_inp)[0, 1]
    label = 'High Risk' if prob >=0.5 else 'Low Risk'
    return {'label': label, 'probability': round(prob, 4)}

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_cleaning import run_cleaning_pipeline
    from feature_engineering import build_features

    cleaned = run_cleaning_pipeline()
    feats = build_features(cleaned)
    results = train(feats['customers_fe'])