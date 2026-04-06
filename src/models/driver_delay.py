import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models', 'driver_delay_model.pkl')

FEATURES = [
    'driver_age', 'driver_experience_years', 'total_assigned_rides',
    'accepted_rides', 'incomplete_rides', 'delay_count',
    'acceptance_rate', 'delay_rate', 'avg_driver_rating',
    'avg_pickup_delay_min', 'reliability_score', 'low_acceptance',
]

TARGET = 'driver_delay_flag'


def train(df):
    print("\n=== Model 4: Driver Delay Prediction (Binary) ===")

    df_clean = df.dropna(subset=FEATURES + [TARGET])
    X = df_clean[FEATURES]
    y = df_clean[TARGET].astype(int)

    print(f"  Training on {len(X):,} rows")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    print("  Training...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['On Time', 'Delayed']))

    # Feature importance
    fi = pd.Series(model.feature_importances_,
                   index=FEATURES).sort_values(ascending=False)
    print("  Feature importances:")
    print(fi.to_string())

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURES}, f)
    print(f"\n  ✅ Model saved → saved_models/driver_delay_model.pkl")

    return {
        'model':              model,
        'accuracy':           acc,
        'auc':                auc,
        'confusion_matrix':   confusion_matrix(y_test, y_pred),
        'feature_importance': fi,
    }


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_delay(input_dict):
    obj    = load_model()
    df_inp = pd.DataFrame([input_dict])[obj['features']]
    prob   = obj['model'].predict_proba(df_inp)[0, 1]
    label  = 'Likely Delayed' if prob >= 0.5 else 'On Time'
    return {'label': label, 'probability': round(prob, 4)}

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_cleaning import run_cleaning_pipeline
    from feature_engineering import build_features

    cleaned = run_cleaning_pipeline()
    feats   = build_features(cleaned)
    results = train(feats['drivers_fe'])  