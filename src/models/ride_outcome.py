import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models', 'ride_outcome_model.pkl')

FEATURES = [
    'ride_distance_km', 'estimated_ride_time_min', 'actual_ride_time_min',
    'hour_of_day', 'is_weekend', 'surge_multiplier', 'base_fare',
    'rush_hour_flag', 'long_distance_flag', 'peak_time_flag',
    'city_enc', 'vehicle_type_enc', 'traffic_level_enc', 'weather_condition_enc',
    'day_of_week_enc', 'demand_level_enc',
    'cust_cancel_rate', 'cust_rating', 'total_bookings',
    'avg_driver_rating', 'delay_rate', 'acceptance_rate', 'driver_experience_years',
    'fare_per_km', 'delay_minutes', 'month', 'quarter',
]

TARGET = 'status_enc'

def train(df):
    print("\n=== Model 1: Ride Outcome (Multi-class) ===")

    # Drop rows with any missing values in our feature columns
    df_clean = df.dropna(subset=FEATURES + [TARGET])
    X = df_clean[FEATURES]
    y = df_clean[TARGET].astype(int)

    print(f"  Training on {len(X):,} rows")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    print("  Training... (this takes ~1 min)")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Metrics
    acc   = accuracy_score(y_test, y_pred)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc   = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  AUC(OvR) : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Completed', 'Cancelled', 'Incomplete']))

    # Feature importance — which columns matter most
    fi = pd.Series(model.feature_importances_,
                   index=FEATURES).sort_values(ascending=False)
    print("  Top 10 important features:")
    print(fi.head(10).to_string())

    # Save model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURES}, f)
    print(f"\n  ✅ Model saved → saved_models/ride_outcome_model.pkl")

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

def predict(input_dict):
    obj    = load_model()
    df_inp = pd.DataFrame([input_dict])[obj['features']]
    pred   = obj['model'].predict(df_inp)[0]
    return ['Completed', 'Cancelled', 'Incomplete'][pred]

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_cleaning import run_cleaning_pipeline
    from feature_engineering import build_features

    cleaned = run_cleaning_pipeline()
    feats   = build_features(cleaned)
    results = train(feats['bookings_fe'])

    