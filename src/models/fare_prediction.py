import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models', 'fare_model.pkl')

FEATURES = [
    'ride_distance_km', 'estimated_ride_time_min', 'surge_multiplier', 'base_fare',
    'hour_of_day', 'is_weekend', 'rush_hour_flag', 'long_distance_flag', 'peak_time_flag',
    'city_enc', 'vehicle_type_enc', 'traffic_level_enc', 'weather_condition_enc',
    'demand_level_enc', 'month', 'quarter',
    'avg_driver_rating', 'driver_experience_years',
]

TARGET = 'booking_value'

def train(df):
    print('\n===== Model 2: Fare Prediction (Regression) =====')

    df_clean = df.dropna(subset=FEATURES + [TARGET])
    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    print(f' Training on {len(X):,} rows')
    print(f' Avg fare: ₹{y.mean():.2f} | Min: ₹{y.min():.2f} | Max: ₹{y.max():.2f}')

    # Split 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    print(' Training...(this takes ~1-2 mins)')
    model.fit(X_train, y_train)

    # Predict 
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse_pct = (rmse / y_test.mean()) * 100


    print(f'\nRMSE     : ₹{rmse:.2f}')
    print(f'RMSE%    : {rmse_pct:.1f}% of avg fare')
    print(f'MAE      : ₹{mae:.2f}')
    print(f'R2 score : {r2:.4f}')

    # feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\n Top 10 important features:')
    print(fi.head(10).to_string())

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURES}, f)
    print(f'\n Model saved -> saved_models/fare_model.pkl')

    return {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmse_pct': rmse_pct,
        'feature_importance': fi,
    }

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)
    
def predict(input_dict):
    obj = load_model()
    df_inp = pd.DataFrame([input_dict])[obj['features']]
    fare = obj['model'].predict(df_inp)[0]
    return round(float(fare), 2)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_cleaning import run_cleaning_pipeline
    from feature_engineering import build_features

    cleaned = run_cleaning_pipeline()
    feats = build_features(cleaned)
    result = train(feats['bookings_fe'])