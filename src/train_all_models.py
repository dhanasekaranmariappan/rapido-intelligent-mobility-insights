import os
import sys

# This makes sure Python can find all src files
sys.path.insert(0, os.path.dirname(__file__))

from data_cleaning import run_cleaning_pipeline
from feature_engineering import build_features
from database import load_to_db

from models.ride_outcome import train as train_ride_outcome
from models.fare_prediction import train as train_fare
from models.cancellation_risk import train as train_cancel_risk
from models.driver_delay import train as train_driver_delay

def main():
    print('=' * 55)
    print(' RAPIDO INTELLIGENT MOBILITY - TRAINING PIPELINE')
    print('=' * 55)

    # Step 1: Clean all data
    cleaned  = run_cleaning_pipeline()

    # Step 2: Build features
    feats = build_features(cleaned)
    bk = feats['bookings_fe']
    cu = feats['customers_fe']
    dr = feats['drivers_fe']

    # Step 3: Load into SQL database
    load_to_db(cleaned)

    # Step 4: Train all 4 models
    print('----- Training All Models -----')
    results = {}
    results['ride_outcome'] = train_ride_outcome(bk)
    results['fare_prediction'] = train_fare(bk)
    results['cancellation_risk'] = train_cancel_risk(cu)
    results['driver_delay'] = train_driver_delay(dr)

    # Step 5: Print final summary

    print('\n' + '=' * 55)
    print(' TRAINING SUMMARY ')
    print('=' * 55)

    r = results['ride_outcome']
    print(f' Ride Outcome  | Acc: {r['accuracy']:.3f} | AUC: {r["auc"]:.3f}')
    r = results['fare_prediction']
    print(f' Fare Pred     | RMSE: {r['rmse_pct']:.1f}% | R2 Score: {r['r2']:.3f}')
    r = results['cancellation_risk']
    print(f' Cancel Risk   | Acc: {r["accuracy"]:.3f} | AUC: {r["auc"]:.3f}')
    r = results['driver_delay']
    print(f' Driver Delay  | Acc: {r["accuracy"]:.3f} | AUC: {r["auc"]:.3f}')

    print('=' * 55)
    print('\n All models trained and saved successfully!')


if __name__ == '__main__':
    main()