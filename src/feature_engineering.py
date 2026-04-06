import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_features(cleaned):
    print("\n=== Feature Engineering ===")

    bk = cleaned['bookings'].copy()
    cu = cleaned['customers'].copy()
    dr = cleaned['drivers'].copy()
    ld = cleaned['location_demand'].copy()
    tf = cleaned['time_features'].copy()

    # New calculated columns
    bk['fare_per_km']        = bk['booking_value'] / (bk['ride_distance_km'] + 0.01)
    bk['fare_per_min']       = bk['booking_value'] / (bk['actual_ride_time_min'] + 0.01)
    bk['delay_minutes']      = bk['actual_ride_time_min'] - bk['estimated_ride_time_min']
    bk['rush_hour_flag']     = bk['hour_of_day'].apply(
                                   lambda h: 1 if h in list(range(7,10)) + list(range(17,21)) else 0)
    bk['long_distance_flag'] = (bk['ride_distance_km'] > 15).astype(int)
    bk['city_pair']          = bk['pickup_location'] + '_' + bk['drop_location']
    bk['month']              = bk['booking_date'].dt.month
    bk['quarter']            = bk['booking_date'].dt.quarter

    print("  ✅ Booking features created")

    # Add peak_time_flag from time_features table
    peak_map = tf.set_index(['hour_of_day', 'day_of_week'])['peak_time_flag'].to_dict()
    bk['peak_time_flag'] = bk.apply(
        lambda r: peak_map.get((r['hour_of_day'], r['day_of_week']), 0), axis=1)

    # Merge demand info from location_demand table
    bk = bk.merge(
        ld[['city', 'pickup_location', 'hour_of_day', 'vehicle_type',
            'demand_level', 'avg_surge_multiplier']],
        on=['city', 'pickup_location', 'hour_of_day', 'vehicle_type'],
        how='left'
    )
    bk['demand_level']         = bk['demand_level'].fillna('Low')
    bk['avg_surge_multiplier'] = bk['avg_surge_multiplier'].fillna(1.0)

    print("  ✅ Peak time + demand merged")

    # Merge customer features into bookings
    cu_feat = cu[['customer_id', 'cancellation_rate', 'avg_customer_rating',
                  'total_bookings', 'completed_rides', 'customer_signup_days_ago']].copy()
    cu_feat.rename(columns={'avg_customer_rating': 'cust_rating',
                            'cancellation_rate':   'cust_cancel_rate'}, inplace=True)
    bk = bk.merge(cu_feat, on='customer_id', how='left')

    # Merge driver features into bookings
    dr_feat = dr[['driver_id', 'avg_driver_rating', 'delay_rate',
                  'acceptance_rate', 'driver_experience_years']].copy()
    bk = bk.merge(dr_feat, on='driver_id', how='left')

    # Fill any nulls from the merge
    for col in ['cust_rating', 'cust_cancel_rate', 'total_bookings',
                'completed_rides', 'customer_signup_days_ago',
                'avg_driver_rating', 'delay_rate', 'acceptance_rate',
                'driver_experience_years']:
        bk[col] = bk[col].fillna(bk[col].median())

    print("  ✅ Customer + driver info merged")

    # Convert text columns to numbers (ML models need numbers)
    le = LabelEncoder()
    cat_cols = ['city', 'vehicle_type', 'traffic_level', 'weather_condition',
                'day_of_week', 'demand_level', 'pickup_location', 'drop_location']
    for col in cat_cols:
        bk[col + '_enc'] = le.fit_transform(bk[col].astype(str))

    # Encode target: Completed=0, Cancelled=1, Incomplete=2
    status_map = {'Completed': 0, 'Cancelled': 1, 'Incomplete': 2}
    bk['status_enc'] = bk['booking_status'].map(status_map)

    print("  ✅ Categorical columns encoded")

    # Customer-level scores
    cu['loyalty_score']  = cu['completed_rides'] / (cu['total_bookings'] + 1)
    cu['high_risk_flag'] = (cu['cancellation_rate'] > 0.3).astype(int)

    # Driver-level scores
    dr['reliability_score'] = dr['avg_driver_rating'] * (1 - dr['delay_rate'])
    dr['low_acceptance']    = (dr['acceptance_rate'] < 0.5).astype(int)

    print("  ✅ Customer & driver scores created")
    print(f"\n  Bookings final shape: {bk.shape}")
    print("\n✅ Feature engineering done!\n")

    return {
        'bookings_fe':  bk,
        'customers_fe': cu,
        'drivers_fe':   dr,
    }

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import run_cleaning_pipeline

    cleaned = run_cleaning_pipeline()
    feats   = build_features(cleaned)
    print("Bookings columns:", feats['bookings_fe'].columns.tolist())