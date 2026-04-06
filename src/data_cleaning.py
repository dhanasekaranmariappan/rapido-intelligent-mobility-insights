import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_raw_data():
    files = {
        'bookings':       'bookings.csv',
        'customers':      'customers.csv',
        'drivers':        'drivers.csv',
        'location_demand':'location_demand.csv',
        'time_features':  'time_features.csv',
    }
    dfs = {}
    for key, fname in files.items():
        dfs[key] = pd.read_csv(os.path.join(RAW_DIR, fname))
        print(f"Loaded {key}: {dfs[key].shape}")
    return dfs

def clean_bookings(df):
    df = df.copy()
    df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce')
    df['actual_ride_time_min'] = df['actual_ride_time_min'].fillna(df['estimated_ride_time_min'])
    df['incomplete_ride_reason'] = df['incomplete_ride_reason'].fillna('None')
    for col in ['ride_distance_km', 'estimated_ride_time_min', 'actual_ride_time_min',
                'base_fare', 'surge_multiplier', 'booking_value']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['booking_value'], inplace=True)
    df['is_weekend'] = df['is_weekend'].astype(int)
    print(f"Bookings cleaned: {df.shape}")
    return df

def clean_customers(df):
    df = df.copy()
    df['cancellation_rate'] = pd.to_numeric(df['cancellation_rate'], errors='coerce').fillna(0)
    df['avg_customer_rating'] = df['avg_customer_rating'].fillna(df['avg_customer_rating'].median())
    df['customer_cancel_flag'] = df['customer_cancel_flag'].astype(int)
    print(f"Customers cleaned: {df.shape}")
    return df

def clean_drivers(df):
    df = df.copy()
    df['delay_rate'] = pd.to_numeric(df['delay_rate'], errors='coerce').fillna(0)
    df['avg_driver_rating'] = df['avg_driver_rating'].fillna(df['avg_driver_rating'].median())
    df['driver_delay_flag'] = df['driver_delay_flag'].astype(int)
    print(f"Drivers cleaned: {df.shape}")
    return df

def clean_location_demand(df):
    df = df.copy()
    df['avg_surge_multiplier'] = pd.to_numeric(df['avg_surge_multiplier'], errors='coerce').fillna(1.0)
    print(f"Location demand cleaned: {df.shape}")
    return df

def clean_time_features(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    for col in ['is_weekend', 'is_holiday', 'peak_time_flag']:
        df[col] = df[col].astype(int)
    print(f"Time features cleaned: {df.shape}")
    return df

def run_cleaning_pipeline():
    print("\n=== Data Cleaning ===")
    raw = load_raw_data()
    cleaned = {
        'bookings':        clean_bookings(raw['bookings']),
        'customers':       clean_customers(raw['customers']),
        'drivers':         clean_drivers(raw['drivers']),
        'location_demand': clean_location_demand(raw['location_demand']),
        'time_features':   clean_time_features(raw['time_features']),
    }
    print("\n✅ Cleaning done!\n")
    return cleaned

if __name__ == '__main__':
    run_cleaning_pipeline()