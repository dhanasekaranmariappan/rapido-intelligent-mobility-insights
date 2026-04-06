import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'rapido.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_schema(conn):
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            customer_gender TEXT,
            customer_age INTEGER,
            customer_city TEXT,
            customer_signup_days_ago INTEGER,
            preferred_vehicle_type TEXT,
            total_bookings INTEGER,
            completed_rides INTEGER,
            cancelled_rides INTEGER,
            incomplete_rides INTEGER,
            cancellation_rate REAL,
            avg_customer_rating REAL,
            customer_cancel_flag INTEGER
        );

        CREATE TABLE IF NOT EXISTS drivers (
            driver_id TEXT PRIMARY KEY,
            driver_age INTEGER,
            driver_city TEXT,
            vehicle_type TEXT,
            driver_experience_years REAL,
            total_assigned_rides INTEGER,
            accepted_rides INTEGER,
            incomplete_rides INTEGER,
            delay_count INTEGER,
            acceptance_rate REAL,
            delay_rate REAL,
            avg_driver_rating REAL,
            avg_pickup_delay_min REAL,
            driver_delay_flag INTEGER
        );

        CREATE TABLE IF NOT EXISTS bookings (
            booking_id TEXT PRIMARY KEY,
            booking_date TEXT,
            booking_time TEXT,
            day_of_week TEXT,
            is_weekend INTEGER,
            hour_of_day INTEGER,
            city TEXT,
            pickup_location TEXT,
            drop_location TEXT,
            vehicle_type TEXT,
            ride_distance_km REAL,
            estimated_ride_time_min REAL,
            actual_ride_time_min REAL,
            traffic_level TEXT,
            weather_condition TEXT,
            base_fare REAL,
            surge_multiplier REAL,
            booking_value REAL,
            booking_status TEXT,
            incomplete_ride_reason TEXT,
            customer_id TEXT,
            driver_id TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (driver_id)   REFERENCES drivers(driver_id)
        );

        CREATE TABLE IF NOT EXISTS location_demand (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            pickup_location TEXT,
            hour_of_day INTEGER,
            vehicle_type TEXT,
            total_requests INTEGER,
            completed_rides INTEGER,
            cancelled_rides INTEGER,
            avg_wait_time_min REAL,
            avg_surge_multiplier REAL,
            demand_level TEXT
        );

        CREATE TABLE IF NOT EXISTS time_features (
            datetime TEXT PRIMARY KEY,
            hour_of_day INTEGER,
            day_of_week TEXT,
            is_weekend INTEGER,
            is_holiday INTEGER,
            peak_time_flag INTEGER,
            season TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_bookings_customer ON bookings(customer_id);
        CREATE INDEX IF NOT EXISTS idx_bookings_driver   ON bookings(driver_id);
        CREATE INDEX IF NOT EXISTS idx_bookings_city     ON bookings(city);
        CREATE INDEX IF NOT EXISTS idx_bookings_status   ON bookings(booking_status);
        CREATE INDEX IF NOT EXISTS idx_bookings_date     ON bookings(booking_date);
    """)
    conn.commit()
    print("  ✅ Tables + indexes created")

def load_to_db(cleaned):
    print("\n=== Step 3: SQL Database ===")
    conn = get_connection()
    create_schema(conn)

    table_map = {
        'customers':       'customers',
        'drivers':         'drivers',
        'bookings':        'bookings',
        'location_demand': 'location_demand',
        'time_features':   'time_features',
    }

    for key, table in table_map.items():
        df = cleaned[key].copy()
        # SQLite doesn't support datetime type — convert to string
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].astype(str)
        df.to_sql(table, conn, if_exists='replace', index=False)
        print(f"  ✅ Loaded {table}: {len(df)} rows")

    conn.close()
    print(f"\n✅ Database saved → rapido.db\n")

def run_sample_queries():
    print("\n=== Sample SQL Queries ===")
    conn = get_connection()

    queries = {
        "Bookings by status": """
            SELECT booking_status, COUNT(*) as total
            FROM bookings
            GROUP BY booking_status
            ORDER BY total DESC
        """,
        "Avg fare by vehicle": """
            SELECT vehicle_type, ROUND(AVG(booking_value), 2) as avg_fare
            FROM bookings
            GROUP BY vehicle_type
        """,
        "Cancellation rate by city": """
            SELECT city,
                   COUNT(*) as total,
                   SUM(CASE WHEN booking_status='Cancelled' THEN 1 ELSE 0 END) as cancelled,
                   ROUND(100.0 * SUM(CASE WHEN booking_status='Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 1) as cancel_pct
            FROM bookings
            GROUP BY city
            ORDER BY cancel_pct DESC
        """,
    }

    for title, sql in queries.items():
        print(f"\n--- {title} ---")
        print(pd.read_sql(sql, conn).to_string(index=False))

    conn.close()

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import run_cleaning_pipeline

    cleaned = run_cleaning_pipeline()
    load_to_db(cleaned)
    run_sample_queries()