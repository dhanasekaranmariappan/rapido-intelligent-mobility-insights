import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="Rapido Mobility Insights",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE       = os.path.join(os.path.dirname(__file__), '..')
DB_PATH    = os.path.join(BASE, 'rapido.db')
MODELS_DIR = os.path.join(BASE, 'saved_models')

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    .card-green  { background: #d4edda; border-left: 6px solid #28a745; }
    .card-red    { background: #f8d7da; border-left: 6px solid #dc3545; }
    .card-yellow { background: #fff3cd; border-left: 6px solid #ffc107; }
    .card-blue   { background: #d1ecf1; border-left: 6px solid #17a2b8; }

    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 13px; color: #6c757d; margin-top: 4px; }

    .section-header {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 18px;
    }

    .input-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }

    .insight-box {
        background: #eef2ff;
        border-left: 4px solid #4f46e5;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 14px;
        color: #3730a3;
    }

    .prob-bar-wrap {
        background: #e9ecef;
        border-radius: 20px;
        height: 12px;
        margin: 6px 0;
    }
    .prob-bar-fill {
        height: 12px;
        border-radius: 20px;
    }

    section[data-testid="stSidebar"] {
        background: #1a1a2e;
        color: white;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loaders ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    bk = pd.read_sql("SELECT * FROM bookings",        con=conn)
    cu = pd.read_sql("SELECT * FROM customers",       con=conn)
    dr = pd.read_sql("SELECT * FROM drivers",         con=conn)
    ld = pd.read_sql("SELECT * FROM location_demand", con=conn)
    conn.close()
    bk['booking_date'] = pd.to_datetime(bk['booking_date'], errors='coerce')
    return bk, cu, dr, ld

@st.cache_resource
def load_models():
    models = {}
    for name in ['ride_outcome_model', 'fare_model',
                 'cancellation_risk_model', 'driver_delay_model']:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    return models

bk, cu, dr, ld = load_data()
models = load_models()

ENC = {
    'city':          {'Bangalore':0,'Chennai':1,'Delhi':2,'Hyderabad':3,'Mumbai':4},
    'vehicle_type':  {'Auto':0,'Bike':1,'Cab':2},
    'traffic_level': {'High':0,'Low':1,'Medium':2},
    'weather':       {'Clear':0,'Fog':1,'Heavy Rain':2,'Rain':3},
    'demand_level':  {'High':0,'Low':1,'Medium':2},
    'day_of_week':   {'Friday':0,'Monday':1,'Saturday':2,
                      'Sunday':3,'Thursday':4,'Tuesday':5,'Wednesday':6},
}


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🛵 Rapido Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 EDA Dashboard",
    "🔮 Ride Outcome Predictor",
    "💰 Fare Estimator",
    "⚠️ Customer Risk",
    "🚗 Driver Reliability",
])
st.sidebar.divider()
st.sidebar.markdown(f"**Total Bookings:** {len(bk):,}")
st.sidebar.markdown(f"**Customers:** {len(cu):,} | **Drivers:** {len(dr):,}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛵 Rapido Intelligent Mobility Insights")
    st.caption("Ride Patterns · Cancellations · Fare Forecasting")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bookings",  f"{len(bk):,}")
    c2.metric("Completed",
              f"{(bk['booking_status']=='Completed').sum():,}",
              f"{100*(bk['booking_status']=='Completed').mean():.1f}%")
    c3.metric("Cancelled",
              f"{(bk['booking_status']=='Cancelled').sum():,}",
              f"-{100*(bk['booking_status']=='Cancelled').mean():.1f}%")
    c4.metric("Avg Fare",    f"₹{bk['booking_value'].mean():.0f}")
    c5.metric("Avg Distance", f"{bk['ride_distance_km'].mean():.1f} km")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Booking Status Distribution")
        st.bar_chart(bk['booking_status'].value_counts())
    with col2:
        st.subheader("Avg Fare by Vehicle Type")
        st.bar_chart(bk.groupby('vehicle_type')['booking_value'].mean().sort_values())

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Rides by City")
        st.bar_chart(bk['city'].value_counts())
    with col4:
        st.subheader("Model Performance Summary")
        perf = pd.DataFrame({
            'Model':  ['Ride Outcome','Fare Prediction','Cancel Risk','Driver Delay'],
            'Metric': ['Accuracy 92.5%','RMSE 3.4%','AUC 1.00','AUC 1.00'],
            'Status': ['✅','✅','✅','✅'],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs([
        "⏰ Time Patterns", "🗺️ City Analysis",
        "💸 Fare Analysis",  "🌦️ External Factors"
    ])

    with tab1:
        st.subheader("Ride Volume by Hour of Day")
        hourly = bk.groupby(['hour_of_day','booking_status']).size().unstack(fill_value=0)
        st.area_chart(hourly)

        st.subheader("Cancellations by Hour")
        st.line_chart(bk[bk['booking_status']=='Cancelled']
                        .groupby('hour_of_day').size())

        st.subheader("Rides by Day of Week")
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = bk.groupby('day_of_week').size().reindex(dow_order).fillna(0)
        st.bar_chart(dow)

    with tab2:
        st.subheader("City Statistics")
        city_stats = bk.groupby('city').apply(lambda x: pd.Series({
            'Total':        len(x),
            'Completed':    (x['booking_status']=='Completed').sum(),
            'Cancelled':    (x['booking_status']=='Cancelled').sum(),
            'Cancel Rate%': round(100*(x['booking_status']=='Cancelled').mean(), 1),
            'Avg Fare ₹':   round(x['booking_value'].mean(), 0),
        })).reset_index()
        st.dataframe(city_stats, use_container_width=True, hide_index=True)

        st.subheader("Vehicle Mix by City")
        st.bar_chart(bk.groupby(['city','vehicle_type']).size().unstack(fill_value=0))

        city_sel = st.selectbox("Select City for Demand Heatmap", bk['city'].unique())
        ld_city  = ld[ld['city']==city_sel].groupby(
                     ['pickup_location','hour_of_day'])['total_requests'].sum().unstack(fill_value=0)
        if not ld_city.empty:
            st.subheader(f"Demand Heatmap — {city_sel}")
            st.dataframe(ld_city.style.background_gradient(cmap='YlOrRd'),
                         use_container_width=True)

    with tab3:
        st.subheader("Avg Fare by City & Vehicle")
        fare_hm = bk.groupby(['city','vehicle_type'])['booking_value'].mean().unstack()
        st.dataframe(fare_hm.style.background_gradient(cmap='Blues').format('₹{:.0f}'),
                     use_container_width=True)

        st.subheader("Surge Multiplier Distribution")
        st.bar_chart(bk['surge_multiplier'].value_counts().sort_index())

        st.subheader("Fare per KM by Vehicle")
        bk_tmp = bk.copy()
        bk_tmp['fare_per_km'] = bk_tmp['booking_value'] / (bk_tmp['ride_distance_km'] + 0.01)
        st.bar_chart(bk_tmp.groupby('vehicle_type')['fare_per_km'].mean())

    with tab4:
        st.subheader("Cancellation Rate by Weather")
        wx = bk.groupby('weather_condition').apply(
             lambda x: round(100*(x['booking_status']=='Cancelled').mean(), 1))
        st.bar_chart(wx)

        st.subheader("Cancellation Rate by Traffic Level")
        trf = bk.groupby('traffic_level').apply(
              lambda x: round(100*(x['booking_status']=='Cancelled').mean(), 1))
        st.bar_chart(trf)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Ratings")
            st.bar_chart(cu['avg_customer_rating'].value_counts().sort_index())
        with c2:
            st.subheader("Driver Ratings")
            st.bar_chart(dr['avg_driver_rating'].value_counts().sort_index())


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RIDE OUTCOME PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Ride Outcome Predictor":
    st.markdown('<div class="section-header">🔮 Ride Outcome Predictor</div>',
                unsafe_allow_html=True)
    st.markdown("Predict if a booking will be **Completed**, **Cancelled**, or **Incomplete** before the ride starts.")
    st.divider()

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚦 Ride Details**")
        city     = st.selectbox("📍 City", list(ENC['city'].keys()))
        vehicle  = st.selectbox("🛵 Vehicle Type", list(ENC['vehicle_type'].keys()))
        distance = st.slider("📏 Distance (km)", 1.0, 50.0, 8.0, 0.5)
        est_time = st.slider("⏱️ Estimated Ride Time (min)", 5, 120, 30)
        hour     = st.slider("🕐 Hour of Day", 0, 23, 9)
        traffic  = st.selectbox("🚗 Traffic Level", list(ENC['traffic_level'].keys()))
    with col2:
        st.markdown("**🌦️ Conditions & Behaviour**")
        weather   = st.selectbox("🌧️ Weather", list(ENC['weather'].keys()))
        surge     = st.slider("⚡ Surge Multiplier", 1.0, 3.0, 1.5, 0.1)
        base_fare = st.slider("💵 Base Fare (₹)", 20, 500, 150)
        cust_cr   = st.slider("👤 Customer Cancel Rate", 0.0, 1.0, 0.1, 0.01)
        drv_delay = st.slider("🚘 Driver Delay Rate", 0.0, 1.0, 0.1, 0.01)
        is_wknd   = st.checkbox("📅 Is Weekend?")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict Now", use_container_width=True, type="primary"):
        dow = datetime.datetime.now().strftime('%A')
        inp = {
            'ride_distance_km':        distance,
            'estimated_ride_time_min': est_time,
            'actual_ride_time_min':    est_time,
            'hour_of_day':             hour,
            'is_weekend':              int(is_wknd),
            'surge_multiplier':        surge,
            'base_fare':               base_fare,
            'rush_hour_flag':          1 if hour in list(range(7,10))+list(range(17,21)) else 0,
            'long_distance_flag':      1 if distance > 15 else 0,
            'peak_time_flag':          0,
            'city_enc':                ENC['city'].get(city, 0),
            'vehicle_type_enc':        ENC['vehicle_type'].get(vehicle, 0),
            'traffic_level_enc':       ENC['traffic_level'].get(traffic, 1),
            'weather_condition_enc':   ENC['weather'].get(weather, 0),
            'day_of_week_enc':         ENC['day_of_week'].get(dow, 0),
            'demand_level_enc':        1,
            'cust_cancel_rate':        cust_cr,
            'cust_rating':             4.0,
            'total_bookings':          20,
            'avg_driver_rating':       4.0,
            'delay_rate':              drv_delay,
            'acceptance_rate':         0.7,
            'driver_experience_years': 3.0,
            'fare_per_km':             (base_fare * surge) / (distance + 0.01),
            'delay_minutes':           0,
            'month':                   datetime.datetime.now().month,
            'quarter':                 (datetime.datetime.now().month - 1) // 3 + 1,
        }
        m = models.get('ride_outcome_model')
        if m:
            df_inp = pd.DataFrame([inp])[m['features']]
            pred   = m['model'].predict(df_inp)[0]
            probs  = m['model'].predict_proba(df_inp)[0]
            label  = ['Completed', 'Cancelled', 'Incomplete'][pred]
            card   = ['card-green', 'card-red', 'card-yellow'][pred]
            icon   = ['✅', '❌', '⚠️'][pred]

            st.divider()
            st.markdown(f"""
            <div class="result-card {card}">
                <h2>{icon} Predicted Outcome: {label}</h2>
                <p style="font-size:16px; color:#555;">
                    Model confidence based on ride conditions and behaviour history
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Probability Breakdown")
            for lbl, prob, color in [
                ('✅ Completed',  probs[0], '#28a745'),
                ('❌ Cancelled',  probs[1], '#dc3545'),
                ('⚠️ Incomplete', probs[2], '#ffc107'),
            ]:
                st.markdown(f"""
                <div style="margin:8px 0;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-weight:500">{lbl}</span>
                        <span style="font-weight:bold">{prob*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill"
                             style="width:{prob*100:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            tips = {
                'Completed':  "✅ High chance of completion. Assign a reliable driver.",
                'Cancelled':  "⚠️ High cancellation risk! Consider offering a discount or sending a top-rated driver.",
                'Incomplete': "🔔 Incomplete ride risk detected. Monitor this booking closely.",
            }
            st.markdown(f'<div class="insight-box">💡 {tips[label]}</div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FARE ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰 Fare Estimator":
    st.markdown('<div class="section-header">💰 Dynamic Fare Estimator</div>',
                unsafe_allow_html=True)
    st.markdown("Estimate the exact fare before booking confirmation.")
    st.divider()

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚦 Trip Details**")
        city      = st.selectbox("📍 City", list(ENC['city'].keys()))
        vehicle   = st.selectbox("🛵 Vehicle Type", list(ENC['vehicle_type'].keys()))
        distance  = st.slider("📏 Distance (km)", 1.0, 50.0, 10.0, 0.5)
        est_time  = st.slider("⏱️ Est. Ride Time (min)", 5, 120, 25)
        base_fare = st.slider("💵 Base Fare (₹)", 20, 500, 120)
    with col2:
        st.markdown("**⚡ Pricing Factors**")
        surge   = st.slider("⚡ Surge Multiplier", 1.0, 3.0, 1.2, 0.1)
        traffic = st.selectbox("🚗 Traffic Level", list(ENC['traffic_level'].keys()))
        weather = st.selectbox("🌧️ Weather", list(ENC['weather'].keys()))
        hour    = st.slider("🕐 Hour of Day", 0, 23, 10)
        is_wknd = st.checkbox("📅 Weekend?")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💰 Estimate Fare", use_container_width=True, type="primary"):
        inp = {
            'ride_distance_km':        distance,
            'estimated_ride_time_min': est_time,
            'surge_multiplier':        surge,
            'base_fare':               base_fare,
            'hour_of_day':             hour,
            'is_weekend':              int(is_wknd),
            'rush_hour_flag':          1 if hour in list(range(7,10))+list(range(17,21)) else 0,
            'long_distance_flag':      1 if distance > 15 else 0,
            'peak_time_flag':          0,
            'city_enc':                ENC['city'].get(city, 0),
            'vehicle_type_enc':        ENC['vehicle_type'].get(vehicle, 0),
            'traffic_level_enc':       ENC['traffic_level'].get(traffic, 1),
            'weather_condition_enc':   ENC['weather'].get(weather, 0),
            'demand_level_enc':        1,
            'month':                   datetime.datetime.now().month,
            'quarter':                 (datetime.datetime.now().month - 1) // 3 + 1,
            'avg_driver_rating':       4.2,
            'driver_experience_years': 3.0,
        }
        m = models.get('fare_model')
        if m:
            df_inp = pd.DataFrame([inp])[m['features']]
            fare   = float(m['model'].predict(df_inp)[0])
            lo, hi = fare * 0.9, fare * 1.1

            st.divider()
            st.markdown(f"""
            <div class="result-card card-blue">
                <h1 style="color:#0c5460; font-size:48px;">₹{fare:.2f}</h1>
                <p style="color:#555; font-size:16px;">Estimated Fare</p>
                <p style="color:#888; font-size:13px;">
                    Expected range: ₹{lo:.0f} – ₹{hi:.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Fare Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""<div class="metric-box">
                <div class="metric-value">₹{fare:.2f}</div>
                <div class="metric-label">Total Fare</div></div>""",
                unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-box">
                <div class="metric-value">₹{fare/distance:.2f}</div>
                <div class="metric-label">Per KM</div></div>""",
                unsafe_allow_html=True)
            c3.markdown(f"""<div class="metric-box">
                <div class="metric-value">₹{fare/max(est_time,1):.2f}</div>
                <div class="metric-label">Per Min</div></div>""",
                unsafe_allow_html=True)
            c4.markdown(f"""<div class="metric-box">
                <div class="metric-value">{surge}x</div>
                <div class="metric-label">Surge Applied</div></div>""",
                unsafe_allow_html=True)

            surge_msg = "🔴 High surge active — consider booking later." if surge > 2.0 \
                   else "🟡 Moderate surge — prices slightly elevated." if surge > 1.5 \
                   else "🟢 Low surge — great time to book!"
            st.markdown(
                f'<div class="insight-box">💡 {surge_msg} '
                f'Base fare ₹{base_fare} × {surge}x surge = ₹{base_fare*surge:.0f} before distance adjustment.</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER RISK
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Customer Risk":
    st.markdown('<div class="section-header">⚠️ Customer Cancellation Risk</div>',
                unsafe_allow_html=True)
    st.markdown("Identify customers who are likely to cancel their booking.")
    st.divider()

    tab1, tab2 = st.tabs(["🔍 Live Prediction", "📋 High-Risk Customers"])

    with tab1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age    = st.slider("👤 Customer Age", 18, 70, 30)
            signup = st.slider("📅 Days Since Signup", 1, 2000, 300)
            total  = st.slider("📦 Total Bookings", 1, 200, 20)
            comp   = st.slider("✅ Completed Rides", 0, 200, 15)
        with col2:
            canc   = st.slider("❌ Cancelled Rides", 0, 100, 3)
            incomp = st.slider("⚠️ Incomplete Rides", 0, 50, 1)
            cr     = st.slider("📉 Cancellation Rate", 0.0, 1.0, 0.15, 0.01)
            rating = st.slider("⭐ Avg Rating", 1.0, 5.0, 4.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        loyalty   = comp / (total + 1)
        high_risk = 1 if cr > 0.3 else 0

        if st.button("⚠️ Assess Risk", use_container_width=True, type="primary"):
            inp = {
                'customer_age':            age,
                'customer_signup_days_ago':signup,
                'total_bookings':          total,
                'completed_rides':         comp,
                'cancelled_rides':         canc,
                'incomplete_rides':        incomp,
                'cancellation_rate':       cr,
                'avg_customer_rating':     rating,
                'loyalty_score':           loyalty,
                'high_risk_flag':          high_risk,
            }
            m = models.get('cancellation_risk_model')
            if m:
                df_inp = pd.DataFrame([inp])[m['features']]
                prob   = m['model'].predict_proba(df_inp)[0, 1]
                is_hr  = prob >= 0.5
                card   = 'card-red' if is_hr else 'card-green'
                label  = 'HIGH RISK 🔴' if is_hr else 'LOW RISK 🟢'
                color  = '#dc3545' if is_hr else '#28a745'

                st.divider()
                st.markdown(f"""
                <div class="result-card {card}">
                    <h2>{label}</h2>
                    <h3 style="color:#555;">Cancel Probability: {prob*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin:15px 0;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-weight:500">Cancellation Risk</span>
                        <span style="font-weight:bold">{prob*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill"
                             style="width:{prob*100:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:{color};">{prob*100:.1f}%</div>
                    <div class="metric-label">Cancel Probability</div></div>""",
                    unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{loyalty:.2f}</div>
                    <div class="metric-label">Loyalty Score</div></div>""",
                    unsafe_allow_html=True)

                tip = "🔴 High-risk customer! Consider a reminder or small discount to retain them." \
                      if is_hr else \
                      "🟢 Low-risk customer. Likely to complete the ride without issues."
                st.markdown(f'<div class="insight-box">💡 {tip}</div>',
                            unsafe_allow_html=True)

    with tab2:
        st.subheader("High-Risk Customers (cancel_flag = 1)")
        hr = cu[cu['customer_cancel_flag']==1][
            ['customer_id','customer_city','customer_age',
             'cancellation_rate','avg_customer_rating']
        ].sort_values('cancellation_rate', ascending=False)
        st.dataframe(hr.head(50), use_container_width=True, hide_index=True)

        st.subheader("Risk Rate by City")
        st.bar_chart(cu.groupby('customer_city')['customer_cancel_flag']
                       .mean().sort_values(ascending=False))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DRIVER RELIABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚗 Driver Reliability":
    st.markdown('<div class="section-header">🚗 Driver Reliability Scoring</div>',
                unsafe_allow_html=True)
    st.markdown("Predict which drivers are likely to cause delays or incomplete rides.")
    st.divider()

    tab1, tab2 = st.tabs(["🔍 Live Prediction", "📋 Driver Summary"])

    with tab1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age   = st.slider("👤 Driver Age", 20, 60, 32)
            exp   = st.slider("📅 Experience (yrs)", 0, 15, 3, 1)
            total = st.slider("📦 Total Assigned Rides", 1, 500, 80)
            acc   = st.slider("✅ Accepted Rides", 1, 500, 60)
        with col2:
            incomp  = st.slider("⚠️ Incomplete Rides", 0, 100, 5)
            dc      = st.slider("🕐 Delay Count", 0, 100, 8)
            ar      = st.slider("📊 Acceptance Rate", 0.0, 1.0, 0.75, 0.01)
            dr_rate = st.slider("📉 Delay Rate", 0.0, 1.0, 0.1, 0.01)
            rating  = st.slider("⭐ Avg Rating", 1.0, 5.0, 4.2, 0.1)
            pickup  = st.slider("⏱️ Avg Pickup Delay (min)", 0.0, 30.0, 5.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        rel     = rating * (1 - dr_rate)
        low_acc = 1 if ar < 0.5 else 0

        if st.button("🚗 Check Reliability", use_container_width=True, type="primary"):
            inp = {
                'driver_age':              age,
                'driver_experience_years': exp,
                'total_assigned_rides':    total,
                'accepted_rides':          acc,
                'incomplete_rides':        incomp,
                'delay_count':             dc,
                'acceptance_rate':         ar,
                'delay_rate':              dr_rate,
                'avg_driver_rating':       rating,
                'avg_pickup_delay_min':    pickup,
                'reliability_score':       rel,
                'low_acceptance':          low_acc,
            }
            m = models.get('driver_delay_model')
            if m:
                df_inp = pd.DataFrame([inp])[m['features']]
                prob   = m['model'].predict_proba(df_inp)[0, 1]
                is_del = prob >= 0.5
                card   = 'card-red' if is_del else 'card-green'
                label  = '🔴 LIKELY DELAYED' if is_del else '🟢 RELIABLE DRIVER'
                color  = '#dc3545' if is_del else '#28a745'

                st.divider()
                st.markdown(f"""
                <div class="result-card {card}">
                    <h2>{label}</h2>
                    <h3 style="color:#555;">Delay Probability: {prob*100:.1f}%</h3>
                    <p style="color:#777;">Reliability Score: {rel:.2f} / 5.0</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin:15px 0;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-weight:500">Delay Risk</span>
                        <span style="font-weight:bold">{prob*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill"
                             style="width:{prob*100:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:{color};">{prob*100:.1f}%</div>
                    <div class="metric-label">Delay Probability</div></div>""",
                    unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{rel:.2f}</div>
                    <div class="metric-label">Reliability Score</div></div>""",
                    unsafe_allow_html=True)
                c3.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{rating}⭐</div>
                    <div class="metric-label">Driver Rating</div></div>""",
                    unsafe_allow_html=True)

                tip = "🔴 High delay risk! Assign a backup driver or notify the customer early." \
                      if is_del else \
                      "🟢 Reliable driver. Safe to assign this booking."
                st.markdown(f'<div class="insight-box">💡 {tip}</div>',
                            unsafe_allow_html=True)

    with tab2:
        st.subheader("Delayed Drivers (delay_flag = 1)")
        bd = dr[dr['driver_delay_flag']==1][
            ['driver_id','driver_city','vehicle_type',
             'delay_rate','avg_driver_rating','avg_pickup_delay_min']
        ].sort_values('delay_rate', ascending=False)
        st.dataframe(bd.head(50), use_container_width=True, hide_index=True)

        st.subheader("Delay Rate by City")
        st.bar_chart(dr.groupby('driver_city')['delay_rate']
                       .mean().sort_values(ascending=False))

        st.subheader("Avg Rating by Vehicle Type")
        st.bar_chart(dr.groupby('vehicle_type')['avg_driver_rating'].mean())