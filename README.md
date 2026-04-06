# 🛵 Rapido: Intelligent Mobility Insights

> Ride Patterns · Cancellations · Fare Forecasting · Driver Reliability

A complete end-to-end Machine Learning project built on real-scale ride-hailing
data (100,000 bookings) to predict ride outcomes, estimate fares dynamically,
identify high-risk customers, and score driver reliability.

---

## 🎯 Problem Statement

Rapido operates a large-scale ride-hailing platform where millions of bookings
are created daily. Key challenges include:

- Ride cancellations reducing revenue
- Inaccurate fare estimation
- Inefficient driver allocation
- Poor customer experience during peak demand

This project builds a **unified ML-driven decision system** to solve all four.

---

## 📊 Dataset

| File | Description | Rows |
|------|-------------|------|
| `bookings.csv` | Core transactional data with booking outcomes | 100,000 |
| `customers.csv` | Customer behaviour & cancellation history | 10,000 |
| `drivers.csv` | Driver performance & reliability metrics | 5,000 |
| `location_demand.csv` | Demand patterns by location & time | 17,941 |
| `time_features.csv` | Temporal signals (hour, weekday, peaks) | 8,760 |

---

## 🤖 ML Models Built

| Model | Type | Algorithm | Result |
|-------|------|-----------|--------|
| Ride Outcome | Multi-class Classification | Random Forest | **Accuracy 92.5%** |
| Fare Prediction | Regression | Gradient Boosting | **R² 0.997 \| RMSE 3.4%** |
| Cancellation Risk | Binary Classification | Gradient Boosting | **AUC 1.00** |
| Driver Delay | Binary Classification | Random Forest | **AUC 1.00** |

All models exceed industry benchmarks (85–90% accuracy, RMSE ≤ 10%).

---

## 🗂️ Project Structure

## 🗂️ Project Structure
```
rapido_insight/
│
├── 📂 data/
│   ├── bookings.csv
│   ├── customers.csv
│   ├── drivers.csv
│   ├── location_demand.csv
│   └── time_features.csv
│
├── 📂 src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── database.py
│   ├── train_all_models.py
│   └── 📂 models/
│       ├── ride_outcome.py
│       ├── fare_prediction.py
│       ├── cancellation_risk.py
│       └── driver_delay.py
│
├── 📂 saved_models/
│   ├── ride_outcome_model.pkl
│   ├── fare_model.pkl
│   ├── cancellation_risk_model.pkl
│   └── driver_delay_model.pkl
│
├── 📂 streamlit_app/
│   └── app.py
│
├── rapido.db
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rapido-intelligent-mobility-insights.git
cd rapido-intelligent-mobility-insights
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
Place all 5 CSV files inside the `data/` folder.

---

## 🚀 How to Run

### Train all models (run once)
```bash
python src/train_all_models.py
```

This will:
- Clean all data
- Engineer features
- Load data into SQLite
- Train all 4 ML models
- Save models to `saved_models/`

### Launch the Streamlit dashboard
```bash
streamlit run streamlit_app/app.py
```

Open your browser at `http://localhost:8501`

---

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Key metrics, status distribution, city stats |
| 📊 EDA Dashboard | Time patterns, city analysis, fare & weather analysis |
| 🔮 Ride Outcome Predictor | Predict ride result before trip starts |
| 💰 Fare Estimator | Estimate fare dynamically with surge & conditions |
| ⚠️ Customer Risk | Flag high-risk customers likely to cancel |
| 🚗 Driver Reliability | Score driver reliability before assignment |

---

## 🔑 Key Features Engineered

| Feature | Description |
|---------|-------------|
| `fare_per_km` | Fare efficiency per kilometre |
| `fare_per_min` | Fare efficiency per minute |
| `rush_hour_flag` | 1 if booking is 7–9am or 5–8pm |
| `long_distance_flag` | 1 if trip exceeds 15km |
| `peak_time_flag` | Merged from time_features table |
| `loyalty_score` | Completed rides / total bookings |
| `reliability_score` | Driver rating × (1 − delay rate) |
| `high_risk_flag` | 1 if customer cancel rate > 30% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data cleaning & feature engineering |
| Scikit-learn | ML model training & evaluation |
| SQLite | Normalized database storage |
| Streamlit | Interactive dashboard |

---

## 📈 Business Impact

- ✅ Predict ride outcomes before trip starts → reduce cancellations
- ✅ Estimate fares within ±3.4% error → better pricing transparency  
- ✅ Flag high-risk customers proactively → offer retention incentives
- ✅ Score driver reliability → smarter driver assignment

---

## 👤 Author

**DHANASEKARAN**  
Built as part of the GUVI × HCL Capstone Project  
Domain: Mobility & Transportation Analytics

---

## 📄 License

This project is for educational purposes.