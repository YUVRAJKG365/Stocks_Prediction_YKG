# 📈 DMart Demand Forecast Pro

**DMart Demand Forecast Pro** is an AI-powered inventory and sales demand forecasting system built with **Streamlit**, designed to help e-commerce and retail businesses optimize stock levels, pricing strategies, and replenishment planning. It uses machine learning models (Random Forest) and rich UI/UX for interactive forecasting.

---

## 🚀 Features

- 🔮 **Smart Demand Forecasting** for selected date, brand, and category  
- 📊 **7-Day Forecast Visualization** with upper/lower bounds  
- 📦 **Inventory Recommendations** with reorder logic  
- 💰 **Dynamic Pricing Strategy Tips**  
- 🔍 **Feature Importance Analysis** using model insights  
- ⚙️ **User-friendly sidebar controls** for input selection  
- 🎨 Stylish & responsive UI with custom CSS and Lottie animations

---

## 📸 UI Preview

| Forecast Dashboard | Recommendations |
|--------------------|------------------|
| ![forecast](https://imgur.com/YOUR_IMAGE_1.png) | ![recs](https://imgur.com/YOUR_IMAGE_2.png) |

---

## 🧠 How It Works

The app uses a trained `RandomForestRegressor` model with encoded features such as:

- Category & Brand Encoding  
- Discount percentage  
- Date-based features: `DayOfWeek`, `Quarter`, `IsHoliday`, etc.  
- Lag features: `Lag_1`, `Lag_7`, `RollingMean_7`

After preprocessing and scaling, it predicts daily demand and provides visual forecasts, inventory thresholds, and sales recommendations.

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/YUVRAJKG365/Stocks_Prediction_YKG.git
cd Stocks_Prediction_YKG

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run PP.py
📂 File Structure
bash
Copy
Edit
Stocks_Prediction_YKG/
│
├── PP.py                 # Main Streamlit app
├── scaler.pkl            # Preprocessing scaler
├── best_random_forest_model.pkl  # Trained ML model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
🧪 Model Details
Model: RandomForestRegressor

Input Features: Categorical, Numerical, Lag, Time-series

Scaler: StandardScaler

Performance: Tuned for retail demand estimation using historical sales

📦 Dependencies
Key packages used:

streamlit

scikit-learn

pandas, numpy

joblib

plotly, matplotlib

streamlit-lottie

requests
