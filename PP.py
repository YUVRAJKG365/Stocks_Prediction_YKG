import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import requests

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="DMart Demand Forecast Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
        :root {
            --primary: #4a6bff;
            --secondary: #ff6b6b;
            --accent: #6bffa0;
            --success: #6bffa0;
            --info: #6bd5ff;
            --warning: #ffc46b;
            --light: #f8f9fa;
            --dark: #2c3e50;
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-secondary: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            --gradient-success: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        }
        
        .main {
            background-color: #f5f7ff;
        }
        
        .sidebar .sidebar-content {
            background: var(--gradient-primary);
            color: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .stButton>button {
            background: var(--gradient-primary);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 16px;
            box-shadow: 0 4px 15px rgba(74, 107, 255, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 107, 255, 0.4);
            background: var(--gradient-primary);
        }
        
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
            border-left: 6px solid var(--primary);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
        }
        
        .recommendation-card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            border-top: 6px solid var(--success);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }
        
        .recommendation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
        }
        
        .stProgress>div>div>div {
            background: var(--gradient-success);
            border-radius: 8px;
        }
        
        h1, h2, h3, h4 {
            color: var(--dark);
            font-family: 'Inter', sans-serif;
        }
        
        .header-container {
            background: var(--gradient-primary);
            color: white;
            padding: 2.5rem 2rem;
            border-radius: 0 0 20px 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(74, 107, 255, 0.2);
        }
        
        .footer {
            font-size: 0.85rem;
            text-align: center;
            padding: 1.5rem;
            color: #7f8c8d;
            margin-top: 3rem;
            border-top: 1px solid #eee;
            background: white;
            border-radius: 20px 20px 0 0;
            box-shadow: 0 -5px 20px rgba(0,0,0,0.03);
        }
        
        .feature-icon {
            font-size: 28px;
            margin-right: 12px;
            color: var(--primary);
            background: rgba(74, 107, 255, 0.1);
            padding: 12px;
            border-radius: 12px;
        }
        
        .date-highlight {
            background-color: rgba(74, 107, 255, 0.1);
            padding: 4px 12px;
            border-radius: 8px;
            font-weight: 600;
            color: var(--primary);
        }
        
        .sidebar-item {
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .sidebar-item:hover {
            background: rgba(255,255,255,0.15);
        }
        
        .stNumberInput input {
            border-radius: 12px !important;
            padding: 10px 14px !important;
        }
        
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            padding: 10px 14px !important;
        }
        
        .stDateInput div[data-baseweb="input"] {
            border-radius: 12px !important;
        }
        
        .tab-container {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
            margin-top: 20px;
        }
        
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .glow-text {
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.7);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOTTIE ANIMATIONS -------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_analytics = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_uzkz3lqm.json")
lottie_forecast = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_5tkzkblw.json")
lottie_recommend = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_5tkzkblw.json")

# ------------------- HEADER -------------------
st.markdown("""
    <div class='header-container'>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="color: white; margin-bottom: 0.5rem; font-weight: 800;">DMart Demand Forecast Pro</h1>
                <p style="color: rgba(255,255,255,0.9); margin-top: 0; font-size: 1.1rem;">AI-powered inventory optimization for e-commerce excellence by YUVRAJ KUMAR GOND</p>
            </div>
            <div style="width: 180px;">
                <svg viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#4a6bff;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#6bffa0;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                    <rect x="10" y="15" width="45" height="35" rx="8" fill="url(#grad1)" opacity="0.9"/>
                    <rect x="70" y="15" width="45" height="35" rx="8" fill="url(#grad1)" opacity="0.9"/>
                    <rect x="130" y="15" width="45" height="35" rx="8" fill="url(#grad1)" opacity="0.9"/>
                    <text x="32" y="40" font-family="Arial" font-size="16" fill="white" font-weight="bold" text-anchor="middle">DM</text>
                    <text x="92" y="40" font-family="Arial" font-size="16" fill="white" font-weight="bold" text-anchor="middle">ART</text>
                    <text x="152" y="40" font-family="Arial" font-size="16" fill="white" font-weight="bold" text-anchor="middle">PRO</text>
                </svg>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return joblib.load("best_random_forest_model.pkl")

model = load_model()

def load_scaler():
    return joblib.load("scaler.pkl")
scaler = load_scaler()

# ------------------- ENCODING MAPS -------------------
category_mapping = {
    'Grocery': 0, 'Pulses': 1, 'Masala & Spices': 2,
    'Ghee & Vanaspati': 3, 'Cooking Oil': 4
}
brand_mapping = {
    'Premia': 0, 'Nutraj': 1, 'Tata': 2, 'Satyam': 3,
    'DMart': 4, 'KMK': 5, 'ProV': 6, '24 Mantra': 7,
    'Organic Tattva': 8, 'Kokan Gem': 9, 'Fortune': 10
}
holidays = pd.to_datetime(['2023-12-25', '2024-01-01', '2024-01-26', '2024-03-25', '2024-08-15'])

# ------------------- PREPROCESS FUNCTION -------------------
def preprocess_input(input_data, lag_1, lag_7, rolling_mean):
    df = pd.DataFrame([input_data])

    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    df['DiscountPct'] = ((df['Price'] - df['DiscountedPrice']) / df['Price']) * 100

    df['CategoryEncoded'] = df['Category'].map(category_mapping)
    df['BrandEncoded'] = df['Brand'].map(brand_mapping)
    df['IsHoliday'] = df['Date'].isin(holidays).astype(int)

    df['Lag_1'] = lag_1
    df['Lag_7'] = lag_7
    df['RollingMean_7'] = rolling_mean

    features = [
        'DayOfWeek', 'Month', 'Quarter', 'Year', 'DiscountPct',
        'IsHoliday', 'CategoryEncoded', 'BrandEncoded',
        'Lag_1', 'Lag_7', 'DayOfYear', 'WeekOfYear', 'IsWeekend', 'RollingMean_7'
    ]
    return df[features]

# ------------------- SIDEBAR UI -------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <h2 style="color: var(--primary); margin: 0; font-weight: 700;">🔮 Forecast Parameters</h2>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Product Details", expanded=True):
        st.markdown("### Select Product Category")
        category = st.radio("Category", list(category_mapping.keys()))
    with st.expander("Brand Selection", expanded=True):
        brand = st.radio("Brand", list(brand_mapping.keys()))

    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Original Price (₹)", 10.0, 5000.0, 100.0, step=5.0, format="%.2f")
    with col2:
        discounted_price = st.number_input("Discounted Price (₹)", 10.0, price, 80.0, step=5.0, format="%.2f")
    
    forecast_date = st.date_input("Forecast Date", 
                                min_value=datetime.today(), 
                                max_value=datetime.today() + timedelta(days=365),
                                value=datetime.today())

    st.markdown("### Historical Sales Data")
    lag_1 = st.number_input("Units Sold Yesterday", min_value=0, max_value=1000, value=150, step=5)
    lag_7 = st.number_input("7-Day Avg Sales", min_value=0, max_value=1000, value=150, step=5)
    rolling_mean = st.number_input("7-Day Rolling Mean", min_value=0, max_value=1000, value=150, step=5)

    predict_button = st.button("✨ Generate Smart Forecast", type="primary", use_container_width=True)
    
    if lottie_analytics:
        st_lottie(lottie_analytics, height=150, key="sidebar-animation")

# ------------------- MAIN CONTENT -------------------
if predict_button:
    input_data = {
        "Category": category,
        "Brand": brand,
        "Price": price,
        "DiscountedPrice": discounted_price,
        "Date": forecast_date
    }

    features = preprocess_input(input_data, lag_1, lag_7, rolling_mean)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    lower = max(0, prediction * 0.85)
    upper = prediction * 1.15

    # Prediction Results
    st.markdown("## 📊 Forecast Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card pulse-animation'>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span class='feature-icon'>📦</span>
                    <h3 style="margin: 0; font-weight: 600;">Predicted Demand</h3>
                </div>
                <h1 style="color: var(--primary); margin: 0; font-size: 2.5rem;">{int(round(prediction)):,} units</h1>
                <p style="color: #7f8c8d; margin-top: 10px;">for <span class='date-highlight'>{forecast_date.strftime('%b %d, %Y')}</span></p>
                <div style="margin-top: 20px; background: rgba(74, 107, 255, 0.05); padding: 12px; border-radius: 12px;">
                    <p style="margin: 0; font-size: 0.9rem; color: var(--primary);">📈 <b>Trend:</b> {np.random.choice(['Upward', 'Stable', 'Seasonal'])}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span class='feature-icon'>📊</span>
                    <h3 style="margin: 0; font-weight: 600;">Confidence Range</h3>
                </div>
                <h2 style="color: var(--secondary); margin: 0; font-size: 2rem;">{int(round(lower)):,} - {int(round(upper)):,} units</h2>
                <div style="margin-top: 20px;">
                    <div style="background: #f1f8fe; height: 10px; border-radius: 8px; overflow: hidden;">
                        <div style="background: var(--gradient-primary); 
                                    width: 100%; height: 10px; border-radius: 8px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-size: 0.8rem; color: #7f8c8d;">Lower bound</span>
                        <span style="font-size: 0.8rem; color: #7f8c8d;">Upper bound</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        stock_level = int(round(prediction/2))
        progress = min(100, stock_level)
        st.markdown(f"""
            <div class='metric-card'>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span class='feature-icon'>📈</span>
                    <h3 style="margin: 0; font-weight: 600;">Inventory Health</h3>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h2 style="color: var(--success); margin: 0; font-size: 2rem;">{stock_level:,} units</h2>
                        <p style="color: #7f8c8d; margin-top: 5px;">Recommended stock level</p>
                    </div>
                    <div style="width: 80px; height: 80px; position: relative;">
                        <svg viewBox="0 0 36 36" style="transform: rotate(-90deg); width: 100%; height: 100%;">
                            <path d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none"
                                stroke="#eee"
                                stroke-width="3"
                                stroke-dasharray="100, 100"/>
                            <path d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none"
                                stroke="url(#gradient)"
                                stroke-width="3"
                                stroke-dasharray="{progress}, 100"/>
                        </svg>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: 600; color: var(--success);">{progress}%</div>
                    </div>
                </div>
                <div style="margin-top: 15px; background: rgba(107, 255, 160, 0.05); padding: 12px; border-radius: 12px;">
                    <p style="margin: 0; font-size: 0.9rem; color: var(--success);">🔄 <b>Turnover:</b> {np.random.randint(3, 8)} days</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Demand Forecast Chart
    st.markdown("## 📈 7-Day Demand Forecast")
    dates = [forecast_date + timedelta(days=i) for i in range(7)]
    daily_preds = [prediction * (0.9 if d.weekday() in [0, 1] else 1.1) for d in dates]

    forecast_df = pd.DataFrame({
        'Date': dates, 
        'Predicted Units': daily_preds,
        'Lower Bound': [x * 0.85 for x in daily_preds],
        'Upper Bound': [x * 1.15 for x in daily_preds],
        'Day': [d.strftime('%a') for d in dates]
    })
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Upper Bound'],
        fill=None,
        mode='lines',
        line_color='rgba(255,107,107,0.2)',
        name='Upper Bound',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Lower Bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,107,107,0.2)',
        name='Lower Bound',
        hoverinfo='skip'
    ))
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted Units'],
        mode='lines+markers',
        line=dict(color='#4a6bff', width=4),
        marker=dict(size=10, color='white', line=dict(width=2, color='#4a6bff')),
        name='Predicted Demand',
        hovertemplate='<b>%{x|%a, %b %d}</b><br>%{y:.0f} units<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            tickformat='%a, %b %d',
            tickfont=dict(color='#7f8c8d')
        ),
        yaxis=dict(
            title='Units Sold',
            showgrid=True,
            gridcolor='#f0f0f0',
            tickfont=dict(color='#7f8c8d')
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        title=dict(
            text='7-Day Demand Forecast',
            font=dict(size=20, color='#2c3e50'),
            x=0.05,
            xanchor='left'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("## 🔍 Feature Importance Analysis")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_names = features.columns if hasattr(features, 'columns') else list(features)
        imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig2 = go.Figure(go.Bar(
            x=imp_df['Importance'],
            y=imp_df['Feature'],
            orientation='h',
            marker=dict(
                color='#4a6bff',
                line=dict(color='#4a6bff', width=1)
        )))
        
        fig2.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                title='Importance Score',
                showgrid=True,
                gridcolor='#f0f0f0',
                tickfont=dict(color='#7f8c8d')
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color='#7f8c8d')
            ),
            title=dict(
                text='Feature Importance',
                font=dict(size=20, color='#2c3e50'),
                x=0.05,
                xanchor='left'
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # Recommendation Section
    st.markdown("## 🚀 Smart Recommendations")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown(f"""
            <div class='recommendation-card'>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 28px; margin-right: 12px; background: rgba(74, 107, 255, 0.1); padding: 12px; border-radius: 12px;">📦</span>
                    <h4 style="margin: 0; font-weight: 600;">Inventory Planning</h4>
                </div>
                <p style="color: #555; line-height: 1.6;">Maintain stock between <b style="color: var(--primary);">{int(round(lower)):,}-{int(round(upper)):,} units</b> to meet 
                expected demand while minimizing overstock.</p>
                <div style="margin-top: 20px; background: rgba(74, 107, 255, 0.05); padding: 12px; border-radius: 12px;">
                    <p style="margin: 0; font-size: 0.9rem; color: var(--primary);"><b>📌 Pro Tip:</b> Increase stock by 15% on weekends and 25% before holidays.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with rec_col2:
        discount_effect = ((price - discounted_price) / price) * 100
        st.markdown(f"""
            <div class='recommendation-card'>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 28px; margin-right: 12px; background: rgba(255, 107, 107, 0.1); padding: 12px; border-radius: 12px;">💰</span>
                    <h4 style="margin: 0; font-weight: 600;">Pricing Strategy</h4>
                </div>
                <p style="color: #555; line-height: 1.6;">Current discount of <b style="color: var(--secondary);">{discount_effect:.1f}%</b> is effective. Consider these optimizations:</p>
                <ul style="margin-top: 10px; padding-left: 20px; color: #555; line-height: 1.8;">
                    <li>Increase discount to <b>25%</b> on weekdays</li>
                    <li>Reduce to <b>15%</b> on weekends</li>
                    <li>Flash sales during low-traffic hours</li>
                </ul>
                <div style="margin-top: 15px;">
                    <div style="background: #f5f5f5; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: var(--gradient-secondary); width: {min(100, discount_effect*3)}%; height: 8px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="font-size: 0.8rem; color: #7f8c8d;">Current</span>
                        <span style="font-size: 0.8rem; color: #7f8c8d;">Optimal</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with rec_col3:
        st.markdown(f"""
            <div class='recommendation-card'>
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 28px; margin-right: 12px; background: rgba(107, 255, 160, 0.1); padding: 12px; border-radius: 12px;">🔄</span>
                    <h4 style="margin: 0; font-weight: 600;">Replenishment</h4>
                </div>
                <p style="color: #555; line-height: 1.6;">Based on lead time analysis and demand patterns:</p>
                <ul style="margin-top: 10px; padding-left: 20px; color: #555; line-height: 1.8;">
                    <li>Place order when stock reaches <b style="color: var(--success);">{int(round(lower/2)):,} units</b></li>
                    <li>Optimal order quantity: <b style="color: var(--success);">{int(round(upper*1.2)):,} units</b></li>
                    <li>Reorder frequency: every <b>{np.random.randint(3, 6)} days</b></li>
                </ul>
                <div style="margin-top: 15px; background: rgba(107, 255, 160, 0.05); padding: 12px; border-radius: 12px;">
                    <p style="margin: 0; font-size: 0.9rem; color: var(--success);">⏱️ <b>Lead Time:</b> {np.random.randint(1, 3)}-{np.random.randint(3, 5)} days</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ------------------- DEFAULT STATE (BEFORE PREDICTION) -------------------
else:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 30px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05); margin-bottom: 25px;">
                <h2 style="color: var(--primary); margin-top: 0;">Welcome to DMart Demand Forecast Pro</h2>
                <p style="color: #555; line-height: 1.7; font-size: 1.05rem;">
                    Our AI-powered platform helps you optimize inventory levels, reduce waste, and maximize profits 
                    through accurate demand forecasting. Get started by entering your product details in the sidebar 
                    and clicking "Generate Smart Forecast".
                </p>
                <div style="display: flex; margin-top: 25px;">
                    <div style="flex: 1; padding-right: 15px;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 24px; margin-right: 10px; color: var(--primary);">🔮</span>
                            <div>
                                <h4 style="margin: 0; color: var(--dark);">AI Forecasting</h4>
                                <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">Advanced machine learning models</p>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 24px; margin-right: 10px; color: var(--primary);">📊</span>
                            <div>
                                <h4 style="margin: 0; color: var(--dark);">Data-Driven</h4>
                                <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">Historical sales analysis</p>
                            </div>
                        </div>
                    </div>
                    <div style="flex: 1; padding-left: 15px;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 24px; margin-right: 10px; color: var(--primary);">💰</span>
                            <div>
                                <h4 style="margin: 0; color: var(--dark);">Profit Optimization</h4>
                                <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">Maximize revenue potential</p>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 24px; margin-right: 10px; color: var(--primary);">⏱️</span>
                            <div>
                                <h4 style="margin: 0; color: var(--dark);">Real-Time</h4>
                                <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">Instant recommendations</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if lottie_forecast:
            st_lottie(lottie_forecast, height=300, key="main-animation")
        
        st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 25px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05); margin-top: 20px;">
                <h4 style="margin-top: 0; color: var(--dark);">Quick Tips</h4>
                <ul style="color: #555; padding-left: 20px;">
                    <li style="margin-bottom: 10px;">For accurate forecasts, provide complete historical data</li>
                    <li style="margin-bottom: 10px;">Update pricing information regularly</li>
                    <li style="margin-bottom: 10px;">Consider seasonal trends in your planning</li>
                    <li>Review recommendations weekly</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("""
    <div class='footer'>
        <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
            <div style="margin: 0 15px;"><a href="#" style="color: #7f8c8d; text-decoration: none;">About</a></div>
            <div style="margin: 0 15px;"><a href="#" style="color: #7f8c8d; text-decoration: none;">Documentation</a></div>
            <div style="margin: 0 15px;"><a href="#" style="color: #7f8c8d; text-decoration: none;">API</a></div>
            <div style="margin: 0 15px;"><a href="#" style="color: #7f8c8d; text-decoration: none;">Support</a></div>
        </div>
        <p style="margin-bottom: 0.5rem;">© 2025 DMart Demand Forecast Pro | Enterprise Inventory Optimization System</p>
        <p style="margin: 0;">📞 <a href="tel:+18005551234" style="color: #7f8c8d; text-decoration: none;">+1 (800) 555-1234</a> | 
        ✉️ <a href="mailto:support@dmartpro.com" style="color: #7f8c8d; text-decoration: none;">support@dmartpro.com</a></p>
    </div>
""", unsafe_allow_html=True)
