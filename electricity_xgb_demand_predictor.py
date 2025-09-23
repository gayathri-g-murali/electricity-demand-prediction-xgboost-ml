# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from collections import deque
import plotly.graph_objects as go
import requests
from PIL import Image
from io import BytesIO

# ===============================
# 1. Load the trained XGBoost model
# ===============================
with open("electricity_xgb_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# ===============================
# 2. Page configuration
# ===============================
st.set_page_config(
    page_title="⚡ Smart Electricity Demand Predictor ⚡",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# 3. Sidebar Inputs
# ===============================
st.sidebar.header("🌟 Input Features")
selected_datetime = st.sidebar.date_input("Pick Date 📅", datetime.today())
selected_hour = st.sidebar.slider("Pick Hour ⏰", 0, 23, datetime.now().hour)
dt = pd.Timestamp(selected_datetime) + pd.Timedelta(hours=selected_hour)

cities = [
    "Delhi", "Noida", "Gurugram", "Lucknow", "Jaipur", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
    "Thiruvananthapuram", "Kochi", "Kozhikode", "Kollam",
    "Coimbatore", "Madurai",
    "Vijayawada", "Visakhapatnam", "Guntur",
    "Warangal", "Karimnagar"
]
selected_city = st.sidebar.selectbox("Select City 🏙️", cities)
base_demand = st.sidebar.number_input("Base Demand (Units) 🔌", 0.0, 10000.0, 1500.0)

# ===============================
# 4. Fetch live weather with granular data
# ===============================
API_KEY = None
try:
    API_KEY = st.secrets["openweathermap"]["api_key"]
except:
    st.info("ℹ️ No API key provided. Using default temperature/humidity.")

def get_weather(city_name, api_key):
    try:
        if not api_key:
            raise Exception("No key")
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url).json()
        lat, lon = geo_response[0]['lat'], geo_response[0]['lon']

        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()

        temp = weather_response['main']['temp']
        hum = weather_response['main']['humidity']
        wind_speed = weather_response['wind']['speed']
        cloud_cover = weather_response['clouds']['all']
        icon_code = weather_response['weather'][0]['icon']

        return temp, hum, wind_speed, cloud_cover, icon_code
    except:
        st.warning("Could not fetch live weather. Using default values.")
        return 30, 50, 3, 20, "01d"

def get_weather_icon(icon_code):
    try:
        url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        fallback_url = "https://upload.wikimedia.org/wikipedia/commons/c/c3/Weather-icon.png"
        response = requests.get(fallback_url)
        img = Image.open(BytesIO(response.content))
        return img

Temperature, Humidity, Wind, Clouds, icon_code = get_weather(selected_city, API_KEY)
icon_img = get_weather_icon(icon_code)

# ===============================
# 5. Header with weather icon
# ===============================
st.markdown(
    "<h1 style='text-align: center; color: #FF8C00;'>⚡🔋 Smart Electricity Demand Predictor 🔋⚡</h1>",
    unsafe_allow_html=True
)
st.image(icon_img, width=100, caption=f"Weather Icon - {selected_city}")
st.markdown(
    f"<p style='text-align: center;'>Forecast electricity demand using simulated or real-time weather 🌡️💧💨☁️</p>",
    unsafe_allow_html=True
)

# ===============================
# 6. Initialize rolling features
# ===============================
last_24hr = deque([base_demand]*24, maxlen=24)
last_168hr = deque([base_demand]*168, maxlen=168)
future_hours = 48
future_datetimes = [dt + timedelta(hours=i+1) for i in range(future_hours)]
future_predictions = []
upper_bounds = []
lower_bounds = []

# Dynamic variations
temp_variation = 3 * np.sin(np.linspace(0, 2*np.pi, future_hours))
humidity_variation = np.random.uniform(-5, 5, future_hours)

# ===============================
# 7. Forecast loop
# ===============================
for i, future_dt in enumerate(future_datetimes):
    hour = future_dt.hour
    dayofweek = future_dt.dayofweek
    month = future_dt.month
    year = future_dt.year
    dayofyear = future_dt.dayofyear
    weekofyear = future_dt.isocalendar()[1]
    quarter = future_dt.quarter
    is_weekend = 1 if dayofweek >= 5 else 0

    temp = Temperature + temp_variation[i]
    hum = np.clip(Humidity + humidity_variation[i], 0, 100)

    rolling_mean_24hrs = np.mean(last_24hr)
    rolling_std_24hrs = np.std(last_24hr)
    lag_24 = last_24hr[-1]
    lag_168 = last_168hr[-1]

    input_future = pd.DataFrame({
        "hour": [hour],
        "dayofweek": [dayofweek],
        "month": [month],
        "year": [year],
        "dayofyear": [dayofyear],
        "weekofyear": [weekofyear],
        "quarter": [quarter],
        "is_weekend": [is_weekend],
        "Temperature": [temp],
        "Humidity": [hum],
        "Demand_lag_24hr": [lag_24],
        "Demand_lag_168hrs": [lag_168],
        "Demand_rolling_mean_24hrs": [rolling_mean_24hrs],
        "Demand_rolling_std_24hrs": [rolling_std_24hrs]
    })

    pred = model.predict(input_future)[0]
    future_predictions.append(pred)
    upper_bounds.append(pred * 1.10)
    lower_bounds.append(pred * 0.90)

    last_24hr.append(pred)
    last_168hr.append(pred)

# ===============================
# 8. Plot forecast with shaded confidence bands and enterprise look
# ===============================
# Historical data placeholder (for demo, use base_demand history)
hist_datetimes = [dt - timedelta(hours=i) for i in reversed(range(24))]
hist_values = [base_demand + np.random.uniform(-100, 100) for _ in range(24)]

fig = go.Figure()

# Shaded confidence
fig.add_trace(go.Scatter(
    x=future_datetimes + future_datetimes[::-1],
    y=upper_bounds + lower_bounds[::-1],
    fill='toself',
    fillcolor='rgba(173,216,230,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Confidence Range'
))

# Predicted line
fig.add_trace(go.Scatter(
    x=future_datetimes,
    y=future_predictions,
    mode='lines+markers',
    name='Predicted Demand',
    line=dict(color='blue'),
    marker=dict(size=6),
    hovertemplate='Time: %{x}<br>Predicted: %{y:.2f} Units<extra></extra>'
))

# Historical line
fig.add_trace(go.Scatter(
    x=hist_datetimes,
    y=hist_values,
    mode='lines+markers',
    name='Historical Demand',
    line=dict(color='orange'),
    marker=dict(size=6),
    hovertemplate='Time: %{x}<br>Historical: %{y:.2f} Units<extra></extra>'
))

fig.update_layout(
    title=f"📊 Electricity Demand Forecast - {selected_city} (Next {future_hours} Hours)",
    xaxis_title="Datetime",
    yaxis_title="Electricity Demand (Units)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# 9. Latest forecast with professional layout
# ===============================
latest_pred = future_predictions[-1]

st.markdown(
    """
    <div style="
        text-align: center;
        background: linear-gradient(90deg, #141e30, #243b55);
        padding: 16px;
        border-radius: 12px;
        color: white;
        font-size: 22px;
        font-weight: bold;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.35);
    ">
        🔮 Energy & Weather Outlook 🌡️💧
    </div>
    """,
    unsafe_allow_html=True
)


# CSS for card styling with different colors + fonts + icons
st.markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
    .card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px;
        font-family: 'Trebuchet MS', sans-serif;
        box-shadow: 3px 3px 12px rgba(0,0,0,0.2);
    }
    .card h4 {
        margin-bottom: 8px;
        font-size: 18px;
        font-weight: bold;
    }
    .card p {
        font-size: 22px;
        font-weight: bold;
        margin: 0;
    }
    .pred {background: linear-gradient(135deg, #3498db, #2980b9);}
    .temp {background: linear-gradient(135deg, #e67e22, #d35400);}
    .hum {background: linear-gradient(135deg, #27ae60, #1e8449);}
    .wind {background: linear-gradient(135deg, #8e44ad, #6c3483);}
    .icon {
        font-size: 26px;
        margin-bottom: 6px;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class='card pred'>
            <i class="fas fa-bolt icon"></i>
            <h4>Predicted Demand</h4>
            <p>{latest_pred:.2f} Units</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='card temp'>
            <i class="fas fa-temperature-high icon"></i>
            <h4>Temperature</h4>
            <p>{Temperature:.1f} °C</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class='card hum'>
            <i class="fas fa-tint icon"></i>
            <h4>Humidity</h4>
            <p>{Humidity:.1f} %</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class='card wind'>
            <i class="fas fa-wind icon"></i>
            <h4>Wind & Clouds</h4>
            <p>{Wind:.1f} m/s | {Clouds}%</p>
        </div>
    """, unsafe_allow_html=True)
# ===============================
# 10. Download forecast CSV
# ===============================
forecast_df = pd.DataFrame({
    "Datetime": [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in future_datetimes],
    "Predicted_Demand": future_predictions,
    "Upper_Bound": upper_bounds,
    "Lower_Bound": lower_bounds
})
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download Forecast CSV",
    data=csv,
    file_name=f"{selected_city}_electricity_forecast.csv",
    mime='text/csv'
)

# ===============================
# 11. Footer with professional look
# ===============================
st.markdown("&copy; 2025 Gayathri G Murali | Made with ❤️ using Streamlit", unsafe_allow_html=True)
