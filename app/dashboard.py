"""
Interactive Streamlit dashboard for Ghana rain forecasting
"""
import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path to resolve src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import get_prediction_summary
from src.config import (
    PROCESSED_DATA_PATH, MODEL_METRICS_PATH, DASHBOARD_TITLE,
    FORECAST_HORIZON, DASHBOARD_REFRESH_INTERVAL
)
from src.logger import setup_logger

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Ghana Rain Nowcaster",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .rain-alert {
        background-color: #FFE5E5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF0000;
    }
    .no-rain-safe {
        background-color: #E5F5FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0066CC;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL)
def load_processed_data():
    """Load processed weather data with caching."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Add index as pseudo-timestamp if timestamp column doesn't exist
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(datetime.now()) - pd.to_timedelta(
                range(len(df) - 1, -1, -1), unit='min'
            )
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
        return df.sort_values('timestamp')
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL)
def load_model_metrics():
    """Load model evaluation metrics."""
    try:
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading metrics: {e}")
    return {}


def create_temperature_chart(df):
    """Create interactive temperature chart."""
    if 'timestamp' not in df.columns or 'temp' not in df.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temp'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    fig.update_layout(
        title="Temperature Trend",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=400,
        hovermode='x unified',
        template="plotly_white"
    )
    return fig


def create_humidity_pressure_chart(df):
    """Create dual-axis humidity and pressure chart."""
    if 'timestamp' not in df.columns:
        return None
    
    fig = go.Figure()
    
    if 'humidity' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['humidity'],
            name='Humidity',
            yaxis='y1',
            line=dict(color='#4ECDC4', width=2),
            mode='lines'
        ))
    
    if 'pressure' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['pressure'],
            name='Pressure',
            yaxis='y2',
            line=dict(color='#95E1D3', width=2),
            mode='lines'
        ))
    
    fig.update_layout(
        title="Humidity & Pressure Trends",
        xaxis_title="Time",
        yaxis=dict(title="Humidity (%)", side='left'),
        yaxis2=dict(title="Pressure (hPa)", overlaying='y', side='right'),
        height=400,
        hovermode='x unified',
        template="plotly_white"
    )
    return fig


def create_rain_history_chart(df):
    """Create rain occurrence chart."""
    if 'timestamp' not in df.columns or 'rain_1h' not in df.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['rain_1h'],
        name='Rainfall (mm)',
        marker=dict(color='#5DADE2')
    ))
    
    fig.update_layout(
        title="Rainfall History (Last Hour)",
        xaxis_title="Time",
        yaxis_title="Rainfall (mm)",
        height=300,
        template="plotly_white"
    )
    return fig


def display_prediction_card(prediction):
    """Display main prediction in a card format."""
    if prediction is None:
        st.error("Unable to load prediction. Please check data availability.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction['will_rain']:
            st.markdown("""
                <div class="rain-alert">
                <h2 style="margin: 0; color: #FF0000;">🌧️ RAIN EXPECTED</h2>
                <p style="margin: 5px 0; font-size: 14px; color: #666;">in the next 30 minutes</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="no-rain-safe">
                <h2 style="margin: 0; color: #0066CC;">☀️ NO RAIN</h2>
                <p style="margin: 5px 0; font-size: 14px; color: #666;">in the next 30 minutes</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Rain Probability",
            f"{prediction['rain_probability']:.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Prediction Confidence",
            f"{prediction['confidence']:.1%}",
            f"{prediction['confidence_level']}"
        )


def display_weather_metrics(df):
    """Display current weather conditions."""
    if df is None or len(df) == 0:
        st.warning("No weather data available")
        return
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{latest.get('temp', 'N/A')}°C")
    
    with col2:
        st.metric("💧 Humidity", f"{latest.get('humidity', 'N/A')}%")
    
    with col3:
        st.metric("🌪️ Wind Speed", f"{latest.get('wind_speed', 'N/A')} m/s")
    
    with col4:
        st.metric("📊 Pressure", f"{latest.get('pressure', 'N/A')} hPa")


def display_model_info(metrics):
    """Display model performance information."""
    with st.expander("📊 Model Information", expanded=False):
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            
            with col2:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.1%}")
            
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
            
            with col4:
                st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        else:
            st.info("Model metrics not yet available. Train the model first.")


def main():
    """Main dashboard application."""
    # Header
    st.title(":cloud: Ghana Rain Nowcasting System")
    st.markdown("**30-Minute Rainfall Prediction using Machine Learning**")
    st.markdown("---")
    
    # Load data
    df = load_processed_data()
    metrics = load_model_metrics()
    
    if df is None or len(df) == 0:
        st.error("""
            ⚠️ **No data available yet**
            
            Please:
            1. Run `python -m src.collect_data` to collect weather data
            2. Run `python -m src.preprocess` to process the data
            3. Run `python -m src.train_model` to train the model
        """)
        return
    
    # Get prediction
    prediction = get_prediction_summary()
    
    # Main prediction section
    st.subheader("📍 Current Forecast")
    display_prediction_card(prediction)
    
    st.markdown("---")
    
    # Current conditions
    st.subheader("🌤️ Current Weather Conditions")
    display_weather_metrics(df)
    
    st.markdown("---")
    
    # Charts section
    st.subheader("📈 Weather Trends")
    
    tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity & Pressure", "Rainfall History"])
    
    with tab1:
        temp_chart = create_temperature_chart(df)
        if temp_chart:
            st.plotly_chart(temp_chart, use_container_width=True)
    
    with tab2:
        humidity_chart = create_humidity_pressure_chart(df)
        if humidity_chart:
            st.plotly_chart(humidity_chart, use_container_width=True)
    
    with tab3:
        rain_chart = create_rain_history_chart(df)
        if rain_chart:
            st.plotly_chart(rain_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Model information
    display_model_info(metrics)
    
    # Footer
    st.markdown("""
    ---
    **Ghana Rain Nowcasting System** | Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + """
    
    📧 For issues or feedback, please contact the development team.
    """)


if __name__ == "__main__":
    main()
