import streamlit as st
import pandas as pd
import joblib

st.title("🌧️ Ghana Rain Nowcasting Dashboard")

model = joblib.load("../src/rain_model.pkl")
df = pd.read_csv("../data/processed/weather_features.csv")

latest = df.iloc[-1]

prediction = model.predict(
    latest[[
        "temp",
        "humidity",
        "pressure",
        "wind_speed",
        "humidity_change",
        "pressure_drop"
    ]].values.reshape(1, -1)
)[0]

st.metric("Prediction (Next 30 min)", "RAIN" if prediction else "NO RAIN")
st.line_chart(df["humidity"])
st.line_chart(df["pressure"])
