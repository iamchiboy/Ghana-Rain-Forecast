import joblib
import pandas as pd

model = joblib.load("src/rain_model.pkl")

latest = pd.read_csv("data/processed/weather_features.csv").iloc[-1]

X = latest[[
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "humidity_change",
    "pressure_drop"
]].values.reshape(1, -1)

prediction = model.predict(X)[0]

print("🌧️ Rain in next 30 minutes?" , "YES" if prediction == 1 else "NO")
