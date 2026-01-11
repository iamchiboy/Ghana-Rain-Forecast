import pandas as pd

df = pd.read_csv("data/raw/weather_accra.csv")

df["rain_next_30min"] = df["rain_1h"].shift(-3) > 0
df["rain_next_30min"] = df["rain_next_30min"].astype(int)

df["humidity_change"] = df["humidity"].diff()
df["pressure_drop"] = df["pressure"].shift(1) - df["pressure"]

df = df.dropna()

df.to_csv("data/processed/weather_features.csv", index=False)
print("Preprocessing complete")
