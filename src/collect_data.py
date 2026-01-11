import requests
import pandas as pd
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Accra"
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

FILE_PATH = "data/raw/weather_accra.csv"

def collect():
    r = requests.get(URL).json()

    row = {
        "timestamp": datetime.utcnow(),
        "temp": r["main"]["temp"],
        "humidity": r["main"]["humidity"],
        "pressure": r["main"]["pressure"],
        "wind_speed": r["wind"]["speed"],
        "wind_deg": r["wind"].get("deg", 0),
        "rain_1h": r.get("rain", {}).get("1h", 0)
    }

    df = pd.DataFrame([row])

    if not os.path.exists(FILE_PATH):
        df.to_csv(FILE_PATH, index=False)
    else:
        df.to_csv(FILE_PATH, mode="a", header=False, index=False)

    print("Data collected:", row)

if __name__ == "__main__":
    while True:
        collect()
        time.sleep(600)  # 10 minutes
