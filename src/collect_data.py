import requests
import pandas as pd
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Load from parent directory
env_path = os.path.join(os.path.dirname(__file__), "..", "openweather.env")
load_dotenv(env_path)

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Accra"
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

FILE_PATH = "data/raw/weather_accra.csv"

def collect():
    response = requests.get(URL)
    response.raise_for_status()
    r = response.json()

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
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

    if not os.path.exists(FILE_PATH):
        df.to_csv(FILE_PATH, index=False)
    else:
        df.to_csv(FILE_PATH, mode="a", header=False, index=False)

    print("Data collected:", row)

if __name__ == "__main__":
    while True:
        collect()
        time.sleep(600)  # 10 minutes
