"""
Configuration management for Ghana Rain Forecast application
"""
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", "openweather.env")
load_dotenv(env_path)

# ==================== API Configuration ====================
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Accra"
COUNTRY_CODE = "GH"
API_URL = "https://api.openweathermap.org/data/2.5/weather"
API_TIMEOUT = 10  # seconds
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# ==================== Data Configuration ====================
RAW_DATA_PATH = "data/raw/weather_accra.csv"
PROCESSED_DATA_PATH = "data/processed/weather_features.csv"
MODEL_PATH = "src/rain_model.pkl"
MODEL_METRICS_PATH = "src/model_metrics.json"

# Data collection
DATA_COLLECTION_INTERVAL = 600  # 10 minutes in seconds
MIN_DATA_POINTS_FOR_TRAINING = 100
TARGET_COLUMN = "rain_next_30min"
FORECAST_HORIZON = 30  # minutes

# ==================== Feature Configuration ====================
FEATURE_COLUMNS = [
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_deg",
    "cloudiness",
    "visibility",
    "humidity_change",
    "pressure_drop",
    "temp_change"
]

TARGET_FEATURE = "rain_next_30min"

# Feature engineering
ROLLING_WINDOW = 3  # for rolling averages
LAG_FEATURES = [1, 2]  # previous timesteps

# ==================== Model Configuration ====================
MODEL_TYPE = "RandomForest"
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced"
}

TRAIN_TEST_SPLIT = 0.2
CROSS_VAL_FOLDS = 5
RANDOM_STATE = 42

# ==================== Logging Configuration ====================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== Dashboard Configuration ====================
DASHBOARD_TITLE = "🌧️ Ghana Rain Nowcasting - 30 Minute Forecast"
DASHBOARD_REFRESH_INTERVAL = 60  # seconds
CHART_HEIGHT = 400

# Confidence thresholds for predictions
RAIN_PROBABILITY_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.75

# ==================== Validation Rules ====================
VALID_TEMP_RANGE = (-10, 50)  # Celsius
VALID_HUMIDITY_RANGE = (0, 100)  # Percentage
VALID_PRESSURE_RANGE = (900, 1050)  # hPa
VALID_WIND_SPEED_RANGE = (0, 50)  # m/s
VALID_VISIBILITY_RANGE = (0, 10000)  # meters

def validate_config():
    """Validate that all required configuration values are set."""
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found in openweather.env")
    
    required_paths = [
        os.path.dirname(RAW_DATA_PATH),
        os.path.dirname(PROCESSED_DATA_PATH),
        os.path.dirname(MODEL_PATH),
        LOG_DIR
    ]
    
    for path in required_paths:
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    validate_config()
    print("✅ Configuration validated successfully")
