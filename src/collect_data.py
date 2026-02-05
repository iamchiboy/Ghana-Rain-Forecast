"""
Data collection module for OpenWeatherMap API
Fetches weather data every 10 minutes and stores it locally
"""
import requests
import pandas as pd
import time
import os
from datetime import datetime
from src.config import (
    API_KEY, CITY, COUNTRY_CODE, API_URL, API_TIMEOUT,
    API_RETRY_ATTEMPTS, API_RETRY_DELAY, DATA_COLLECTION_INTERVAL,
    RAW_DATA_PATH, VALID_TEMP_RANGE, VALID_HUMIDITY_RANGE,
    VALID_PRESSURE_RANGE, VALID_WIND_SPEED_RANGE, VALID_VISIBILITY_RANGE
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def validate_weather_data(data):
    """
    Validate weather data is within reasonable ranges.
    
    Args:
        data (dict): Weather data dictionary
    
    Returns:
        tuple: (is_valid, error_message)
    """
    validations = [
        (VALID_TEMP_RANGE[0] <= data.get("temp", 0) <= VALID_TEMP_RANGE[1],
         f"Temperature out of range: {data.get('temp')}°C"),
        (VALID_HUMIDITY_RANGE[0] <= data.get("humidity", 0) <= VALID_HUMIDITY_RANGE[1],
         f"Humidity out of range: {data.get('humidity')}%"),
        (VALID_PRESSURE_RANGE[0] <= data.get("pressure", 0) <= VALID_PRESSURE_RANGE[1],
         f"Pressure out of range: {data.get('pressure')} hPa"),
        (VALID_WIND_SPEED_RANGE[0] <= data.get("wind_speed", 0) <= VALID_WIND_SPEED_RANGE[1],
         f"Wind speed out of range: {data.get('wind_speed')} m/s"),
    ]
    
    for is_valid, message in validations:
        if not is_valid:
            return False, message
    
    return True, None


def fetch_weather_data(retry_count=0):
    """
    Fetch weather data from OpenWeatherMap API with retry logic.
    
    Args:
        retry_count (int): Current retry attempt
    
    Returns:
        dict: Weather data dictionary or None if failed
    """
    try:
        url = f"{API_URL}?q={CITY},{COUNTRY_CODE}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        
        api_data = response.json()
        
        # Extract weather data
        weather_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "temp": api_data["main"]["temp"],
            "humidity": api_data["main"]["humidity"],
            "pressure": api_data["main"]["pressure"],
            "wind_speed": api_data["wind"]["speed"],
            "wind_deg": api_data["wind"].get("deg", 0),
            "cloudiness": api_data["clouds"]["all"],
            "visibility": api_data.get("visibility", 10000),
            "rain_1h": api_data.get("rain", {}).get("1h", 0),
            "weather_main": api_data["weather"][0]["main"],
            "weather_desc": api_data["weather"][0]["description"],
        }
        
        # Validate the data
        is_valid, error_msg = validate_weather_data(weather_data)
        if not is_valid:
            logger.warning(f"Invalid data: {error_msg}")
            return None
        
        logger.debug(f"Successfully fetched weather data for {CITY}")
        return weather_data
    
    except requests.exceptions.Timeout:
        logger.error(f"API request timeout (attempt {retry_count + 1}/{API_RETRY_ATTEMPTS})")
        if retry_count < API_RETRY_ATTEMPTS - 1:
            time.sleep(API_RETRY_DELAY)
            return fetch_weather_data(retry_count + 1)
        return None
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error (attempt {retry_count + 1}/{API_RETRY_ATTEMPTS})")
        if retry_count < API_RETRY_ATTEMPTS - 1:
            time.sleep(API_RETRY_DELAY)
            return fetch_weather_data(retry_count + 1)
        return None
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("Invalid API key")
        else:
            logger.error(f"HTTP error: {e.response.status_code}")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {str(e)}")
        return None


def save_weather_data(weather_data):
    """
    Save weather data to CSV file.
    
    Args:
        weather_data (dict): Weather data dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        
        df = pd.DataFrame([weather_data])
        
        if not os.path.exists(RAW_DATA_PATH):
            df.to_csv(RAW_DATA_PATH, index=False)
            logger.info(f"Created new data file: {RAW_DATA_PATH}")
        else:
            df.to_csv(RAW_DATA_PATH, mode="a", header=False, index=False)
            logger.debug(f"Appended data to {RAW_DATA_PATH}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving weather data: {str(e)}")
        return False


def collect():
    """
    Main collection function: fetch and save weather data.
    """
    logger.info("Starting data collection...")
    
    weather_data = fetch_weather_data()
    if weather_data is None:
        logger.warning("Failed to fetch weather data, skipping this cycle")
        return False
    
    if save_weather_data(weather_data):
        logger.info(f"Data collected successfully - Temp: {weather_data['temp']}°C, "
                   f"Humidity: {weather_data['humidity']}%, Rain: {weather_data['rain_1h']}mm")
        return True
    
    return False


def start_continuous_collection():
    """
    Start the continuous data collection loop.
    """
    logger.info(f"Starting continuous data collection every {DATA_COLLECTION_INTERVAL} seconds")
    from src.config import validate_config
    validate_config()
    
    try:
        while True:
            try:
                collect()
            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
            
            time.sleep(DATA_COLLECTION_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user")


if __name__ == "__main__":
    start_continuous_collection()
