"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FORECAST_HORIZON,
    ROLLING_WINDOW, LAG_FEATURES, FEATURE_COLUMNS, TARGET_FEATURE
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_raw_data():
    """
    Load raw weather data from CSV file.
    
    Returns:
        pd.DataFrame: Raw weather data
    """
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
        return df
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        return None


def create_target_variable(df):
    """
    Create target variable: rain_next_30min
    Labels data point as 1 if rain occurred within next 30 minutes.
    
    Args:
        df (pd.DataFrame): Raw weather data with timestamp
    
    Returns:
        pd.DataFrame: Data with target variable
    """
    df = df.copy()
    df['rain_next_30min'] = 0
    
    # Check if any rain occurred in the next 30 minutes
    for i in range(len(df) - 1):
        current_time = df.iloc[i]['timestamp']
        future_window = df[
            (df['timestamp'] > current_time) & 
            (df['timestamp'] <= current_time + timedelta(minutes=FORECAST_HORIZON))
        ]
        
        if len(future_window) > 0 and future_window['rain_1h'].sum() > 0:
            df.at[i, 'rain_next_30min'] = 1
    
    logger.info(f"Created target variable. Rain samples: {df['rain_next_30min'].sum()}, "
               f"No rain samples: {(df['rain_next_30min'] == 0).sum()}")
    
    return df


def create_rolling_features(df):
    """
    Create rolling window features (mean, std, max, min).
    
    Args:
        df (pd.DataFrame): Weather data
    
    Returns:
        pd.DataFrame: Data with rolling features
    """
    df = df.copy()
    
    rolling_cols = ['temp', 'humidity', 'pressure', 'wind_speed']
    
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=ROLLING_WINDOW, min_periods=1).std()
    
    logger.debug(f"Created rolling features with window={ROLLING_WINDOW}")
    return df


def create_lag_features(df):
    """
    Create lag features from previous timesteps.
    
    Args:
        df (pd.DataFrame): Weather data
    
    Returns:
        pd.DataFrame: Data with lag features
    """
    df = df.copy()
    
    lag_cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'rain_1h']
    
    for col in lag_cols:
        if col in df.columns:
            for lag in LAG_FEATURES:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    logger.debug(f"Created lag features: {LAG_FEATURES}")
    return df


def create_derived_features(df):
    """
    Create domain-specific derived features.
    
    Args:
        df (pd.DataFrame): Weather data
    
    Returns:
        pd.DataFrame: Data with derived features
    """
    df = df.copy()
    
    # Humidity change
    df['humidity_change'] = df['humidity'].diff().fillna(0)
    df['humidity_change'] = df['humidity_change'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    # Pressure drop (often indicates rain)
    df['pressure_drop'] = df['pressure'].diff().fillna(0)
    df['pressure_drop'] = df['pressure_drop'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    # Temperature change
    df['temp_change'] = df['temp'].diff().fillna(0)
    df['temp_change'] = df['temp_change'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    # Dew point approximation (indicator of moisture)
    df['dew_point'] = 243.04 * (np.log(df['humidity'] / 100) + 
                                (17.625 * df['temp']) / (243.04 + df['temp'])) / (
                                17.625 - np.log(df['humidity'] / 100) - 
                                (17.625 * df['temp']) / (243.04 + df['temp']))
    
    logger.debug("Created derived features: humidity_change, pressure_drop, temp_change, dew_point")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Weather data with potential missing values
    
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    df = df.copy()
    
    # Forward fill then backward fill for time series data
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Drop remaining rows with NaN (typically the first few due to lag features)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing values")
    
    return df


def select_features(df):
    """
    Select relevant features for the model.
    
    Args:
        df (pd.DataFrame): Full processed data
    
    Returns:
        pd.DataFrame: Data with selected features and target
    """
    # Get all available feature columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # Add derived features that might exist
    derived_features = [col for col in df.columns if any(
        derived in col for derived in ['rolling', 'lag', 'change', 'dew_point']
    )]
    
    selected_cols = list(set(available_features + derived_features))
    
    if TARGET_FEATURE in df.columns:
        selected_cols.append(TARGET_FEATURE)
    
    logger.info(f"Selected {len(selected_cols) - 1} features for modeling")
    return df[selected_cols]


def preprocess(save=True):
    """
    Main preprocessing pipeline.
    
    Args:
        save (bool): Whether to save processed data to CSV
    
    Returns:
        pd.DataFrame: Processed and engineered features
    """
    logger.info("Starting data preprocessing...")
    
    # Load raw data
    df = load_raw_data()
    if df is None or len(df) == 0:
        logger.error("No raw data available for preprocessing")
        return None
    
    logger.info(f"Raw data shape: {df.shape}")
    
    # Create target variable
    df = create_target_variable(df)
    
    # Feature engineering
    df = create_rolling_features(df)
    df = create_lag_features(df)
    df = create_derived_features(df)
    
    # Data cleaning
    df = handle_missing_values(df)
    
    # Feature selection
    df = select_features(df)
    
    logger.info(f"Processed data shape: {df.shape}")
    
    # Save processed data
    if save:
        try:
            df.to_csv(PROCESSED_DATA_PATH, index=False)
            logger.info(f"Saved processed data to {PROCESSED_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
    
    return df


if __name__ == "__main__":
    df = preprocess()
    if df is not None:
        print("\n✅ Preprocessing complete!")
        print(f"Shape: {df.shape}")
        print(f"\nTarget distribution:\n{df['rain_next_30min'].value_counts()}")
        print(f"\nFeatures: {list(df.columns)}")
