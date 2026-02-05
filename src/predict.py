"""
Prediction module with confidence scores and interval estimation
"""
import joblib
import pandas as pd
import json
import os
from datetime import datetime

from src.config import (
    PROCESSED_DATA_PATH, MODEL_PATH, MODEL_METRICS_PATH, 
    RAIN_PROBABILITY_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    FEATURE_COLUMNS, TARGET_FEATURE
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_model():
    """
    Load trained model from disk.
    
    Returns:
        RandomForestClassifier: Trained model
    """
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def load_metrics():
    """
    Load model metrics from disk.
    
    Returns:
        dict: Model evaluation metrics
    """
    try:
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            logger.debug(f"Loaded metrics from {MODEL_METRICS_PATH}")
            return metrics
    except Exception as e:
        logger.warning(f"Error loading metrics: {str(e)}")
    
    return {}


def get_latest_weather_data():
    """
    Get the latest weather data from processed dataset.
    
    Returns:
        pd.Series: Latest weather features
    """
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        if len(df) == 0:
            logger.error("No data available in processed dataset")
            return None
        
        latest = df.iloc[-1]
        logger.debug(f"Retrieved latest data from {PROCESSED_DATA_PATH}")
        return latest
    
    except Exception as e:
        logger.error(f"Error loading latest weather data: {str(e)}")
        return None


def prepare_features(latest_data):
    """
    Prepare feature matrix for prediction from latest data point.
    
    Args:
        latest_data (pd.Series): Latest weather data
    
    Returns:
        tuple: (feature_matrix, feature_names) or (None, None) if failed
    """
    try:
        # Get all available feature columns from the data
        available_features = [col for col in latest_data.index 
                             if col != TARGET_FEATURE]
        
        # Create feature matrix
        X = latest_data[available_features].values.reshape(1, -1)
        
        logger.debug(f"Prepared {len(available_features)} features for prediction")
        return X, available_features
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None, None


def make_prediction(model, X):
    """
    Make rain prediction with probability.
    
    Args:
        model: Trained model
        X: Feature matrix
    
    Returns:
        dict: Prediction results with confidence
    """
    try:
        # Get prediction and probability
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        rain_probability = probabilities[1]  # Probability of rain
        confidence = max(probabilities)  # Confidence in the prediction
        
        # Determine prediction confidence level
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "HIGH"
        elif confidence >= RAIN_PROBABILITY_THRESHOLD:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "will_rain": bool(prediction),
            "rain_probability": float(rain_probability),
            "no_rain_probability": float(probabilities[0]),
            "confidence": float(confidence),
            "confidence_level": confidence_level,
            "prediction": "RAIN ☔" if prediction else "NO RAIN ☀️"
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None


def predict():
    """
    Main prediction pipeline: load model, get data, and make prediction.
    
    Returns:
        dict: Prediction results or None if failed
    """
    logger.info("Starting prediction pipeline...")
    
    # Load model
    model = load_model()
    if model is None:
        logger.error("Cannot proceed: Model not available")
        return None
    
    # Get latest data
    latest_data = get_latest_weather_data()
    if latest_data is None:
        logger.error("Cannot proceed: No weather data available")
        return None
    
    # Prepare features
    X, features = prepare_features(latest_data)
    if X is None:
        logger.error("Cannot proceed: Feature preparation failed")
        return None
    
    # Make prediction
    result = make_prediction(model, X)
    
    if result:
        logger.info(f"✅ Prediction: {result['prediction']}")
        logger.info(f"   Probability: {result['rain_probability']:.2%}")
        logger.info(f"   Confidence: {result['confidence']:.2%} ({result['confidence_level']})")
    
    return result


def get_prediction_summary():
    """
    Get prediction with model information for display.
    
    Returns:
        dict: Complete prediction summary
    """
    prediction = predict()
    metrics = load_metrics()
    
    if prediction is None:
        return None
    
    prediction['model_metrics'] = {
        'accuracy': metrics.get('accuracy'),
        'f1_score': metrics.get('f1_score'),
        'recall': metrics.get('recall'),
        'precision': metrics.get('precision')
    }
    
    return prediction


if __name__ == "__main__":
    result = predict()
    if result:
        print("\n" + "="*50)
        print("🌧️ RAIN FORECAST (Next 30 Minutes)")
        print("="*50)
        print(f"Prediction: {result['prediction']}")
        print(f"Rain Probability: {result['rain_probability']:.1%}")
        print(f"Confidence: {result['confidence']:.1%} ({result['confidence_level']})")
        print(f"Timestamp: {result['timestamp']}")
        print("="*50)
