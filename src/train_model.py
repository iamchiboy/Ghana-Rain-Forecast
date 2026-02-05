"""
Model training module with validation and hyperparameter tuning
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PROCESSED_DATA_PATH, MODEL_PATH, MODEL_METRICS_PATH, 
    TRAIN_TEST_SPLIT, CROSS_VAL_FOLDS, RANDOM_STATE, MODEL_PARAMS,
    TARGET_FEATURE, MIN_DATA_POINTS_FOR_TRAINING
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_processed_data():
    """
    Load processed weather features.
    
    Returns:
        tuple: (X, y) feature matrix and target variable
    """
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info(f"Loaded processed data: {df.shape}")
        
        if len(df) < MIN_DATA_POINTS_FOR_TRAINING:
            logger.warning(f"Insufficient data: {len(df)} rows < {MIN_DATA_POINTS_FOR_TRAINING} required")
            return None, None
        
        if TARGET_FEATURE not in df.columns:
            logger.error(f"Target variable '{TARGET_FEATURE}' not found in processed data")
            return None, None
        
        # Separate features and target
        y = df[TARGET_FEATURE]
        X = df.drop(columns=[TARGET_FEATURE])
        
        logger.info(f"Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
        return X, y
    
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        return None, None


def train_model(X_train, y_train):
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        RandomForestClassifier: Trained model
    """
    logger.info("Training Random Forest model...")
    logger.info(f"Model parameters: {MODEL_PARAMS}")
    
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    logger.info("✅ Model training complete")
    return model


def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        X_train: Training features (for cross-validation)
        y_train: Training target (for cross-validation)
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    metrics = {}
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Cross-validation score
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring='f1'
        )
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
    
    # Feature importance
    feature_importance = {
        name: importance 
        for name, importance in zip(X_test.columns, model.feature_importances_)
    }
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    metrics['feature_importance'] = feature_importance
    
    # Classification report
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))
    
    logger.info(f"✅ Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"✅ Test F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"✅ ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"✅ Recall (Sensitivity): {metrics['recall']:.4f}")
    logger.info(f"✅ Precision: {metrics['precision']:.4f}")
    
    return metrics


def save_model_and_metrics(model, metrics):
    """
    Save trained model and evaluation metrics.
    
    Args:
        model: Trained model
        metrics (dict): Evaluation metrics
    """
    try:
        # Save model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"✅ Model saved to {MODEL_PATH}")
        
        # Save metrics as JSON
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'feature_importance'}
        
        with open(MODEL_METRICS_PATH, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        logger.info(f"✅ Metrics saved to {MODEL_METRICS_PATH}")
    
    except Exception as e:
        logger.error(f"Error saving model/metrics: {str(e)}")


def plot_feature_importance(metrics, save_path="src/feature_importance.png"):
    """
    Plot and save feature importance chart.
    
    Args:
        metrics (dict): Evaluation metrics containing feature_importance
        save_path (str): Path to save the plot
    """
    try:
        if 'feature_importance' not in metrics:
            logger.warning("No feature importance data available")
            return
        
        importance_dict = metrics['feature_importance']
        top_n = 15
        
        top_features = dict(list(importance_dict.items())[:top_n])
        
        plt.figure(figsize=(10, 6))
        plt.barh(list(top_features.keys()), list(top_features.values()))
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"✅ Feature importance plot saved to {save_path}")
    
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")


def train():
    """
    Main training pipeline.
    """
    logger.info("="*60)
    logger.info("Starting model training pipeline")
    logger.info("="*60)
    
    # Load data
    X, y = load_processed_data()
    if X is None or y is None:
        logger.error("Cannot proceed with training: No data available")
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, 
        random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train rain samples: {(y_train == 1).sum()}, No rain: {(y_train == 0).sum()}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Save
    save_model_and_metrics(model, metrics)
    plot_feature_importance(metrics)
    
    logger.info("="*60)
    logger.info("✅ Training pipeline complete!")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    train()
