import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

def train_matching_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42
) -> LGBMClassifier:
    """
    Trains a LightGBM gradient boosted tree model for pair-level entity matching.
    """
    print(f"Training LightGBM on {len(X_train):,} candidate pairs (Positives: {np.sum(y_train):,}, Negatives: {len(y_train) - np.sum(y_train):,})...", flush=True)
    
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print("LightGBM training complete!", flush=True)
    return model

def save_model(model: LGBMClassifier, model_path: str):
    """Saves trained model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}", flush=True)

def load_model(model_path: str) -> LGBMClassifier:
    """Loads trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
