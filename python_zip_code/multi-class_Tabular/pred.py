import numpy as np
import pandas as pd
from pathlib import Path
from pycaret.classification import load_model
import os

# Specify the model file path
MODEL_PATH = Path(__file__).absolute().parent / "best_model_multiclass.pkl"

# Check if the file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the model
model = load_model(str(MODEL_PATH.with_suffix('')))

def predict(inputs: pd.DataFrame) -> np.ndarray:
    """
    Function to make predictions on input data for multi-class classification.
    
    Args:
        inputs (pd.DataFrame): Data to use for prediction.
        
    Returns:
        np.ndarray: Prediction probabilities for each class.
    """
    # Return probabilities for each class
    probabilities = model.predict_proba(inputs)
    
    # Check if probabilities are valid
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise ValueError("Model returned invalid probabilities. Please check the model.")
    
    return probabilities
