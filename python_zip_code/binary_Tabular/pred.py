import numpy as np
import pandas as pd
from pathlib import Path
from pycaret.classification import load_model

# Specify the model file path
MODEL_PATH = Path(__file__).absolute().parent / "best_model_rank.pkl"

# Check if the file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the model (load_model automatically appends the necessary suffix)
model = load_model(MODEL_PATH.with_suffix(""))

def predict(inputs: pd.DataFrame, return_proba: bool = False, threshold: float = 0.5) -> np.ndarray:
    """
    Function to make predictions on input data for binary classification.
    
    Args:
        inputs (pd.DataFrame): Data to use for prediction.
        return_proba (bool): Whether to return class probabilities instead of labels.
        threshold (float): Threshold for converting probabilities to class labels. Default is 0.5.
        
    Returns:
        np.ndarray: Prediction results (labels or probabilities).
    """
    # Ensure input is a DataFrame
    if not isinstance(inputs, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if return_proba:
        # Return probabilities for the positive class (e.g., class 1)
        probabilities = model.predict_proba(inputs)[:, 1]
        return probabilities
    else:
        # Predict probabilities and convert to class labels based on threshold
        probabilities = model.predict_proba(inputs)[:, 1]
        labels = (probabilities >= threshold).astype(int)
        return labels
