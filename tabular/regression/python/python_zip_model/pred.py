import numpy as np
import pandas as pd
from pathlib import Path
from pycaret.regression import load_model

# Specify the model file path
MODEL_PATH = Path(__file__).absolute().parent / "best_model_weight_pounds.pkl"

# Check if the file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the model (load_model automatically appends the necessary suffix)
model = load_model(MODEL_PATH.with_suffix(""))

def predict(inputs: pd.DataFrame) -> np.ndarray:
    """
    Function to make predictions on input data
    inputs: pd.DataFrame - Data to use for prediction
    return: np.ndarray - Prediction results
    """
    # Return the prediction results as a numpy array
    predictions = model.predict(inputs)
    return predictions