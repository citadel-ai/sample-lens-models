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

def predict(inputs: pd.DataFrame, return_proba: bool = False) -> pd.DataFrame:
    """
    Function to make predictions on input data for multi-class classification.
    
    Args:
        inputs (pd.DataFrame): Data to use for prediction.
        return_proba (bool): Whether to return class probabilities instead of labels.
        
    Returns:
        pd.DataFrame: Prediction results (labels or probabilities).
    """

    if return_proba:
        # Return probabilities for each class
        probabilities = model.predict_proba(inputs)
        if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
            raise ValueError("Model returned invalid probabilities. Please check the model.")
        # Convert to DataFrame with class names as columns
        probabilities_df = pd.DataFrame(probabilities, columns=model.classes_)
        return probabilities_df
    else:
        # Predict class labels
        labels = model.predict(inputs)
        # Convert to DataFrame
        labels_df = pd.DataFrame(labels, columns=['Predicted_Label'])
        return labels_df
    