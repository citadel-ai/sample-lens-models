import numpy as np
import pandas as pd
import os
from pycaret.regression import load_model

# Specify the model file name
MODEL_FILE_NAME = "best_model_weight_pounds.pkl"

# Load the model file from the current execution directory
current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, MODEL_FILE_NAME)

# Remove the extension (since load_model automatically appends it)
if model_path.endswith('.pkl'):
    model_path = model_path[:-4]  # Remove the '.pkl' at the end

# Check if the file exists
if not os.path.exists(model_path + ".pkl"):
    raise FileNotFoundError(f"Model file not found at {model_path}.pkl")

# Load the model
model = load_model(model_path)

def predict(inputs: pd.DataFrame) -> np.ndarray:
    """
    Function to make predictions on input data
    inputs: pd.DataFrame - Data to use for prediction
    return: np.ndarray - Prediction results
    """
    # Return the prediction results as a numpy array
    predictions = model.predict(inputs)
    return predictions
