# Binary Classification Model Package

This repository provides a binary classification model trained to predict a target variable, such as tipping behavior, using PyCaret. The project includes scripts for data preparation, model training, and making predictions.

---

## Requirements

To set up the environment, install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Files in the Repository

1. **`taxi_2020_test_converted_1000.csv`** - Example dataset used for testing predictions.
2. **`pred.py`** - Script to make predictions using a pre-trained binary classification model.
3. **`best_model_with_tip.pkl`** - The saved classification model.
4. **`pyCaret_test.ipynb`** - Jupyter Notebook demonstrating model training using PyCaret.

---

## Using the Jupyter Notebook (`pyCaret_test.ipynb`)

### Step-by-Step Guide

1. **Load the Dataset**:
   - The notebook uses a dataset structured with features relevant to tipping behavior or similar binary outcomes.

2. **PyCaret Setup**:
   - Set up a PyCaret classification environment with a binary target variable, such as `with_tip`.
   - Preprocess the data and set a random seed for reproducibility.

3. **Model Selection and Tuning**:
   - Use PyCaret’s `compare_models` to evaluate and select a model with the highest AUC or lowest log loss.
   - Fine-tune the selected model to optimize its parameters further.

4. **Visualization**:
   - Generate plots such as the ROC curve and feature importance to assess model performance.

5. **Save the Model**:
   - Save the tuned model as `best_model_with_tip.pkl` using PyCaret’s `save_model`.

6. **Export Model for Deployment**:
   - The last cell creates a deployment-ready ZIP file, `classification_model_package.zip`, which includes the model, `pred.py`, and `requirements.txt`.

---

## Script Details

### How `pred.py` Works

1. **Load Model**:
   - The script expects a model file named `best_model_with_tip.pkl` in the same directory.
   - The model is loaded using PyCaret's `load_model` function, which automatically handles the `.pkl` file extension.

2. **Prepare Data**:
   - The input data must be a `pandas.DataFrame` with the same feature structure as the training data.

3. **Make Prediction**:
   - The `predict` function takes the input data and outputs either class labels or probabilities.

---

## Model Prediction Script Details

### Usage

1. **Specify Model File**:
   - The script expects `best_model_with_tip.pkl` in the same directory. PyCaret’s `load_model` function removes the `.pkl` extension before loading.

2. **Code Overview**:
   - The model path is verified to ensure the file exists.
   - The model is loaded using PyCaret’s `load_model`.
   - A `predict` function is provided to handle new input data.

3. **Function Details**:
   - **`predict(inputs: pd.DataFrame, return_proba: bool = False, threshold: float = 0.5) -> np.ndarray`**:
     - Accepts a DataFrame of input data.
     - Returns class labels or probabilities as a numpy array.

---

### Error Handling

1. **Model File Missing**:
   - If the model file is not found, a `FileNotFoundError` is raised:
     ```python
     raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
     ```

2. **Invalid Input**:
   - If the input is not a `pandas.DataFrame`, the script raises a `ValueError`:
     ```python
     raise ValueError("Input data must be a pandas DataFrame.")
     ```

