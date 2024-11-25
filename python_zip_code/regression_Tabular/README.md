# Regression Model Package

This repository provides a regression model trained to predict birth weight based on the dataset `natality_2001_train_100k.csv`. The project uses [PyCaret](https://pycaret.org/) for model training, and includes scripts for data preparation, training, and making predictions.

## Requirements

To set up the environment, install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Files in the Repository

1. **natality_2001_train_100k.csv** - Training dataset used for model training and evaluation.
2. **pred.py** - Script to make predictions using a pre-trained model.
3. **best_model_weight_pounds.pkl** - The saved regression model.
4. **pyCaret_test.ipynb** - Jupyter Notebook demonstrating model training using PyCaret.

## Using the Jupyter Notebook (`pyCaret_test.ipynb`)

### Step-by-Step Guide

1. **Load the Dataset**: The notebook loads the training data from `natality_2001_train_100k.csv`.
   
2. **PyCaret Setup**: 
   - Set up a PyCaret regression environment with `weight_pounds` as the target variable.
   - Normalize the data and set a random seed for reproducibility.

3. **Model Selection and Tuning**:
   - Use PyCaret's `compare_models` to evaluate and select a model with the lowest RMSE.
   - Fine-tune the selected model to optimize its parameters further.

4. **Visualization**:
   - Plot residuals and prediction errors to evaluate model performance.

5. **Save the Model**:
   - Save the tuned model as `best_model_weight_pounds.pkl` using PyCaret’s `save_model`.

6. **Export Model for Deployment**:
   - The last cell creates a ZIP file, `regression_model_package.zip`, which includes the model, `pred.py`, and `requirements.txt` for easy deployment.

## Making Predictions with `pred.py`

The `pred.py` script loads a pre-trained model and uses it to make predictions on new input data.

### Usage

The script can be used as follows:

```bash
python pred.py
```

### How `pred.py` Works

- **Load Model**: Loads the model from `best_model_weight_pounds.pkl`.
- **Prepare Data**: Prepares sample input data for prediction.
- **Make Prediction**: Uses the loaded model to generate predictions and outputs the results.

## Model Prediction Script

This script loads a pre-trained regression model and makes predictions on input data.

### Usage

1. **Specify Model File**: The script expects a model file named `best_model_weight_pounds.pkl` in the same directory. Note that PyCaret’s `load_model` function appends `.pkl` automatically, so the filename extension is removed in the code before loading.

2. **Code Overview**:
   - The model path is set and verified for existence, with `.pkl` removed if already present in the filename.
   - The model is loaded using PyCaret’s `load_model` function.
   - A `predict` function is defined to take a `pandas.DataFrame` as input and output predictions as a `numpy.ndarray`.

3. **Function Details**:
   - `predict(inputs: pd.DataFrame) -> np.ndarray`: Accepts a DataFrame containing input data and returns prediction results as a numpy array.

### Error Handling
If the model file isn’t found at the specified location, a `FileNotFoundError` is raised.



