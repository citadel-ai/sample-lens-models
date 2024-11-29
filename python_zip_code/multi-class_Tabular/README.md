# Multi-Class Classification Model Package

This repository provides a multi-class classification model trained using PyCaret. It includes scripts for data preparation, model training, and making predictions.

---

## Requirements

Install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Files in the Repository

1. **`class.txt`** - A text file mapping class indices to their labels.
2. **`test_label_idx.csv`** - Example dataset with ground truth class indices for testing predictions.
3. **`pred.py`** - Script to make predictions using a pre-trained multi-class classification model.
4. **`best_model_multiclass.pkl`** - The saved multi-class classification model.
5. **`pyCaret_test_multiclass.ipynb`** - Jupyter Notebook demonstrating model training using PyCaret.
6. **`multiclass_model_package.zip`** - Deployment-ready package containing the model and scripts.

---

## Using the Jupyter Notebook (`pyCaret_test_multiclass.ipynb`)

### Step-by-Step Guide

1. **Load the Dataset**:
   - The notebook uses a dataset with features relevant to the classification task.

2. **PyCaret Setup**:
   - Set up a PyCaret multi-class classification environment with the appropriate target variable.
   - Preprocess the data and set a random seed for reproducibility.

3. **Model Selection and Tuning**:
   - Use PyCaret’s `compare_models` to evaluate and select the best-performing model.
   - Fine-tune the selected model to optimize its parameters.

4. **Visualization**:
   - Generate plots such as confusion matrices and feature importance to assess model performance.

5. **Save the Model**:
   - Save the tuned model as `best_model_multiclass.pkl` using PyCaret’s `save_model`.

6. **Export Model for Deployment**:
   - The final cell creates a deployment-ready ZIP file, `multiclass_model_package.zip`.

---

## Making Predictions with `pred.py`

The `pred.py` script loads a pre-trained multi-class classification model and makes predictions on new data.

### Usage

To use the script programmatically:

```python
from pred import predict
import pandas as pd

# Example input data
data = pd.read_csv("test_label_idx.csv")

# Predict class labels
predicted_labels = predict(data)
print("Predicted Labels:", predicted_labels)

# Predict probabilities for each class
predicted_probabilities = predict(data, return_proba=True)
print("Predicted Probabilities:", predicted_probabilities)
```

---

## Script Details

### How `pred.py` Works

1. **Load Model**:
   - The script expects a model file named `best_model_multiclass.pkl` in the same directory.
   - The model is loaded using PyCaret's `load_model` function.

2. **Prepare Data**:
   - Input data must be a `pandas.DataFrame` with the same feature structure as the training data.

3. **Make Predictions**:
   - The `predict` function outputs either class labels or probabilities for each class.

### Error Handling

1. **Missing Model File**:
   - Raises a `FileNotFoundError` if the model file is not found.

2. **Invalid Input**:
   - Raises a `ValueError` if the input is not a `pandas.DataFrame`.

