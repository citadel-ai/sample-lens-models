# Multi-Label Classification Model Package

This repository provides a multi-label classification model trained using PyCaret. It includes scripts for data preparation, model training, and making predictions.

---

## Requirements

Install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Files in the Repository

1. **`classes.txt`** - A text file mapping class indices to their labels.
2. **`test_label_idx.csv`** - Example dataset with ground truth labels for testing predictions.
3. **`pred.py`** - Script to make predictions using a pre-trained multi-label classification model.
4. **`multi_label_pipeline_model.joblib`** - The saved multi-label classification model pipeline.
5. **`pyCaret_test_multilabel.ipynb`** - Jupyter Notebook demonstrating model training using PyCaret.
6. **`multilabel_model_package.zip`** - Deployment-ready package containing the model and scripts.

---

## Using the Jupyter Notebook (`pyCaret_test_multilabel.ipynb`)

### Step-by-Step Guide

1. **Load the Dataset**:
   - The notebook uses a dataset with features relevant to the classification task.

2. **PyCaret Setup**:
   - Set up a PyCaret environment for multi-label classification with the appropriate target variables.
   - Preprocess the data and set a random seed for reproducibility.

3. **Model Selection and Tuning**:
   - Use PyCaret’s `compare_models` to evaluate and select the best-performing model.
   - Fine-tune the selected model to optimize its parameters.

4. **Visualization**:
   - Generate plots such as label-wise performance metrics and feature importance to assess model performance.

5. **Save the Model**:
   - Save the tuned model as `multi_label_pipeline_model.joblib` using PyCaret’s `save_model`.

6. **Export Model for Deployment**:
   - The final cell creates a deployment-ready ZIP file, `multilabel_model_package.zip`.

---

## Making Predictions with `pred.py`

The `pred.py` script loads a pre-trained multi-label classification model and makes predictions on new data.

### Usage

To use the script programmatically:

```python
from pred import predict
import pandas as pd

# Example input data
data = pd.read_csv("test_label_idx.csv")

# Predict labels for each instance
predicted_labels = predict(data)
print("Predicted Labels:", predicted_labels)

# Predict probabilities for each label
predicted_probabilities = predict(data, return_proba=True)
print("Predicted Probabilities:", predicted_probabilities)
```

---

## Script Details

### How `pred.py` Works

1. **Load Model**:
   - The script expects a model file named `multi_label_pipeline_model.joblib` in the same directory.
   - The model is loaded using `joblib.load`.

2. **Prepare Data**:
   - Input data must be a `pandas.DataFrame` with the same feature structure as the training data.

3. **Make Predictions**:
   - The `predict` function outputs either binary labels for each class or probabilities for each label.

### Error Handling

1. **Missing Model File**:
   - Raises a `FileNotFoundError` if the model file is not found.

2. **Invalid Input**:
   - Raises a `ValueError` if the input is not a `pandas.DataFrame`.