# Multi-Label Classification Model Package

This repository provides a multi-label classification model trained using Scikit-learn. It includes scripts for data preparation, model training, and making predictions.

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
5. **`Scikit-learn_multilabel.ipynb`** - Jupyter Notebook demonstrating model training and evaluation.
6. **`multilabel_model_package.zip`** - Deployment-ready package containing the model and scripts.

---

## Using the Jupyter Notebook (`Scikit-learn_multilabel.ipynb`)

### Step-by-Step Guide

1. **Load the Dataset**:
   - The notebook demonstrates loading and processing a dataset suitable for multi-label classification.

2. **Model Training**:
   - The model is built using Scikit-learn's `Pipeline` with preprocessing steps and a classifier (e.g., Random Forest or Support Vector Classifier).
   - Metrics such as `hamming_loss`, `precision`, and `recall` are used to evaluate model performance.

3. **Save the Model**:
   - The trained model pipeline is saved as `multi_label_pipeline_model.joblib` using `joblib.dump`.

4. **Visualize Results**:
   - The notebook generates classification reports and visualizes performance metrics.

---

## Script Details

### How `pred.py` Works

1. **Load Model**:
   - The script expects a model file named `multi_label_pipeline_model.joblib` in the same directory.
   - The model is loaded using `joblib.load`.

2. **Prepare Data**:
   - Input data must be a `pandas.DataFrame` with the same structure as the training data.

3. **Make Predictions**:
   - The `predict` function outputs binary labels for each class.

---

## Error Handling

1. **Missing Model File**:
   - Raises a `FileNotFoundError` if the model file is not found.

2. **Invalid Input**:
   - Raises a `ValueError` if the input is not a `pandas.DataFrame`.

---
