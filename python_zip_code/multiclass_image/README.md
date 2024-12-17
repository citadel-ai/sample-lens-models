# Multi-class Classification Model Package

This repository provides a **multi-class classification model** trained using Keras. It includes scripts for running predictions and a deployment-ready package.

---

## Requirements

Install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Files in the Repository

1. **`pred.py`** - Script to make predictions using the pre-trained multi-class classification model.
2. **`EfficientNetB0_keras_saved_model.keras`** - Saved multi-class classification model.
3. **`keras_test_image_multiclass.ipynb`** - Jupyter Notebook demonstrating model inference.
4. **`multiclass_model_package_image.zip`** - Deployment-ready package containing the model and scripts.

---

## Usage

### 1. Running Predictions

To make predictions on new images using `pred.py`:

```bash
python pred.py --image_path /path/to/image.jpg
```

- Replace `/path/to/image.jpg` with the path to your input image.
- Ensure the model file `EfficientNetB0_keras_saved_model.keras` is present in the same directory.

### 2. Jupyter Notebook

The `keras_test_image_multiclass.ipynb` notebook provides a step-by-step guide for:

- Loading the saved model.
- Performing inference on sample images.
- Showing predictions and results.

---

## Model Details

- **Architecture**: EfficientNetB0
- **Framework**: Keras
- **Task**: Multi-class Image Classification

---

## Notes

- Ensure all dependencies listed in `requirements.txt` are installed.

---

## Error Handling

1. **Missing Model File**:
   - If `EfficientNetB0_keras_saved_model.keras` is not found, a `FileNotFoundError` is raised.

2. **Invalid Input**:
   - If the input image path is incorrect, an appropriate error message will be displayed.

