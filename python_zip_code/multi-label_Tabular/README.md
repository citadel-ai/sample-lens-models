# Multi-class Classification Model Package

This repository provides a **multi-class classification model** trained using Keras. It includes scripts for running predictions, testing, and a deployment-ready package.

---

## Requirements

Install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Files in the Repository

1. **`EfficientNetB0_keras_saved_model.keras`** - The saved multi-class classification model.
2. **`pred.py`** - Script to make predictions using the pre-trained multi-class model.
3. **`keras_test_image_multiclass.ipynb`** - Jupyter Notebook demonstrating model inference and testing.
4. **`multiclass_model_package_image.zip`** - A ZIP file containing the deployment-ready model and scripts.
5. **`requirements.txt`** - List of required dependencies.

---

## Script Details

### How `pred.py` Works

1. **Load the Model**:
   - The script loads the pre-trained `EfficientNetB0_keras_saved_model.keras` using Keras.

2. **Prepare Input Image**:
   - Input images are resized to 224x224 and normalized to match the model's requirements.

3. **Make Predictions**:
   - The script outputs the predicted class label and confidence score for the input image.

#### Example Usage:

```bash
python pred.py --image_path /path/to/image.jpg
```

Replace `/path/to/image.jpg` with the path to your input image.

---

## Using the Jupyter Notebook

### How `keras_test_image_multiclass.ipynb` Works

1. **Load the Pre-trained Model**:
   - The notebook demonstrates loading the `EfficientNetB0_keras_saved_model.keras` model.
2. **Perform Inference**:
   - The notebook runs predictions on sample images and outputs the class labels and confidence scores.

3. **Showing Results**:
   - Outputs predictions results for better understanding.

---

## Model Details

- **Model Architecture**: EfficientNetB0
- **Framework**: Keras
- **Task**: Multi-class Image Classification

---

## Notes

- Ensure the model file `EfficientNetB0_keras_saved_model.keras` is in the same directory as `pred.py`.
- Install all dependencies listed in `requirements.txt` before running any script.

---

## Error Handling

1. **Missing Model File**:
   - Raises a `FileNotFoundError` if `EfficientNetB0_keras_saved_model.keras` is not found.

2. **Invalid Image Path**:
   - Displays an error message if the specified image file does not exist.

3. **Input Size Mismatch**:
   - Automatically resizes input images to 224x224 for compatibility with the model.

---
