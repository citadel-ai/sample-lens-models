import numpy as np
import cv2
import torch
from pathlib import Path

# Use CUDA if a GPU is available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model path
MODEL_PATH = Path(__file__).absolute().parent / "yolox_s_traced.pt"
# Load the model
model = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval().float()


def _preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize the image and add padding.

    Args:
        img (np.ndarray): Input image

    Returns:
        tuple[np.ndarray, float]: Padded image and inverse scale factor
                                  to map back to original size
    """
    input_size = (416, 416)
    pad_color = 114

    # Initialize the output image with the padding color
    if img.ndim == 3:
        padded_img = np.full(
            (input_size[0], input_size[1], 3), pad_color, dtype=np.uint8
        )
    else:
        padded_img = np.full(input_size, pad_color, dtype=np.uint8)

    # Resize image while preserving aspect ratio
    scale = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    resized_img = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)

    # Place the resized image onto the padded canvas
    padded_img[:new_h, :new_w] = resized_img

    # Convert from (H, W, C) to (C, H, W)
    padded_img = padded_img.transpose((2, 0, 1))
    return padded_img, 1 / scale


def _make_predictions(model, X: np.ndarray) -> np.ndarray:
    """Run model inference on input data.

    Args:
        model: A PyTorch model
        X (np.ndarray): Input data batch

    Returns:
        np.ndarray: Model predictions
    """
    # If the model returns a tuple, use the first element
    extractor = lambda p: p[0] if isinstance(p, tuple) else p

    with torch.inference_mode():
        inputs = torch.from_numpy(X.astype(np.float32))
        preds = extractor(model(inputs)).cpu().numpy()
        return preds


def predict(image: np.ndarray) -> np.ndarray:
    """Predict bounding boxes for a single image.

    Args:
        image (np.ndarray): Input image in (H, W, C) format

    Returns:
        np.ndarray: Predicted bounding boxes and class probabilities.
                    Shape is (M, K), where M is the number of detections and
                    K = 4 (bbox coords) + 1 (object confidence) + num_classes.
                    Bounding box format is [center_x, center_y, width, height].
    """
    img, scale_inv = _preprocess(image)
    pred = _make_predictions(model, img[np.newaxis, :])[0]
    # Rescale bounding box coordinates back to the original image size
    pred[:, :4] *= scale_inv
    return pred
