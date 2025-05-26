import numpy as np
import cv2
import torch
from pathlib import Path


def load_model(filepath: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(filepath, map_location=device)
    model.eval()
    model.float()
    return model


def preprocess(img):
    input_size = (416, 416)
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    rr = 1 / r
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose((2, 0, 1))
    return padded_img, rr


def make_predictions(model, X: np.ndarray) -> np.ndarray:
    extractor = lambda p: p[0] if isinstance(p, tuple) else p

    with torch.inference_mode():
        inputs = torch.from_numpy(X.astype(np.float32))
        preds = extractor(model(inputs)).cpu().numpy()
        return preds


model = load_model(Path(__file__).absolute().parent / "yolox_s_traced.pt")


def predict(image: np.ndarray):
    img, rr = preprocess(image)
    pred = make_predictions(model, img[np.newaxis,:])[0]
    pred[:, 0:4] *= rr
    return pred
