import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# パス設定
MODEL_PATH = Path(__file__).absolute().parent / "EfficientNetB0_keras_saved_model.keras"

# モデルのロード
model = load_model(MODEL_PATH)

def predict(image: np.ndarray) -> np.ndarray:
    """
    Returns the prediction for a single image as a numpy array.

    Parameters:
        image (np.ndarray): Input image in the shape (H, W, C),
                            where H is height, W is width, and C is channels.

    Returns:
        np.ndarray: Predicted mask as an array of 0-indexed class indices
                    for each pixel in the shape (H, W).
    """
    # 入力画像の形状確認
    if image.ndim != 3:
        raise ValueError("Input image must have 3 dimensions: (H, W, C).")


    # バッチ次元を追加 (モデルは (1, H, W, C) の形状を期待する)
    image_batch = np.expand_dims(image, axis=0)

    # モデル予測
    predictions = model.predict(image_batch)

    return predictions[0]

