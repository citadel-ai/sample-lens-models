import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# モデルファイルのパスを指定
MODEL_PATH = Path(__file__).absolute().parent / "multi_label_pipeline_model.joblib"

# ファイルが存在するか確認
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# モデルの読み込み
model = joblib.load(MODEL_PATH)

def predict(inputs: pd.DataFrame) -> np.ndarray:
    """
    マルチラベル分類モデルで入力データに対して予測を行う関数。
    
    Args:
        inputs (pd.DataFrame): 予測に使用するデータ。
        
    Returns:
        np.ndarray: 各ラベルの予測クラス。
    """
    # 必要な前処理を行う
    inputs_preprocessed = model.named_steps['preprocessor'].transform(inputs)
    
    # 予測を行う
    predictions = model.named_steps['classifier'].predict(inputs_preprocessed)
    
    return predictions
