import base64
from io import BytesIO

import numpy as np
import orjson
import scipy.ndimage as ndimage
import torch
from flask import Flask, request
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, ScaleIntensityRange, Spacing
from scipy.special import softmax

from unetr import UNETR


def load_model(filepath: str) -> UNETR:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device),
                          strict=False)
    model.eval()
    return model


transform = Compose([
    Spacing(pixdim=(1.5, 1.5, 2), mode="bilinear"),
    ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0,
                        clip=True),
])


def resample_3d(img, target_size):
    imx, imy, imz, c = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy),
                  float(tz) / float(imz), 1)
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def preprocess(img: MetaTensor) -> MetaTensor:
    img = torch.einsum("dhwc->cwhd", img)
    return transform(img)


def postprocess(pred: MetaTensor, orig_size) -> np.ndarray:
    pred = torch.einsum("cwhd->dhwc", pred)
    pred = resample_3d(pred, orig_size)
    pred = softmax(pred, -1)
    pred = np.argmax(pred, -1)
    return pred


def predict_single(model, X: MetaTensor) -> MetaTensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        preds = sliding_window_inference(X[np.newaxis, :].to(device),
                                         (96, 96, 96),
                                         4,
                                         model,
                                         overlap=0.5)
        return preds[0].cpu()


app = Flask(__name__)
model = load_model("UNETR_model_best_acc.pth")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img = np.load(BytesIO(base64.b64decode(data["image"])))
    orig_size = img.shape[:3]
    affine = data["metadata"]["affine"]
    img = MetaTensor(img, affine=affine)
    img = preprocess(img)
    pred = predict_single(model, img)
    pred = postprocess(pred, orig_size)
    buf = BytesIO()
    np.save(buf, pred, allow_pickle=False)
    res = {"predictions": base64.b64encode(buf.getvalue()).decode('utf-8')}
    return orjson.dumps(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
