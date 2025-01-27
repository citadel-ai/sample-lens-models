'''Template preprocessing file for PyTorch image models

This is an example preprocessing file that can be uploaded to Lens for PyTorch
image models.
'''

import numpy as np
from torchvision import transforms


def preprocess_function(img: np.ndarray) -> np.ndarray:
    '''Preprocess function that transforms an RGB image into a format that is
    expected by the model.
    '''
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(img).numpy()
