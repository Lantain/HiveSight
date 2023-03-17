import cv2
import numpy as np
from PIL import Image

class Transformer:
    def cv2_to_pil(self, arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr)

    def pil_to_cv2(self, img: Image.Image) -> np.array:
        np_image = np.array(img)
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    def transform(image_path: str):
        print("Not implemented")