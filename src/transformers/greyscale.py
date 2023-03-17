import cv2
import numpy as np
from src.transformers.model import Transformer
from PIL import Image, ImageOps

class GreyscaleTransformer(Transformer):
    def transform(self, image: Image.Image) -> list[Image.Image]:
        np_image = np.array(image)
        cv2_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        im_pil = Image.fromarray(gray)
        return list([im_pil])
    