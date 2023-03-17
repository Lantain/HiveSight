import cv2
import numpy as np
from src.transformers.model import Transformer
from PIL import Image, ImageOps

class GreyHistogramEqClaheTransformer(Transformer):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def transform(self, image: Image.Image) -> Image.Image:
        cv2_img = self.pil_to_cv2(image)
        cv2_grey = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        clahe_equalized_grey = self.clahe.apply(cv2_grey)
        return list([self.cv2_to_pil(clahe_equalized_grey)])
