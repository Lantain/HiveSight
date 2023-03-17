import cv2
import numpy as np
from src.transformers.model import Transformer
from PIL import Image, ImageOps

class GreyHistogramEqTransformer(Transformer):
    def transform(self, image: Image.Image) -> list[Image.Image]:
        cv2_img = self.pil_to_cv2(image)
        cv2_grey = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        equalized_grey = cv2.equalizeHist(cv2_grey)
        return list([self.cv2_to_pil(equalized_grey)])
