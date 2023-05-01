import cv2
import numpy as np
from src.transformers.model import Transformer
from PIL import Image, ImageOps

class NoiseReductionTransformer(Transformer):
    kernel_size = 3

    def __init__(self, method: str, kernel = 3) -> None:
        self.method = method
        self.kernel_size = kernel
        super().__init__()

    def transform(self, image: Image.Image) -> list[Image.Image]:
        np_image = np.array(image)
        if self.method == 'gaussian':
            blurred_image = cv2.GaussianBlur(np_image, (self.kernel_size, self.kernel_size), 0)
        elif self.method == 'median':
            blurred_image = cv2.medianBlur(np_image, self.kernel_size)
        else:
            raise ValueError("Invalid method. Choose 'gaussian' or 'median'.")
        
        return list([self.cv2_to_pil(blurred_image)])
    