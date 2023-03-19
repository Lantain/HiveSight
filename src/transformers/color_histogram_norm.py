import cv2
import numpy as np
from src.transformers.model import Transformer
from PIL import Image, ImageOps

class ColorHistogramNormalizeTransformer(Transformer):
    def transform(self, image: Image.Image) -> list[Image.Image]:
        np_image = np.array(image)
        cv2_img_yuv = cv2.cvtColor(np_image, cv2.COLOR_RGB2YUV)
        cv2_img_yuv[:,:,0] = cv2.equalizeHist(cv2_img_yuv[:,:,0])
        result_img = cv2.cvtColor(cv2_img_yuv, cv2.COLOR_YUV2RGB)
        return list([self.cv2_to_pil(result_img)])
    