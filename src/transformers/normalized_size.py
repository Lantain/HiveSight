
from src.transformers.model import Transformer
from PIL import Image, ImageOps
import os
import cv2


class NormalizedSizeTransformer(Transformer):
    files = list()
    maxw: int = None
    maxh: int = None
    
    def __init__(self, maxw: int = None, maxh: int = None) -> None:
        super().__init__()
        self.maxw = maxw
        self.maxh = maxh
        # files = os.listdir(img_dir)
        # for f in files:
        #     self.files.append(f"{img_dir}/{f}")

    def transform(self, img: Image.Image) -> list[Image.Image]:
        # Rotate the image by 90 degrees if the width is greater than the height
        if img.width > img.height:
            img = img.rotate(90, expand=True)

        # Calculate the new size while preserving the aspect ratio
        width_ratio = self.maxw / img.width
        height_ratio = self.maxh / img.height
        scale_ratio = min(width_ratio, height_ratio)
        new_width = int(img.width * scale_ratio)
        new_height = int(img.height * scale_ratio)

        # Resize the image and save it
        img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
        return list([img_resized])
