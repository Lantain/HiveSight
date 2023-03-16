from transformers.model import Transformer
from PIL import Image, ImageOps

class RotateTransformer(Transformer):
    def transform(image_path: str):
        im = Image.open(image_path)
        im.rotate(45)
        return im
    