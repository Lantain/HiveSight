from transformers.model import Transformer
from PIL import Image, ImageOps

class RotateTransformer(Transformer):
    def transform(image: Image.Image):
        transformed_image = ImageOps.mirror(image)
        return transformed_image
    