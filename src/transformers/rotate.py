from src.transformers.model import Transformer
from PIL import Image, ImageOps

class RotateTransformer(Transformer):
    def __init__(self, degree: int) -> None:
        super().__init__()
        self.degree = degree

    degree: int = 0
    def transform(self, image: Image.Image) -> list[Image.Image]:
        result = image.rotate(self.degree)
        return list([result])
    