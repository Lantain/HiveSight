from src.transformers.model import Transformer
from PIL import Image, ImageOps

class MirrorTransformer(Transformer):
    def transform(self, image: Image.Image) -> list[Image.Image]:
        transformed_image = ImageOps.mirror(image)
        return list([transformed_image])
    