import cv2
import numpy as np
from src.transformers.model import Transformer
from PIL import Image, ImageOps

class HistgramEqualizeColorCLAHE(Transformer):
    clip_limit = None
    tile_grid_size = None

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        super().__init__()

    def transform(self, image: Image.Image) -> list[Image.Image]:
        np_image = np.array(image)
        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)

        # Merge the enhanced L channel with the original A and B channels
        enhanced_lab_image = cv2.merge((l_channel_clahe, a_channel, b_channel))

        # Convert the LAB image back to BGR color space
        enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

        return list([Image.fromarray(enhanced_image)])

    