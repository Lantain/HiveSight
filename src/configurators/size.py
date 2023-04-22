from src.configurators.model import Configurator
import os
from object_detection.protos import preprocessor_pb2

class SizeConfigurator(Configurator):
    width: int = 320
    height: int = 320
    def __init__(self, width, heigth) -> None:
        self.width = width
        self.height = heigth
        super().__init__()
        
    def modify(self, configs: dict):
        configs['model'].ssd.image_resizer.fixed_shape_resizer.height = self.height
        configs['model'].ssd.image_resizer.fixed_shape_resizer.width = self.width