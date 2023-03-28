from src.configurators.model import Configurator
import os
from object_detection.protos import preprocessor_pb2

class RandomRotation90Configurator(Configurator):
    def modify(self, configs: dict):
        augmentation = configs['train_config'].data_augmentation_options.add()
        augmentation.random_rotation90.CopyFrom(preprocessor_pb2.RandomRotation90())