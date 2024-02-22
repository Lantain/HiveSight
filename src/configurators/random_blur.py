from src.configurators.model import Configurator
import os
from object_detection.protos import preprocessor_pb2

class RandomBlurConfigurator(Configurator):
    def modify(self, configs: dict):
        augmentation = configs['train_config'].data_augmentation_options.add()
        augmentation.random_patch_gaussian.CopyFrom(preprocessor_pb2.RandomPatchGaussian())