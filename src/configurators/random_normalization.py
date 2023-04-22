from src.configurators.model import Configurator
import os
from object_detection.protos import preprocessor_pb2

class RandomNormalizationConfigurator(Configurator):
    def modify(self, configs: dict):
        augmentation = configs['train_config'].data_augmentation_options.add()
        augmentation.random_adjust_saturation.CopyFrom(preprocessor_pb2.RandomAdjustSaturation())
        augmentation.random_adjust_hue.CopyFrom(preprocessor_pb2.RandomAdjustHue())
        augmentation.random_adjust_contrast.CopyFrom(preprocessor_pb2.RandomAdjustContrast())
        augmentation.random_adjust_brightness.CopyFrom(preprocessor_pb2.RandomAdjustBrightness())