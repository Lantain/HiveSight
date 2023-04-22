import argparse
import shutil
import os
import cv2
import json
import time
import random
import pandas as pd
from matplotlib import pyplot as plt
from src.configurators.model import Configurator
from src.pipeline import Pipeline
from src.hive_fs import HiveFs
from src.hive import Hive
from src.datasets import remo
from sklearn.model_selection import train_test_split
from src.processors import tf_zoo_models as model_processor
from src.processors import csv as csv_processor
from src.processors import labels as labels_processor
from src.processors import record_csv as record_processor
from src.processors import config as config_processor

from src.transformers.greyscale import GreyscaleTransformer
from src.transformers.rotate import RotateTransformer
from src.transformers.grey_histogram_eq import GreyHistogramEqTransformer
from src.transformers.grey_histogram_eq_clahe import GreyHistogramEqClaheTransformer
from src.transformers.color_histogram_norm import ColorHistogramNormalizeTransformer
from src.transformers.normalized_size import NormalizedSizeTransformer

import tensorflow as tf
from object_detection import model_lib_v2
from PIL import Image
from src.transformers.model import Transformer
from src import validate 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str, required=False)
    parser.add_argument('--out', type=str, required=False, default="./out")
    parser.add_argument('--imgs_dir', type=str, required=False, default=".")
    parser.add_argument('--threshold', type=float, required=False, default=.3)
    args = parser.parse_args()
    pipeline = Pipeline([
        # GreyHistogramEqClaheTransformer()
        # GreyHistogramEqTransformer()
    ])

    pipeline_out = "./out/pipe"
    if os.path.exists(pipeline_out):
        shutil.rmtree(pipeline_out)
    os.makedirs(pipeline_out, exist_ok=True)

    pl = pipeline.from_dir(args.imgs_dir)
    pl.pipe_to_dir(pipeline_out)
    hive_fs = HiveFs(out_dir=args.out, hive_path=args.hive)
    hive = Hive(hive_fs)


    files = os.listdir(pipeline_out)
    paths = list()
    for f in files:
        paths.append(f"{pipeline_out}/{f}")

    hive.analyze(paths, args.out, args.threshold)

# Generate me a scientific paper with title "Optimizing bee identification using SSD neural network architecture and tensorflow". 
# Paper should contain the following sections: Introduction, Related Work, Materials and methods, Results and discussion, Conclusions, References
# Introduction should have a following structure: why bees are important for agriculture, the number of bees is decreasing, why bee monitoring is important, how usage of neural networks could help in bee hive monitoring, how usage of neiral networks is better than traditional bee observation methods.
# For object detection task SSD Mobilenet from Tensorflow Object Detection Zoo was used. It was trained on manually annotated dataset consisting of bee photos. At first it was trained as it is, and next it was trained with image normalization techniques(gauss blur, histogram equalization) applied to the dataset and also some instruments from Tensorflow 2 Object Detection API  were utilized: random rotation to 90 degrees, random greyscale, input image resizing.
# Training results can be randomly generated in your response
