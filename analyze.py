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
    args = parser.parse_args()
    pipeline = Pipeline([
        GreyHistogramEqClaheTransformer
    ])
    hive_fs = HiveFs(out_dir=args.out, hive_path=args.hive)
    hive = Hive(hive_fs)

    files = os.listdir(args.imgs_dir)
    paths = list()
    for f in files:
        paths.append(f"{args.imgs_dir}/{f}")

    hive.analyze(paths, args.out)
    