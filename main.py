import argparse
import shutil
import os
import json
import time

from src.hive import Hive
from src.hive_fs import HiveFs
from src.transformers.greyscale import GreyscaleTransformer
from src.transformers.rotate import RotateTransformer
from src.transformers.grey_histogram_eq import GreyHistogramEqTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str, required=False)
    parser.add_argument('--dir', type=str, required=False, default=".")
    args = parser.parse_args()
    
    hive_fs = HiveFs(out_dir="out/", dir="out/myhive", model='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
    hive = Hive(hive_fs)
    hive.set_train_params(10000, 32)
    hive.set_dataset("remo", "./source/remo")
    hive.make(transformers=[
        # RotateTransformer(45), 
        GreyscaleTransformer(), 
        GreyHistogramEqTransformer()
    ])
