import argparse
import shutil
import os
import json
import time
import matplotlib.pyplot as plt
from src.pipeline import Pipeline

from src.hive import Hive
from src.hive_fs import HiveFs
from src.transformers.greyscale import GreyscaleTransformer
from src.transformers.rotate import RotateTransformer
from src.configurators.random_rotation_90 import RandomRotation90Configurator
from src.transformers.grey_histogram_eq import GreyHistogramEqTransformer
from src.transformers.color_histogram_norm import ColorHistogramNormalizeTransformer
from src.transformers.normalized_size import NormalizedSizeTransformer
from src.validate import get_detections, get_processed_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str, required=False)
    parser.add_argument('--dir', type=str, required=False, default=".")
    args = parser.parse_args()
    pipeline = Pipeline([
        # RotateTransformer(45), 
        # GreyscaleTransformer(), 
        # GreyHistogramEqTransformer(),
        # NormalizedSizeTransformer(maxh=80, maxw=80),
        # ColorHistogramNormalizeTransformer()
    ])
    hive_fs = HiveFs(out_dir="out/", dir="out/myhive", model='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
    hive = Hive(hive_fs)
    hive.set_train_params(20000, 2)
    hive.set_dataset("remo", "./source/remo")
    hive.make(pipeline=pipeline, configurators=list([RandomRotation90Configurator()]))
    # hive.train()
    # hive.generate_inference()
    hive.pack("myhive2.hive")
    # img_path = f"~/counted_bees/24_bees.jpg"
    # model_fn = hive.use_checkpoint()
    # detections = get_detections(model_fn, img_path)
    # processed_image = get_processed_image(detections, hive.fs.get_paths()["HIVE_DIR_LABELS"], img_path)
    # fig = plt.figure(figsize=(14, 10))
    # fig.add_subplot(1, 1, 1)
    # plt.imshow(processed_image)
    # plt.axis('off')
    # plt.title("Bruh")
    # fig.show()
    # fig.savefig("figa.jpg")