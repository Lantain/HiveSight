import argparse
import shutil
import os
import cv2
import json
import time
import random
import pandas as pd
from src.hive_fs import HiveFs
from src.hive import Hive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str, required=False)
    parser.add_argument('--out', type=str, required=False, default="./out")
    args = parser.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    hive_fs = HiveFs(out_dir=args.out, hive_path=args.hive)
    hive = Hive(hive_fs)
    hive.evaluate()