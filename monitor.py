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
    parser.add_argument('--hive', type=str, required=True)
    parser.add_argument('--out', type=str, required=False, default="./out/monitor")
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    
    hive_fs = HiveFs(out_dir=args.out, hive_path=args.hive)
    hive = Hive(hive_fs)
    files = os.listdir(args.target_dir)
    images = list()
    for f in files:
        images.append(f"{args.target_dir}/{f}")
    hive.monitor(args.out, images)