import argparse
import shutil
import os
import json
import time

from src.hive import Hive
from src.hive_fs import HiveFs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str)
    parser.add_argument('--dir', type=str, required=False, default=".")
    args = parser.parse_args()
    
    hive_fs = HiveFs(hive_path=f"./out/asd.hive")
    hive = Hive(hive_fs)