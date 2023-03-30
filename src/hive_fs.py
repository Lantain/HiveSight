import argparse
import shutil
import os
import json
import time
import re
import random
import pandas as pd
from src.pipeline import Pipeline
from src.processors import tf_zoo_models as model_processor
from src.processors import labels as labels_processor
from src.datasets import remo
from src.utils.util import natural_keys, remove_files_from_dir
from PIL import Image


class ImgRegion:
    x: int
    y: int
    xmax: int
    ymax: int

    def __init__(self, x, y, xmax, ymax) -> None:
        self.x = x
        self.y = y
        self.xmax = xmax
        self.ymax = ymax

    def crop_from_image(self, img: Image.Image, out_file):
        part = img.crop((self.x, self.y, self.xmax, self.ymax))
        part.save(out_file)


class HiveFs:
    dir = None
    model = None
    out = "./out"

    def __init__(self,  out_dir: str = "./out", dir: str = None, model: str = None, hive_path: str = None):
        if hive_path is not None:
            name = os.path.basename(hive_path).replace('.hive', '')
            self.dir = f"{out_dir}/{name}"
            shutil.unpack_archive(hive_path, self.dir, 'zip')
        else:
            self.dir = dir

        if model is not None:
            self.model = model
        else:
            config = self.get_config_file()
            if config is not None:
                self.model = config['model']

    def create_config_file(self, config):
        paths = self.get_paths()

        with open(paths["HIVE_DIR_CONFIG"], 'w', encoding='UTF8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def create_labels_file(self, labels: list):
        paths = self.get_paths()
        labels_processor.generate_labels_file(labels, paths["HIVE_DIR_LABELS"])

    def get_config_file(self):
        paths = self.get_paths()

        with open(paths["HIVE_DIR_CONFIG"], 'r', encoding='UTF8') as f:
            config = json.load(f)
            return config

    def generate_dir_structure(self):
        paths = self.get_paths()

        if os.path.exists(paths["HIVE_DIR_PATH"]):
            remove_files_from_dir(paths["HIVE_DIR_PATH"])
        os.makedirs(paths["HIVE_MODEL_DIR"])
        os.makedirs(paths["HIVE_DIR_DATASET"])
        os.makedirs(paths["HIVE_DIR_DATASET_IMAGES"])
        os.makedirs(paths["HIVE_DIR_DATASET_IMAGES_CROP"])
        os.makedirs(paths["HIVE_DIR_DATASET_CSVS"])
        # os.makedirs(paths["HIVE_DIR_IMAGES"])
        os.makedirs(paths['HIVE_DIR_IMAGES_TRANSFORMED'])
        os.makedirs(paths["HIVE_TRAINED"])

    def get_paths(self):
        HIVE_DIR_PATH = self.dir
        HIVE_MODEL_DIR = f"{HIVE_DIR_PATH}/{self.model}"
        HIVE_DIR_DATASET = f"{HIVE_DIR_PATH}/dataset"
        return {
            "HIVE_DIR_PATH": HIVE_DIR_PATH,
            "HIVE_MODEL_DIR": HIVE_MODEL_DIR,
            "HIVE_DIR_DATASET": HIVE_DIR_DATASET,
            "HIVE_DIR_TRAINED": f"{HIVE_DIR_PATH}/trained",
            "HIVE_DIR_DATASET_IMAGES": f"{HIVE_DIR_DATASET}/images",
            "HIVE_DIR_DATASET_IMAGES_CROP": f"{HIVE_DIR_DATASET}/crop",
            "HIVE_DIR_DATASET_CSVS": f"{HIVE_DIR_DATASET}/csvs",
            "HIVE_DIR_DATASET_CSV": f"{HIVE_DIR_DATASET}/annotations.csv",
            "HIVE_DIR_CSV": f"{HIVE_DIR_PATH}/annotations.csv",
            "HIVE_DIR_LABELS": f"{HIVE_DIR_PATH}/labels.pbtxt",
            "HIVE_DIR_TFRECORD": f"{HIVE_DIR_PATH}/record.tfrecord",
            "HIVE_DIR_IMAGES": f"{HIVE_DIR_PATH}/images",
            "HIVE_DIR_IMAGES_TRANSFORMED": f"{HIVE_DIR_PATH}/images_transformed",
            "HIVE_DIR_TRANSFORMED_CSV": f"{HIVE_DIR_PATH}/transformed.csv",
            "HIVE_DIR_CONFIG": f"{HIVE_DIR_PATH}/config.json",
            "HIVE_TRAINED": f"{HIVE_DIR_PATH}/trained",
            "HIVE_DIR_PIPELINE": f"{HIVE_MODEL_DIR}/pipeline.config",
            "HIVE_DIR_TEST_CSV": f"{HIVE_DIR_PATH}/test.csv",
            "HIVE_DIR_TEST_TFRECORD": f"{HIVE_DIR_PATH}/test.tfrecord",
            "HIVE_DIR_TRAIN_CSV": f"{HIVE_DIR_PATH}/train.csv",
            "HIVE_DIR_TRAIN_TFRECORD": f"{HIVE_DIR_PATH}/train.tfrecord",
        }

    def generate_dir(self, dir_path, ds_type, ds_path, pipeline: Pipeline):
        self.dir_path = dir_path
        paths = self.get_paths()
        self.generate_dir_structure()
        if ds_type == 'remo':
            remo.generate_dataset(f'{ds_path}/remo.json',
                                  ds_path, paths["HIVE_DIR_DATASET"])
        model_processor.download_model(self.model, self.out)
        model_processor.decompress_model(
            f"{self.out}/{self.model}.tar.gz",
            paths["HIVE_DIR_PATH"]
        )
        # Move images
        pl = pipeline.from_dir(paths["HIVE_DIR_DATASET_IMAGES"])
        pl.pipe_to_dir(paths["HIVE_DIR_IMAGES"])
        # shutil.copytree(
        #     paths["HIVE_DIR_DATASET_IMAGES"],
        #     paths["HIVE_DIR_IMAGES"]
        # )
        # Copy annotations
        shutil.copy(
            paths["HIVE_DIR_DATASET_CSV"],
            paths["HIVE_DIR_CSV"]
        )

    def generate_cropped(self, src_img_url: str, annotations: list):
        i = 1
        paths = self.get_paths()
        im = Image.open(src_img_url)
        [name, ext] = os.path.basename(src_img_url)

        for ann in annotations:
            box = ann["bbox"]
            region = ImgRegion(box["xmin"], box["ymin"],
                               box["xmax"], box["ymax"])
            region.crop_from_image(
                im, f"{paths['HIVE_DIR_DATASET_IMAGES_CROP']}/{name}_{i}_.{ext}")
            i = i + 1

    def pack_hive(self, out_path: str):
        shutil.make_archive(f"{self.dir}.hive", 'zip', self.dir)
        shutil.move(f"{self.dir}.hive.zip", out_path)

    def get_checkpoints(self):
        paths = self.get_paths()
        files = os.listdir(paths["HIVE_DIR_TRAINED"])
        filtered = []
        for f in files:
            if re.search('ckpt-[0-9]{1,3}\.i.+', f):
                filtered.append(f.replace('.index', ''))

        filtered.sort(key=natural_keys)
        print("Initial sorted: ", filtered)
        return filtered

    def get_latest_checkpoint_name(self):
        checkpoints = self.get_checkpoints()
        return checkpoints[-1]

    def get_annotations_csv_for(self, filename: str) -> pd.DataFrame:
        paths = self.get_paths()
        files = os.listdir(paths["HIVE_DIR_DATASET_CSVS"])
        for f in files:
            if filename in f:
                data = pd.read_csv(f"{paths['HIVE_DIR_DATASET_CSVS']}/{f}")
                return data

