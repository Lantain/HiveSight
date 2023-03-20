import argparse
import shutil
import os
import json
import time
import random
import pandas as pd
from pipeline import Pipeline
from src.hive_fs import HiveFs
from src.datasets import remo
from sklearn.model_selection import train_test_split
from src.processors import tf_zoo_models as model_processor
from src.processors import csv as csv_processor
from src.processors import labels as labels_processor
from src.processors import record_csv as record_processor
from src.processors import config as config_processor
# import tensorflow.compat.v2 as tf
import tensorflow as tf
from object_detection import model_lib_v2
from PIL import Image
from src.transformers.model import Transformer

OUT_PATH = "./out"

class Hive:
    fs: HiveFs
    model: str = None
    test_train_ratio: float = 0.2
    batch_size: int = None
    num_steps: int = None
    ds_type: str = None
    ds_path: str = None

    def __init__(self, fs: HiveFs):
        self.fs = fs
        self.model = fs.model

    def set_dataset(self, type, path):
        self.ds_type = type
        self.ds_path = path

    def is_transformed(self):
        paths = self.fs.get_paths()
        return os.path.exists(paths["HIVE_DIR_TRANSFORMED_CSV"])
    
    def set_train_params(self, num_steps, batch_size):
        self.batch_size = batch_size
        self.num_steps = num_steps

    def get_config(self):
        return {
            "created_at": str(time.gmtime()),
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "model": self.model,
            "dataset_type": self.ds_type,
            "dataset_path": self.ds_path,
            "test_train_ratio": self.test_train_ratio,
        }
   
    def generate_labels_file(self) -> list:
        paths = self.fs.get_paths()
        df = pd.read_csv(paths["HIVE_DIR_CSV"])
        labels = df["class"].unique().tolist()
        self.fs.create_labels_file(labels=labels)
        return labels

    def get_csv_split(self, labels: list, csv_path: str = None):
        config = self.fs.get_config_file()
        paths = self.fs.get_paths()
        test = list()
        train = list()
        df = pd.read_csv(csv_path or paths["HIVE_DIR_CSV"])
        for label in labels:
            dft = df[df["class"] == label]
            df_train, df_test = train_test_split(dft, test_size=config["test_train_ratio"])
            train.extend(df_train.values.tolist())
            test.extend(df_test.values.tolist())
        return train, test

    def generate_csv_split(self):
        paths = self.fs.get_paths()
        labels = self.generate_labels_file()
        train, test = self.get_csv_split(
            labels, 
            paths["HIVE_DIR_TRANSFORMED_CSV"] if self.is_transformed() else paths["HIVE_DIR_CSV"]
        )

        random.seed(0)
        random.shuffle(train)
        random.shuffle(test)

        print(f"Packing {len(train)} train rows")
        print(f"Packing {len(test)} test rows")

        csv_processor.save_rows(train, paths["HIVE_DIR_TRAIN_CSV"])
        csv_processor.save_rows(test, paths["HIVE_DIR_TEST_CSV"])

    def generate_records(self):
        paths = self.fs.get_paths()
        print("Generating records...")
        record_processor.create_record_csv(
            paths["HIVE_DIR_TRAIN_CSV"], 
            paths["HIVE_DIR_IMAGES_TRANSFORMED"] if self.is_transformed() else paths["HIVE_DIR_IMAGES"], 
            paths["HIVE_DIR_TRAIN_TFRECORD"], 
            paths["HIVE_DIR_LABELS"]
        )
        record_processor.create_record_csv(
            paths["HIVE_DIR_TEST_CSV"], 
            paths["HIVE_DIR_IMAGES_TRANSFORMED"] if self.is_transformed() else paths["HIVE_DIR_IMAGES"], 
            paths["HIVE_DIR_TEST_TFRECORD"], 
            paths["HIVE_DIR_LABELS"]
        )
    
    def fill_pipeline_config(self):
        paths = self.fs.get_paths()
        config = self.fs.get_config_file()

        config_processor.fill_config(
            config["model"],
            paths["HIVE_MODEL_DIR"],
            paths["HIVE_DIR_LABELS"],
            paths["HIVE_DIR_TRAIN_TFRECORD"],
            paths["HIVE_DIR_TEST_TFRECORD"],
            config["num_steps"],
            config["batch_size"],
        )

    def make(self, pipeline: Pipeline, out_dir: str = None):
        paths = self.fs.get_paths()
        self.fs.generate_dir(out_dir or self.fs.dir, ds_type=self.ds_type, ds_path=self.ds_path)
        self.fs.create_config_file(self.get_config())
        pp = pipeline.from_dir(paths["HIVE_DIR_DATASET_IMAGES_CROP"])
        pp.pipe_to_dir(paths['HIVE_DIR_IMAGES_TRANSFORMED'])
        if pipeline.has_transformers():
            rows = csv_processor.dir_to_features("Bee", paths["HIVE_DIR_IMAGES_TRANSFORMED"])
            csv_processor.save_rows(rows, paths["HIVE_DIR_TRANSFORMED_CSV"])

        self.generate_csv_split()
        self.generate_records()
        self.fill_pipeline_config()

    def train(self):
        paths = self.fs.get_paths()
        tf.config.set_soft_device_placement(True)
        strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=paths["HIVE_DIR_PIPELINE"],
                model_dir=paths["HIVE_DIR_TRAINED"],
                train_steps=self.num_steps,
                use_tpu=False,
                checkpoint_every_n=1000,
                record_summaries=True,
                checkpoint_max_to_keep=20
            )

    def pack(self, path):
        self.fs.pack_hive(path)

    def generate_inference(self, ckpt: str = None):
        paths = self.fs.get_paths()
        if ckpt is None:
            ckpt = self.fs.get_checkpoints()[-1]
        CKPT_PIPELINE_DIR = f"{paths['HIVE_DIR_PATH']}/{ckpt}"
        CKPT_CONFIG = f"{CKPT_PIPELINE_DIR}/pipeline.config"
    
        os.makedirs(CKPT_PIPELINE_DIR, exist_ok=True)

        shutil.copy(paths["HIVE_DIR_PIPELINE"], CKPT_PIPELINE_DIR)

        config_processor.set_checkpoint_value(
            CKPT_CONFIG, 
            f"{paths['HIVE_DIR_TRAINED']}/{ckpt}",
            CKPT_PIPELINE_DIR
        )
        
        config_processor.export_inference_graph(
            CKPT_CONFIG, 
            paths["HIVE_DIR_TRAINED"], 
            CKPT_PIPELINE_DIR,
            ckpt
        )
    
    def use_checkpoint(self, ckpt: str = None):
        paths = self.fs.get_paths()
        if ckpt is None:
            ckpt = self.fs.get_checkpoints()[-1]
        saved_model_path = f"{paths['HIVE_DIR_PATH']}/{ckpt}/saved_model"
        print(f"Loading {ckpt} model...", end=' ')
        model_fn = tf.saved_model.load(saved_model_path)
        print('Done!')
        return model_fn
