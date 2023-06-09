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
from src import validate 
import time
from src.utils import iou as iou
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
            try:
                dft = df[df["class"] == label]
                df_train, df_test = train_test_split(dft, test_size=config["test_train_ratio"])
                train.extend(df_train.values.tolist())
                test.extend(df_test.values.tolist())
            except:
                print("huh")
        return train, test

    def generate_csv_split(self):
        paths = self.fs.get_paths()
        labels = self.generate_labels_file()
        # csv = paths["HIVE_DIR_TRANSFORMED_CSV"] if self.is_transformed() else paths["HIVE_DIR_CSV"]
        csv = paths["HIVE_DIR_CSV"]
        train, test = self.get_csv_split(labels, csv)

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
        img_dir = paths["HIVE_DIR_IMAGES_TRANSFORMED"] if self.is_transformed() else paths["HIVE_DIR_IMAGES"]
        record_processor.create_record_csv(
            paths["HIVE_DIR_TRAIN_CSV"], 
            img_dir, 
            paths["HIVE_DIR_TRAIN_TFRECORD"], 
            paths["HIVE_DIR_LABELS"]
        )
        record_processor.create_record_csv(
            paths["HIVE_DIR_TEST_CSV"], 
            img_dir, 
            paths["HIVE_DIR_TEST_TFRECORD"], 
            paths["HIVE_DIR_LABELS"]
        )
    
    def fill_pipeline_config(self, configurators: list[Configurator]):
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
            configurators
        )

    def make(self, pipeline: Pipeline, configurators: list[Configurator] = list(), out_dir: str = None):
        # paths = self.fs.get_paths()
        self.fs.generate_dir(out_dir or self.fs.dir, ds_type=self.ds_type, ds_path=self.ds_path, pipeline=pipeline)
        self.fs.create_config_file(self.get_config())
        # pp = pipeline.from_dir(paths["HIVE_DIR_IMAGES"])
        # pp.pipe_to_dir(paths['HIVE_DIR_IMAGES_TRANSFORMED'])
        # if pipeline.has_transformers():
            # rows = csv_processor.dir_to_features("Bee", paths["HIVE_DIR_IMAGES_TRANSFORMED"])
            # csv_processor.save_rows(rows, paths["HIVE_DIR_TRANSFORMED_CSV"])

        self.generate_csv_split()
        self.generate_records()
        self.fill_pipeline_config(configurators)

    def train(self, max_checkpoints=5):
        paths = self.fs.get_paths()
        tf.config.set_soft_device_placement(True)
        # tf.debugging.set_log_device_placement(True)
        strategy = tf.compat.v2.distribute.MirroredStrategy()
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print(tf.config.list_physical_devices('GPU'))
        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=paths["HIVE_DIR_PIPELINE"],
                model_dir=paths["HIVE_DIR_TRAINED"],
                train_steps=self.num_steps,
                use_tpu=False,
                checkpoint_every_n=1000,
                record_summaries=True,
                checkpoint_max_to_keep=max_checkpoints
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

    def analyze(self, img_paths: list[str], threshold=.3, eval_mapping = {}):
        paths = self.fs.get_paths()
        out_dir = f"{paths['HIVE_DIR_PATH']}/analyze"
        os.makedirs(out_dir, exist_ok=True)
        ckpt = self.fs.get_checkpoints()[-1]
        saved_model_path = f"{paths['HIVE_DIR_PATH']}/{ckpt}/saved_model"
        ious = list()
        model_fn = tf.saved_model.load(saved_model_path)
        for ip in img_paths:
            basename = os.path.basename(ip)
            detections = validate.get_detections(model_fn, ip)
            thresholds = list([0.1, 0.3, threshold])
            threshold_imgs = list()
            img = cv2.imread(ip, 3)
            
            for t in thresholds:
                img_detections = validate.get_processed_image(detections, paths['HIVE_DIR_LABELS'], ip, t)
                threshold_imgs.append(img_detections)
                
            true_detections_count = 0
            for score in detections['detection_scores']:
                if score > threshold:
                    true_detections_count += 1
            
            iou_score = None    
            if eval_mapping and eval_mapping[basename]:
                detected_boxes = list()
                
                for i in range(len(detections['detection_boxes'])):
                    if detections['detection_scores'][i] >= threshold:
                        box = tuple(detections['detection_boxes'][i].tolist())
                        ymin, xmin, ymax, xmax = box
                        h, w, c = img.shape
                        detected_boxes.append((xmin * w, ymin * h, xmax * w, ymax * h))
                        
                iou_scores = iou.calculate_iou_for_all_detections(detected_boxes, eval_mapping[basename])
                iou_score = 0
                for score in iou_scores:
                    if score > 0:
                        iou_score = iou_score + (score / len(iou_scores))
                        
            if iou_score is not None:
                ious.append(iou_score)
            print(f"{basename} - IoU: {round(iou_score, 2)}")
            b,g,r = cv2.split(img)           # get b, g, r
            rgb_img = cv2.merge([r,g,b])     # switch it to r, g, b

            fig = plt.figure(figsize=(20, 14))
            fig.suptitle(f"{ckpt} - {ip}", fontsize=16)

            rows = 2
            columns = 2

            fig.add_subplot(rows, columns, 1)
            plt.imshow(threshold_imgs[0])
            plt.axis('off')
            plt.title(f"{str(thresholds[0])} Threshold")
            
            fig.add_subplot(rows, columns, 2)
            plt.imshow(threshold_imgs[1])
            plt.axis('off')
            plt.title(f"{str(thresholds[1])} Threshold")
            
            
            fig.add_subplot(rows, columns, 3)
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.title("Original")

            fig.add_subplot(rows, columns, 4)
            plt.imshow(threshold_imgs[len(threshold_imgs) - 1])
            plt.axis('off')
            plt.title(f"{threshold}: {true_detections_count} Detections. IoU: {round(iou_score, 2)}")
            plt.savefig(f"{out_dir}/{ckpt}--{basename}.png")
            plt.close()
        # print(f"Global IoUs: {ious}")
        iou_value = 0
        for i in ious:
            iou_value = iou_value + i / len(ious)
        print(f"Global IoUs: {iou_value}")

    def evaluate(self):
        paths = self.fs.get_paths()
        self.fill_pipeline_config([])
        model_lib_v2.eval_continuously(
            pipeline_config_path=paths['HIVE_DIR_PIPELINE'],
            model_dir=paths['HIVE_MODEL_DIR'],
            checkpoint_dir=paths['HIVE_TRAINED'],
        )
    
    def monitor(self, out_dir: str, img_paths: list[str]):
        paths = self.fs.get_paths()
        ckpt = self.fs.get_checkpoints()[-1]
        saved_model_path = f"{paths['HIVE_DIR_PATH']}/{ckpt}/saved_model"
        model_fn = tf.saved_model.load(saved_model_path)
        
        while True:
            print("Checkin@")
            for ip in img_paths:
                basename = os.path.basename(ip)
                detections = validate.get_detections(model_fn, ip)
                thresholds = list([.1, .3, .5])
                threshold_imgs = list()
                
                for t in thresholds:
                    img_detections = validate.get_processed_image(detections, paths['HIVE_DIR_LABELS'], ip, t)
                    threshold_imgs.append(img_detections)
                    
                true_detections_count = 0
                for score in detections['detection_scores']:
                    if score > .5:
                        true_detections_count += 1
                print(f"Found {true_detections_count}!")
                img = cv2.imread(ip, 3)
                b,g,r = cv2.split(img)           # get b, g, r
                rgb_img = cv2.merge([r,g,b])     # switch it to r, g, b

                fig = plt.figure(figsize=(20, 14))
                fig.suptitle(f"{ckpt} - {ip}", fontsize=16)

                rows = 2
                columns = 2

                fig.add_subplot(rows, columns, 1)
                plt.imshow(threshold_imgs[0])
                plt.axis('off')
                plt.title(f"{str(thresholds[0])} Threshold")
                
                fig.add_subplot(rows, columns, 2)
                plt.imshow(threshold_imgs[1])
                plt.axis('off')
                plt.title(f"{str(thresholds[1])} Threshold")
                
                
                fig.add_subplot(rows, columns, 3)
                plt.imshow(rgb_img)
                plt.axis('off')
                plt.title("Original")

                fig.add_subplot(rows, columns, 4)
                plt.imshow(threshold_imgs[len(threshold_imgs) - 1])
                plt.axis('off')
                plt.title(f"{true_detections_count} Detections")
                plt.savefig(f"{out_dir}/{ckpt}--{basename}.png")
                plt.close()
            time.sleep(3)
