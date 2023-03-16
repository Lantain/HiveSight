import os
import json
import shutil
from src.processors import csv as csv_processor, labels as labels_processor
from PIL import Image

OUT_DIR = "./out"
ASSETS_DIR = "./assets"
SOURCE_DIR = "./source"

def get_labels_list(ann) -> list:
    labels = list()
    
    for a in ann[0]["annotations"]:
        for class_name in a['classes']:
            if class_name not in labels:
                labels.append(class_name)
    return labels

def annotation_to_rows(ann) -> list:
    rows = list()
    height = ann["height"]
    width = ann["width"]
    filename = ann['file_name']

    for a in ann["annotations"]:
        xmin = a['bbox']['xmin']
        xmax = a['bbox']['xmax']
        ymin = a['bbox']['ymin']
        ymax = a['bbox']['ymax']

        for class_name in a['classes']:
            row = [filename, class_name, width, height, xmin, ymin, xmax, ymax]
            rows.append(row)
    
    return rows

def generate_csv_from_annotation(ann, out_dir):
    filename = ann['file_name']
    [name, ext] = filename.split(".") 
    rows = annotation_to_rows(ann)
    csv_processor.save_rows(rows, f"{out_dir}/{name}.csv")

def generate_csv_from_annotation_set(anns, out_file):
    rows = list()
    for ann in anns:
        rs = annotation_to_rows(ann)
        rows.extend(rs)
    csv_processor.save_rows(rows, out_file)

def fs_prepare(path):
    os.mkdir(path)
    os.mkdir(f'{path}/images')
    os.mkdir(f'{path}/crop')
    os.mkdir(f'{path}/csvs')

def copy_source_images(data, source_images_dir, out_dir):
    for a in data:
        shutil.copyfile(f"{source_images_dir}/images/{a['file_name']}", f"{out_dir}/images/{a['file_name']}")
        crop_annotations(f"{source_images_dir}/images", f'{out_dir}/crop', a)

def crop_annotations(source_dir, target_dir, ann):
    [name, ext] = ann["file_name"].split(".") 
    im = Image.open(f"{source_dir}/{ann['file_name']}")
    
    i = 0
    for a in ann["annotations"]:
        box = a["bbox"]
        part = im.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        for c in a["classes"]:
            if os.path.isdir(f"{target_dir}/{c}") == False:
                os.mkdir(f"{target_dir}/{c}")

            part.save(f"{target_dir}/{c}/{name}_{i}.{ext}")
            
        i += 1

def generate_dataset(remo_json, source_images_dir, out_dir):
    f = open(remo_json)
    data = json.load(f)

    # os.makedirs(f"{out_dir}/csvs")
    # os.makedirs(f"{out_dir}/images")
    # os.makedirs(f"{out_dir}/crop")

    copy_source_images(data, source_images_dir, out_dir)
    for a in data:
        generate_csv_from_annotation(a, f'{out_dir}/csvs')

    generate_csv_from_annotation_set(data, f'{out_dir}/annotations.csv')
    labels = get_labels_list(data)
    labels_processor.generate_labels_file(labels, f'{out_dir}/labels.pbtxt')
