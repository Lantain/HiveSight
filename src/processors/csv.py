import csv
import pandas as pd
import os
import shutil
from PIL import Image

header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
def save_rows(rows: list, out_file: str):
    with open(out_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for row in rows:
            writer.writerow(row)

def transfer_n_fields(csv_path, source_dir, target_dir, n):
    data = pd.read_csv(csv_path, nrows=n)
    for row in data:
        shutil.copyfile(f"{source_dir}/{row['filename']}", f"{target_dir}/{row['filename']}")

    return data

def dir_to_features(label, path):
    rows = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            image = Image.open(f"{path}/{f}")
            rows.append([f, label, image.width, image.height, 0, 0, image.width, image.height])
    return rows

def unique_column_values(csv_path, column) -> list:
    data = pd.read_csv(csv_path)
    return data[column].unique().tolist()
