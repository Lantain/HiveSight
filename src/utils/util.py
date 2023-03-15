import os
import cv2
import re
import tensorflow as tf
import numpy as np
from PIL import Image

def remove_files_from_dir(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            remove_files_from_dir(f"{path}/{d}")
            if os.path.isdir(f"{path}/{d}") == True:
                os.removedirs(f"{path}/{d}")
        for f in files:
            os.remove(f"{path}/{f}")
    if os.path.isdir(path) == True:
        os.removedirs(path)



def crop_annotations(source_dir, target_dir, ann, fi):
    [name, ext] = ann["file_name"].split(".") 
    im = Image.open(f"{source_dir}/{ann['file_name']}")
    
    i = 0
    for a in ann["annotations"]:
        box = a["bbox"]
        part = im.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        # print(name)
        for c in a["classes"]:
            if os.path.isdir(f"{target_dir}/{c}") == False:
                os.mkdir(f"{target_dir}/{c}")

            part.save(f"{target_dir}/{c}/{fi}_{i}.{ext}")
            
        i += 1

def image_to_np(path):
    img = cv2.imread(path, 3)
    # print(path, img)
    b,g,r = cv2.split(img)           # get b, g, r
    rgb_img = cv2.merge([r,g,b])     # switch it to r, g, b
    image_np = np.array(rgb_img)
    return image_np

def image_to_input_tensor(path):
    image_np = image_to_np(path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor

def get_last_checkpoint_name(model_dir):
    try:
        files = os.listdir(model_dir)
        filtered = []

        for f in files:
            if re.search('ckpt-[0-9]{1,2}\.i.+', f):
                filtered.append(f.replace('.index', ''))
        
        filtered.sort()

        if len(filtered) > 0:
            last_checkpoint = filtered[len(filtered) - 1]
            return last_checkpoint
    except:
        print(f"Failed to get last checkpoint: {model_dir}")

def load_models_list(path):
    with open(path) as f:
        list = f.read()
        return list.split(',')

def save_models_list(models, path):
    with open(path, 'w') as f:
        f.write(','.join(models))

def find_labels_in_dir(dir):
    pbtxt_regex = re.compile('(.*pbtxt$)')

    for root, dirs, files in os.walk(dir):
        for file in files:
            if pbtxt_regex.match(file):
                print(f"Found PBTXT at {dir}/{file}")
                return "{dir}/{file}"

def find_record_in_dir(dir, type):
    test_rec_regex = re.compile('(.*test.tfrecord$)')
    train_rec_regex = re.compile('(.*train.tfrecord$)')
    regex = None

    if type == 'test':
        regex = test_rec_regex
    elif type == 'train':
        regex = train_rec_regex

    for root, dirs, files in os.walk(dir):
        for file in files:
            if regex.match(file):
                print(f"Found {type} record at {dir}/{file}")
                return "{dir}/{file}"

def find_config_in_dir(dir):
    config_regex = re.compile('(.*config$)')
    for root, dirs, files in os.walk(dir):
        for file in files:
            if config_regex.match(file):
                print(f"Found config at {dir}/{file}")
                return "{dir}/{file}"

def find_checkpoint_in_dir(dir):
    config_regex = re.compile('(.*config$)')
    for root, dirs, files in os.walk(dir):
        for file in files:
            if config_regex.match(file):
                print(f"Found config at {dir}/{file}")
                return "{dir}/{file}"

def get_n_files_from(dir, n):
    for root, dirs, files in os.walk(dir):
        return files[0:n]
    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]