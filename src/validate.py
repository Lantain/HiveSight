import numpy as np

from object_detection.utils import visualization_utils as viz_utils, label_map_util
import matplotlib.pyplot as plt
from src.utils import util as utils

def get_category_index(labels_path):
    return label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

def get_detections(detect_fn, input_img_path):
    input_tensor = utils.image_to_input_tensor(input_img_path)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {
        key: value[0, :num_detections].numpy()
              for key, value in detections.items()
    }
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    print(f"Detections: {num_detections}")
    
    return detections

def show_images_pair(img_np1, img_np2):
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_np1)
    plt.axis('off')
    plt.title("First")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(img_np2)
    plt.axis('off')
    plt.title("Second")

    fig.show()

def get_processed_image(detections, labels_path, input_img_path, threshold=.2):
    category_index = get_category_index(labels_path)
    image_np_with_detections = utils.image_to_np(input_img_path)

    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=threshold, # Adjust this value to set the minimum probability boxes to be classified as True
      agnostic_mode=False)

    return image_np_with_detections