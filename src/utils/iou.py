def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) metric between two bounding boxes.

    Arguments:
    box1, box2 -- Tuples representing the coordinates of the top-left and bottom-right points of the boxes
                  in the format (x1, y1, x2, y2).

    Returns:
    iou -- IoU metric value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate coordinates of the intersection rectangle
    x_intersection_min = max(x1_min, x2_min)
    y_intersection_min = max(y1_min, y2_min)
    x_intersection_max = min(x1_max, x2_max)
    y_intersection_max = min(y1_max, y2_max)

    # Calculate area of intersection rectangle
    intersection_area = max(0, x_intersection_max - x_intersection_min + 1) * max(0, y_intersection_max - y_intersection_min + 1)

    # Calculate area of the bounding boxes
    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    # Calculate IoU metric
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def calculate_iou_for_all_detections(detected_boxes, ground_truth_boxes):
    """
    Calculate Intersection over Union (IoU) metric for all detected objects.

    Arguments:
    detected_boxes -- List of tuples representing the coordinates of the detected bounding boxes.
    ground_truth_boxes -- List of tuples representing the coordinates of the ground truth bounding boxes.

    Returns:
    iou_scores -- List of IoU metric values for each detected object.
    """
    iou_scores = []

    for detected_box in detected_boxes:
        max_iou = 0

        for gt_box in ground_truth_boxes:
            iou = calculate_iou(detected_box, gt_box)
            if iou > max_iou:
                max_iou = iou
    
        iou_scores.append(max_iou)

    return iou_scores
