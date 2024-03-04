import json

def darklabel_to_yolov5(darklabel_path_1, darklabel_path_2, output_path):
    # Load annotations from the first Darklabel file
    with open(darklabel_path_1, 'r') as file1:
        data1 = json.load(file1)

    # Load annotations from the second Darklabel file
    with open(darklabel_path_2, 'r') as file2:
        data2 = json.load(file2)

    # Merge categories from both files
    categories = {category['id']: category['name'] for category in data1['categories']}
    categories.update({category['id']: category['name'] for category in data2['categories']})

    # Create a mapping for YOLOv5 class indices
    class_mapping = {name: i + 1 for i, name in enumerate(categories.values())}

    # Create YOLOv5 format annotations
    yolov5_annotations = []
    for data in [data1, data2]:
        for annotation in data['annotations']:
            image_info = next(image for image in data['images'] if image['id'] == annotation['image_id'])

            # YOLOv5 format: class x_center y_center width height
            x_center = (annotation['bbox'][0] + annotation['bbox'][2] / 2) / image_info['width']
            y_center = (annotation['bbox'][1] + annotation['bbox'][3] / 2) / image_info['height']
            width = annotation['bbox'][2] / image_info['width']
            height = annotation['bbox'][3] / image_info['height']

            yolov5_annotation = {
                'class': class_mapping[categories[annotation['category_id']]],
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            }

            yolov5_annotations.append(yolov5_annotation)

    # Save YOLOv5 annotations to a text file
    with open(output_path, 'w') as output_file:
        for annotation in yolov5_annotations:
            line = f"{annotation['class']} {annotation['x_center']} {annotation['y_center']} {annotation['width']} {annotation['height']}\n"
            output_file.write(line)

if __name__ == "__main__":
    darklabel_path_1 = "/home/agasthya/sea_drones_see_tracking/annotations/instances_train_swimmer.json"
    darklabel_path_2 = "/home/agasthya/sea_drones_see_tracking/annotations/instances_train_objects_in_water.json"
    output_path = "/home/agasthya/sea_drones_see_tracking/annotations_yolo.txt"

    darklabel_to_yolov5(darklabel_path_1, darklabel_path_2, output_path)
