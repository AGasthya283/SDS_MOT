import json
import os

def coco_to_coco8(coco_path, output_dir):
    # Load COCO annotations
    with open(coco_path, 'r') as file:
        coco_data = json.load(file)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through images
    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Get annotations for the current image
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        # Create a list to store normalized annotations
        normalized_annotations = []

        # Convert bounding box annotations to normalized form
        for annotation in annotations:
            x1, y1, w, h = annotation['bbox']
            x_center = (x1 + w / 2) / width
            y_center = (y1 + h / 2) / height
            normalized_w = w / width
            normalized_h = h / height

            normalized_annotation = {
                'category_id': annotation['category_id'],
                'bbox': [x_center, y_center, normalized_w, normalized_h]
            }

            normalized_annotations.append(normalized_annotation)

        # Save normalized annotations to a text file
        output_path = os.path.join(output_dir, f'{file_name.split(".")[0]}.txt')
        with open(output_path, 'w') as output_file:
            for annotation in normalized_annotations:
                line = f"{annotation['category_id']} {annotation['bbox'][0]} {annotation['bbox'][1]} {annotation['bbox'][2]} {annotation['bbox'][3]}\n"
                output_file.write(line)

if __name__ == "__main__":
    coco_path = "/home/agasthya/sea_drones_see_tracking/labels/instances_val_objects_in_water.json"  # Replace with your COCO annotations file path
    output_directory = "/home/agasthya/sea_drones_see_tracking/labels/val"  # Replace with the desired output directory

    coco_to_coco8(coco_path, output_directory)
