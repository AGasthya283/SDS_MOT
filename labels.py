import os
from PIL import Image

def create_annotation_files(annotations_file_path, output_directory):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the annotations file
    with open(annotations_file_path, 'r') as file:
        lines = file.readlines()

    current_image_name = None
    current_annotations = []

    # Process each line in the annotations file
    for line in lines:
        parts = line.strip().split(',')
        image_name = parts[0]
        annotation = ','.join(parts[1:])

        # Check if a new image is encountered
        if current_image_name is None or image_name != current_image_name:
            # Write current annotations to a text file
            if current_image_name is not None:
                output_file_path = os.path.join(output_directory, f'{current_image_name[:-4]}.txt')
                with open(output_file_path, 'w') as output_file:
                    output_file.writelines(current_annotations)

            # Initialize new image annotations
            current_image_name = image_name
            current_annotations = []

        # Add the current annotation to the list
        current_annotations.append(annotation + '\n')

    # Write annotations for the last image
    if current_image_name is not None:
        output_file_path = os.path.join(output_directory, f'{current_image_name[:-4]}.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(current_annotations)

def convert_to_yolov5_format(input_file, output_file, image_folder):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as out_file:
        for line in lines:
            parts = line.strip().split(',')
            image_name = parts[0]
            class_id = int(parts[1])
            x1, y1, w, h = map(int, parts[3:7])

            # Convert to YOLOv5 format (normalized coordinates)
            image_path = os.path.join(image_folder, image_name)
            image_width, image_height = get_image_dimensions(image_path)

            x_center = (x1 + w / 2) / image_width
            y_center = (y1 + h / 2) / image_height
            width_normalized = w / image_width
            height_normalized = h / image_height

            # Write to the output file
            out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_normalized:.6f} {height_normalized:.6f}\n")

def get_image_dimensions(image_path):
    image = Image.open(image_path)
    return image.size

# Replace the placeholders with your actual paths
annotations_file_path = '/home/agasthya/sea_drones_see_tracking/annotations_txt_format/instances_val_objects_in_water.txt'
output_directory = '/home/agasthya/sea_drones_see_tracking/annotaions_yolov5_pytorch_txt/val'
image_folder = '/home/agasthya/sea_drones_see_tracking/Compressed/val'

create_annotation_files(annotations_file_path, output_directory)
convert_to_yolov5_format(annotations_file_path, os.path.join(output_directory, 'yolov5_annotations.txt'), image_folder)
