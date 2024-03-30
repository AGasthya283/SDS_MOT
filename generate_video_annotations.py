import json
import os

def generate_ground_truth(annotations, images):
    ground_truth = {}
    for annotation in annotations:
        video_id = annotation['video_id']
        if video_id not in ground_truth:
            ground_truth[video_id] = []
        
        image_id = annotation['image_id']
        frame_number = get_frame_number(image_id, images)
        object_id = annotation['track_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        
        # Format: Frame number, Object ID, Bounding box coordinates (x, y, width, height), Category ID
        line = f"{frame_number},{object_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{category_id},-1,-1,-1\n"
        
        ground_truth[video_id].append(line)
    
    return ground_truth

def get_frame_number(image_id, images):
    for image in images:
        if image['id'] == image_id:
            return image['source']['frame_no']
    return None

def save_ground_truth(ground_truth, output_dir):
    for video_id, annotations in ground_truth.items():
        file_path = os.path.join(output_dir, f"train_{video_id}_gt.txt")
        with open(file_path, "w") as f:
            f.writelines(annotations)

if __name__ == "__main__":
    input_file = "annotations/instances_train_objects_in_water.json"
    output_dir = "video_files/gt/train"
    
    with open(input_file, "r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise
    
    ground_truth = generate_ground_truth(data["annotations"], data["images"])
    save_ground_truth(ground_truth, output_dir)
