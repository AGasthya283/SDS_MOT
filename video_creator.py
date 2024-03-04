import cv2
import os
import json

def create_video_from_images(images_dir, output_dir, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    videos = {}
    for image in data['images']:
        video_id = image['video_id']
        if video_id not in videos:
            videos[video_id] = []

        file_name = os.path.join(images_dir, image['file_name'])
        videos[video_id].append(file_name)

    for video_id, image_files in videos.items():
        output_file = os.path.join(output_dir, f"{video_id}.mp4")
        frame = cv2.imread(image_files[0])
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

        for image_file in image_files:
            frame = cv2.imread(image_file)
            out.write(frame)

        out.release()

if __name__ == "__main__":
    images_dir = "/home/agasthya/sea_drones_see_tracking/Compressed/val"
    output_dir = "/home/agasthya/sea_drones_see_tracking/video_files/val"
    json_file = "/home/agasthya/sea_drones_see_tracking/annotations/instances_val_objects_in_water.json"

    create_video_from_images(images_dir, output_dir, json_file)
