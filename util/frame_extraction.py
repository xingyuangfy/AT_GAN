"""
Frame extraction utility for video processing
Copyright (c) xingyuangfy 2025. All rights reserved.
"""

import cv2
import os

# Define function to get paths of all images in a folder
def get_image_paths(folder_path):
    # List to store image paths
    image_paths = []
    # Get all filenames in the folder
    files = os.listdir(folder_path)
    # Iterate through filenames
    for file in files:
        # Build complete file path
        file_path = os.path.join(folder_path, file)
        # Check if file is an image (based on extension)
        if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # If it's an image file, add its path to the list
            image_paths.append(file_path)
    return image_paths

def extract_frames(video_path, output_folder, frame_rate=1):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    print('a')
    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # Calculate frame interval based on desired frame_rate
    interval_frames = int(fps / frame_rate)
    print(interval_frames)
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Check if reached end of video
        if not ret:
            break

        # Save frames according to interval
        if frame_count % interval_frames == 0:
            frame_filename = os.path.join(output_folder, f"Age_{frame_count // interval_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
        frame_count += 1

    # Release video file
    cap.release()

# # Folder path
# folder_path = '/path/to/your/folder'

# # Get all image paths in the folder
# image_paths = get_image_paths(folder_path)

# # Print all image paths
# for path in image_paths:
#     print(path)


if __name__ == "__main__":
    # Video file path
    video_path = "tmp2.mp4"
    # Output folder for saving frames
    output_folder = "output_frames"
    # Frame extraction rate, adjustable as needed
    frame_rate = 6
    # Call frame extraction function
    extract_frames(video_path, output_folder, frame_rate)
