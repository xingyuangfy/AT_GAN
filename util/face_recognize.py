"""
Face recognition utility for image processing
Copyright (c) Xingyuangfy 2025. All rights reserved.
"""

import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import time

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
            image_paths.append(file)
    return image_paths

def recognize_faces_in_image(known_faces_folder, unknown_faces_folder, output_folder):
    values = []
    output_info = []
    output_infocount = []

    try:
        os.mkdir(output_folder)
    except:
        pass
    
    known_face_encodings = []
    known_face_names = []
    known_face_dic = {}

    known_faces_filenames = sorted(get_image_paths(known_faces_folder))
    # Load known face images and corresponding names
    for filename in known_faces_filenames:
        image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) > 0:
            face_encoding = face_encoding[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
            known_face_dic[os.path.splitext(filename)[0]] = os.path.join(known_faces_folder, filename)

    # Count matching occurrences
    match_counts = {}
    match_graph = {}

    unknown_faces_filenames = sorted(get_image_paths(unknown_faces_folder))
    # Load unknown face images
    for unknown_filename in unknown_faces_filenames:
        unknown_image = face_recognition.load_image_file(os.path.join(unknown_faces_folder, unknown_filename))
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if len(known_face_encodings) > 0:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                best_match_name = known_face_names[best_match_index]
                unknown_name = os.path.splitext(unknown_filename)[0]

                print(f"Unknown face {unknown_name} The known face that best matches is {best_match_name}")
                values.append(f"Unknown face {unknown_name} The known face that best matches is {best_match_name}")
                # Update match statistics
                if best_match_name in match_counts:
                    match_counts[best_match_name] += 1
                    match_graph[best_match_name].append(os.path.join(output_folder, f"output_{unknown_name}.jpg"))
                else:
                    match_counts[best_match_name] = 1
                    match_graph[best_match_name] = [os.path.join(output_folder, f"output_{unknown_name}.jpg")]

                # Draw face location and matching result on image
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                text_bbox = draw.textbbox((left + 6, bottom - 25), best_match_name)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255),
                               outline=(0, 0, 255))
                draw.text((left + 6, bottom - text_height - 5), best_match_name, fill=(255, 255, 255, 255))

            else:
                print("Failure to detect known faces") 

        del draw
        pil_image.save(os.path.join(output_folder, f"output_{unknown_name}.jpg"))

    # Find the face with the most matches
    max_match_name = max(match_counts, key=match_counts.get)
    output_info.append(max_match_name)
    output_infocount.append(match_counts[max_match_name])
    print(f"The face with the highest number of matches is {max_match_name}, \nmatches {match_counts[max_match_name]} time(s)")

    match_res = sorted([(known_face_dic[k], v) for k, v in match_graph.items()], key=lambda x: -len(x[1]))

    return match_res

if __name__ == "__main__":
    print(recognize_faces_in_image("face_samples", "output_frames", "output_recognize"))
    
