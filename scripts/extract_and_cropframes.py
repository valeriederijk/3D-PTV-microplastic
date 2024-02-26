# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:00:22 2024

@author: valer
"""

import os
import cv2
import csv

def image_crop(image, crop_params):
    x, y, width, height = map(int, crop_params.split(','))

    # Perform the image cropping using array slicing
    cropped_image = image[y:y+height, x:x+width]

    return cropped_image

def apply_crop(frame, crop_params):
    # Call the image_crop function with the frame and crop parameters
    cropped_frame = image_crop(frame, crop_params)
    
    return cropped_frame

def extract_frames(video_path, output_dir, crop_params_path, start_time=0, max_frames=None):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open the video file {video_path}")
        return

    # Set the starting frame based on the specified start time
    start_frame = int(start_time * cap.get(cv2.CAP_PROP_FPS))

    # Set the video capture object to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0

    # Read the crop parameters from the CSV file
    with open(crop_params_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        crop_params_row = next(reader)
        crop_params = f"{crop_params_row['x']},{crop_params_row['y']},{crop_params_row['width']},{crop_params_row['height']}"

    # Loop through the frames and save them as PNG images
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break

        frame_count += 1

        # Apply cropping to the frame
        cropped_frame = apply_crop(frame, crop_params)

        # Define the output file name
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")

        # Save the cropped frame as a PNG image
        cv2.imwrite(frame_filename, cropped_frame)

        # Display progress
        print(f"Extracted and cropped frame {frame_count} from {os.path.basename(video_path)}")

    # Release the video capture object
    cap.release()

    print(f"Frames extracted and cropped, saved to {output_dir}")

# List of video paths
video_paths = [
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\output\cam_left_cut.mp4"
]

# Output directories for frames
output_dirs = [
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\frames_left"
]

# Crop parameters file paths
crop_params_paths = [ 
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\crop_parameters_left.csv"
]

# Starting times for each video (in seconds)
start_times = [19]

# Maximum frames to extract for each video
max_frames_list = [100000]

# Process each video
for video_path, output_dir, crop_params_path, start_time, max_frames in zip(video_paths, output_dirs, crop_params_paths, start_times, max_frames_list):
    extract_frames(video_path, output_dir, crop_params_path, start_time, max_frames)
