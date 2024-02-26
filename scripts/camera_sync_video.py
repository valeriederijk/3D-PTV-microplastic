# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:33:43 2023

@author: valer
"""

import cv2
import os

def save_frame_and_video(cap, output_folder, frame_number):
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the selected frame
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error reading the frame.")
        return

    # Save the frame as an image
    frame_filename = f"frame_{frame_number}.png"
    frame_path = os.path.join(output_folder, frame_filename)
    cv2.imwrite(frame_path, frame)
    print(f"Frame {frame_number} saved as {frame_path}")

    # Create a video writer starting from the selected frame
    output_video_path = os.path.join(output_folder, "cam_right_cut.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print (f"Video saved as {output_video_path}")

    # Write frames to the output video
    while frame_number < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        print (f"printing frame {frame_number}")
        frame_number += 1

    # Release video writer object
    out.release()

    print(f"Video starting from frame {frame_number} saved as {output_video_path}")

def synchronize_videos(video_path, output_folder, start_frame=0):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    frame_number = start_frame

    while True:
        # Set the frame number before reading the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frames
        ret, frame = cap.read()

        # Check for the end of the video
        if not ret:
            break

        # Display the frame
        cv2.imshow("Video", frame)

        # Print the frame number
        print(f"Frame Number: {frame_number}")

        # Wait for key press (0 means wait indefinitely)
        key = cv2.waitKey(0)

        # If 's' is pressed, save the frame and create a new video
        if key == ord('s'):
            save_frame_and_video(cap, output_folder, frame_number)
            cv2.destroyAllWindows()  # Close the OpenCV window
            break

        # If 'a' is pressed, move backward
        elif key == ord('a'):
            frame_number -= 1

        # If 'd' is pressed, move forward
        elif key == ord('d'):
            frame_number += 1
        
        elif key == ord('w'):
            frame_number += 5

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break
        

        # Ensure the frame number is within valid range
        frame_number = max(0, min(frame_number, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))

    # Release video capture object
    cap.release()

if __name__ == "__main__":
    #add your video file and output 
    video_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\videos\cam_right.MOV"
    output_folder = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\output"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the starting frame (you can change this value)/easy to calculate a frame close to the clap to reduced time
    start_frame = 300

    synchronize_videos(video_path, output_folder, start_frame)
